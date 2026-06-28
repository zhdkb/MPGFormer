import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class ComplexDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or self.dropout <= 0:
            return x
        real = nn.functional.dropout(x.real, self.dropout, training=True)
        imag = nn.functional.dropout(x.imag, self.dropout, training=True)
        return torch.complex(real, imag)


class TFEGRU(nn.Module):
    """TFEGRU core adapted to this project's workload tensor format.

    The implementation keeps the paper's useful pieces: a time-frequency
    enhancement block, channel-independent processing with channel embeddings,
    GRU temporal modeling, and multi-head attention over recurrent states.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        input_size,
        hidden_size=128,
        dropout=0.5,
        decomp_kernel=3,
        n_heads=None,
    ):
        super().__init__()
        if hidden_size < 64:
            raise ValueError("hidden_size must be at least 64 for multi-head attention.")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.freq_bins = seq_len // 2 + 1

        self.decomp = SeriesDecomp(decomp_kernel)
        self.freq_filter = nn.Parameter(torch.ones(self.freq_bins, dtype=torch.complex64))
        self.freq_bias = nn.Parameter(torch.zeros(self.freq_bins, dtype=torch.complex64))
        self.freq_dropout = ComplexDropout(dropout)

        self.channel_embedding = nn.Embedding(input_size, hidden_size)
        self.value_embedding = nn.Sequential(nn.Linear(1, hidden_size), nn.ReLU())
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)

        heads = n_heads or max(1, hidden_size // 64)
        self.attention = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.feature_dropout = nn.Dropout(0.05)
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, pred_len),
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("TFEGRU expects input with shape [batch, seq_len, input_size].")
        if x.shape[1] != self.seq_len or x.shape[2] != self.input_size:
            raise ValueError(
                "Input shape mismatch: got {}, expected [batch, {}, {}].".format(
                    tuple(x.shape), self.seq_len, self.input_size
                )
            )

        batch_size = x.shape[0]
        last = x[:, -1:, :]
        centered = x - last
        seasonal, trend = self.decomp(centered)

        # Frequency enhancement in the rFFT domain, then return to time domain.
        freq = torch.fft.rfft(seasonal.permute(0, 2, 1), dim=-1)
        freq = freq * self.freq_filter.view(1, 1, -1) + self.freq_bias.view(1, 1, -1)
        freq = self.freq_dropout(freq)
        enhanced = torch.fft.irfft(freq, n=self.seq_len, dim=-1).permute(0, 2, 1)
        enhanced = enhanced + seasonal + trend

        channel_ids = torch.arange(self.input_size, device=x.device)
        channel_emb = self.channel_embedding(channel_ids).unsqueeze(0).expand(batch_size, -1, -1)

        hidden = enhanced.permute(0, 2, 1).reshape(batch_size * self.input_size, self.seq_len, 1)
        hidden = self.value_embedding(hidden)
        hidden = self.feature_dropout(self.activation(self.conv(hidden.permute(0, 2, 1)))).permute(0, 2, 1)

        recurrent_out, recurrent_state = self.rnn(hidden)
        query = recurrent_state.permute(1, 0, 2)
        attn_out, _ = self.attention(query, recurrent_out, recurrent_out)
        attn_out = (attn_out * query).reshape(batch_size, self.input_size, self.hidden_size)

        fused = torch.cat([channel_emb, attn_out], dim=-1)
        pred = self.output(fused).permute(0, 2, 1)
        return pred + last
