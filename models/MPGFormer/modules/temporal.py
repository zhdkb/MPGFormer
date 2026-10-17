"""Temporal feature extraction layers for MPGFormer."""

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Classic sinusoidal position encoding for sequence order awareness."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # The buffer follows the module device but is not saved as a parameter.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MovingAverageDecomposition(nn.Module):
    """Separate a sequence into seasonal residual and smooth trend parts."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()

        # Keep the average centered around every timestamp.
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad = (self.kernel_size - 1) // 2
        xt = x.transpose(1, 2)
        xt = F.pad(xt, (pad, pad), mode="replicate")

        trend = F.avg_pool1d(xt, kernel_size=self.kernel_size, stride=1).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class StaticContextSelector(nn.Module):
    """Inject static machine/context features into temporal tokens when present."""

    def __init__(self, d_model: int, static_dim: int, dropout: float):
        super().__init__()
        self.enabled = static_dim > 0

        if self.enabled:
            self.query = nn.Linear(d_model, d_model)
            self.value = nn.Linear(static_dim, d_model)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, static_context: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.enabled or static_context is None:
            return x

        # Query-dependent gating lets each timestamp use a different static bias.
        score = torch.softmax(self.query(x), dim=-1)
        value = self.value(static_context).unsqueeze(1)
        return self.norm(x + self.dropout(score * value))


class NonStationaryAttentionLayer(nn.Module):
    """Self-attention layer modulated by target-window statistics.

    A positive per-head scale is generated from pre-RevIN target statistics.
    Queries, keys, and values still come from the normalized history states.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, moving_avg: int,
                 use_temperature: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_temperature = use_temperature

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

        # Produce one positive inverse-temperature scale for each attention head.
        self.factor = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_heads),
        )

        self.decomp = MovingAverageDecomposition(moving_avg)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def raw_window_statistics(raw_target_history: torch.Tensor) -> torch.Tensor:
        """Mean, scale, least-squares slope, and mean absolute first difference."""
        if raw_target_history.ndim != 2 or raw_target_history.size(1) < 2:
            raise ValueError("raw_target_history must have shape [batch, length >= 2]")
        length = raw_target_history.size(1)
        time = torch.arange(length, device=raw_target_history.device,
                            dtype=raw_target_history.dtype)
        centered_time = time - time.mean()
        mean = raw_target_history.mean(dim=1)
        scale = raw_target_history.std(dim=1, unbiased=False).clamp_min(1e-5)
        slope = ((raw_target_history - mean[:, None]) * centered_time).sum(dim=1)
        slope = slope / centered_time.square().sum()
        first_difference = (raw_target_history[:, 1:] - raw_target_history[:, :-1]).abs().mean(dim=1)
        return torch.stack((mean, scale, slope, first_difference), dim=-1)

    def forward(self, x: torch.Tensor, raw_target_history: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        if self.use_temperature:
            stats = self.raw_window_statistics(raw_target_history)
            tau = F.softplus(self.factor(stats)).view(bsz, self.n_heads, 1, 1) + 1e-3
        else:
            tau = x.new_ones(bsz, self.n_heads, 1, 1)

        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = self.attention_weights(q, k, tau)

        context = torch.matmul(self.dropout(attn), v).transpose(1, 2).contiguous()
        context = context.view(bsz, seq_len, self.d_model)

        # Trend information is retained as a small residual bias.
        seasonal, trend = self.decomp(x)
        x = self.norm1(x + self.dropout(self.out(context)) + 0.1 * trend)
        x = self.norm2(x + self.dropout(self.ff(seasonal + x)))
        return x

    @staticmethod
    def attention_weights(q: torch.Tensor, k: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Equation (7): softmax of positive head temperature times scaled QK."""
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        return torch.softmax(tau * logits, dim=-1)


class ShortTermTemporalEncoder(nn.Module):
    """Capture short bursts with convolutions and recurrent state."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.gru = nn.GRU(d_model, d_model // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_in = x.transpose(1, 2)
        local = torch.tanh(self.conv3(conv_in)) + torch.tanh(self.conv5(conv_in))
        local = local.transpose(1, 2)

        recurrent, _ = self.gru(x)
        out = self.proj(torch.cat([local, recurrent], dim=-1))
        return self.norm(x + self.dropout(out))
