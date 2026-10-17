"""Multi-scale time-frequency prototype memory for MPGFormer."""

from typing import Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTimeFrequencyBank(nn.Module):
    """Summarize workload history with patch and frequency prototypes.

    For each patch size, the module embeds target patches and FFT magnitudes,
    retrieves prototypes, and predicts decoder padding.
    """

    def __init__(
        self,
        d_model: int,
        enc_seq_len: int,
        pred_len: int,
        patch_sizes: Sequence[int],
        num_prototypes: int,
        dropout: float,
    ):
        super().__init__()
        self.enc_seq_len = enc_seq_len
        self.patch_sizes = tuple(int(p) for p in patch_sizes)
        self.num_scales = len(self.patch_sizes)
        self.num_prototypes = num_prototypes
        self.pred_len = pred_len

        self.patch_linears = nn.ModuleList([nn.Linear(p, d_model) for p in self.patch_sizes])
        self.freq_linears = nn.ModuleList([nn.Linear(p // 2 + 1, d_model) for p in self.patch_sizes])

        self.scale_gate = nn.Linear(d_model, 1)
        self.query = nn.Linear(d_model, d_model)
        self.prototype = nn.Parameter(torch.randn(self.num_scales, num_prototypes, d_model) * 0.02)

        self.recover = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.future = nn.Linear(d_model, pred_len)

    def _patch_view(self, target_history: torch.Tensor, patch: int) -> torch.Tensor:
        """Return non-overlapping target patches, padding the tail if needed."""

        length = target_history.size(1)
        remainder = length % patch
        if remainder:
            target_history = F.pad(target_history, (0, patch - remainder), mode="replicate")

        return target_history.unfold(dimension=1, size=patch, step=patch)

    def scale_tokens(self, target_history: torch.Tensor) -> torch.Tensor:
        """Combine patch values and FFT magnitudes at each scale."""

        tokens = []
        for patch, p_linear, f_linear in zip(self.patch_sizes, self.patch_linears, self.freq_linears):
            patches = self._patch_view(target_history, patch)

            patch_token = p_linear(patches).mean(dim=1)
            freq = torch.fft.rfft(patches.float(), dim=-1).abs()
            freq_token = f_linear(freq).mean(dim=1)

            tokens.append(torch.tanh(patch_token + freq_token))

        return torch.stack(tokens, dim=1)

    def forward(self, target_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_tokens = self.scale_tokens(target_history)

        # Learned scale weights decide whether short, medium, or long rhythms dominate.
        scale_logits = self.scale_gate(scale_tokens).squeeze(-1)
        scale_weights = torch.softmax(scale_logits, dim=-1)

        query = self.query(scale_tokens)
        logits = torch.einsum("bsd,spd->bsp", query, self.prototype) / math.sqrt(query.size(-1))
        proto_weights = torch.softmax(logits, dim=-1)

        proto = torch.einsum("bsp,spd->bsd", proto_weights, self.prototype)
        proto = self.recover(proto + scale_tokens)

        context = torch.sum(proto * scale_weights.unsqueeze(-1), dim=1)
        future_padding = self.future(context)

        # Flattened weights are exposed so the router can detect prototype uncertainty.
        flat_weights = (proto_weights * scale_weights.unsqueeze(-1)).flatten(1)
        return context, future_padding, flat_weights, scale_weights
