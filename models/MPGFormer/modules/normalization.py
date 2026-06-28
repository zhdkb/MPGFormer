"""Normalization layers used by MPGFormer."""

from typing import Tuple

import torch
import torch.nn as nn


class TargetRevIN(nn.Module):
    """Reversible instance normalization applied only to the target channel.

    Cloud workloads often shift their scale between machines or time periods.
    Normalizing the target history per window keeps the encoder numerically
    stable, while the final denormalization keeps the forecast on the original
    scale expected by the loss and evaluation pipeline.
    """

    def __init__(self, target_index: int = 0, eps: float = 1e-5):
        super().__init__()
        self.target_index = target_index
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize the target channel and return statistics for restoration."""

        target = x[..., self.target_index : self.target_index + 1]
        mean = target.mean(dim=1, keepdim=True)
        std = target.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)

        # Clone before editing so callers do not observe in-place mutations.
        out = x.clone()
        out[..., self.target_index : self.target_index + 1] = (target - mean) / std
        return out, mean, std

    def denormalize_target(self, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Map point or quantile forecasts back to the original target scale."""

        mean = mean.squeeze(1).squeeze(-1).unsqueeze(-1)
        std = std.squeeze(1).squeeze(-1).unsqueeze(-1)

        # Quantile forecasts have one extra trailing dimension.
        if y.dim() == 3:
            mean = mean.unsqueeze(-1)
            std = std.unsqueeze(-1)

        return y * std + mean
