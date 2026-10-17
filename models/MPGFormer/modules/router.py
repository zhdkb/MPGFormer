"""Drift-aware expert routing for MPGFormer."""

from typing import Tuple

import math
import torch
import torch.nn as nn


class DriftAwareRouter(nn.Module):
    """Weight temporal, graph, and prototype branches per window."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(d_model * 3 + 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.drift_head = nn.Sequential(nn.Linear(4, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))

    def forward(
        self,
        time_context: torch.Tensor,
        graph_context: torch.Tensor,
        prototype_context: torch.Tensor,
        target_history: torch.Tensor,
        prototype_weights: torch.Tensor,
        active_experts=(True, True, True),
        use_prototype_entropy: bool = True,
        use_drift_bias: bool = True,
        fixed_average: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route temporal, graph, and prototype outputs using raw target history."""
        diff = target_history[:, 1:] - target_history[:, :-1]

        # High entropy means the prototype bank is uncertain about the current pattern.
        if use_prototype_entropy:
            entropy = -(prototype_weights * prototype_weights.clamp_min(1e-8).log()).sum(dim=-1)
            entropy = entropy / math.log(max(2, prototype_weights.size(-1)))
        else:
            entropy = target_history.new_zeros(target_history.size(0))

        drift_features = torch.stack(
            [
                (target_history[:, -1] - target_history.mean(dim=1)).abs()
                / target_history.std(dim=1, unbiased=False).clamp_min(1e-5),
                diff.abs().mean(dim=1),
                target_history.std(dim=1, unbiased=False),
                entropy,
            ],
            dim=-1,
        )

        logits = self.router(torch.cat([time_context, graph_context, prototype_context, drift_features], dim=-1))
        drift_score = torch.sigmoid(self.drift_head(drift_features)).squeeze(-1)

        if use_drift_bias:
            logits = logits + torch.stack([1.0 - drift_score, drift_score, torch.zeros_like(drift_score)], dim=-1)
        active = torch.as_tensor(active_experts, device=logits.device, dtype=torch.bool)
        if fixed_average:
            weights = active.to(logits.dtype).expand_as(logits)
            return weights / weights.sum(dim=-1, keepdim=True), drift_score
        logits = logits.masked_fill(~active, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1), drift_score
