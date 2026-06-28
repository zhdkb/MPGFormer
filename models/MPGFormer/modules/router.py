"""Drift-aware expert routing for MPGFormer."""

from typing import Tuple

import math
import torch
import torch.nn as nn


class DriftAwareRouter(nn.Module):
    """Route between temporal, graph, and prototype experts.

    The router sees both representation vectors and simple drift features. This
    keeps the mixture adaptive: stable workloads can lean on temporal memory,
    while abrupt or uncertain windows can allocate more weight to graph/prototype
    context.
    """

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        diff = target_history[:, 1:] - target_history[:, :-1]

        # High entropy means the prototype bank is uncertain about the current pattern.
        entropy = -(prototype_weights * prototype_weights.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / math.log(max(2, prototype_weights.size(-1)))

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

        # Bias the first two experts according to estimated drift severity.
        logits = logits + torch.stack([1.0 - drift_score, drift_score, torch.zeros_like(drift_score)], dim=-1)
        return torch.softmax(logits, dim=-1), drift_score
