"""Typed outputs returned by MPGFormer.

The training code can still use the point forecast directly, while analysis
code can inspect the uncertainty interval, routing weights, drift score, and
graph/prototype diagnostics from the same object.
"""

from dataclasses import dataclass

import torch


@dataclass
class MPGFormerOutput:
    """Container for all predictions and auxiliary signals.

    Attributes:
        point: Main point forecast with shape ``[batch, pred_len]``.
        quantiles: Ordered quantile forecasts with shape ``[batch, pred_len, 3]``.
        lower: Lower interval bound, currently the first quantile.
        upper: Upper interval bound, currently the last quantile.
        router_weights: Mixture weights for temporal, graph, and prototype experts.
        drift_score: Drift probability estimated from recent workload statistics.
        contrastive_loss: Auxiliary graph-view alignment loss.
        prototype_weights: Flattened multi-scale prototype assignments.
        graph_context: Batch-level adaptive graph used for diagnostics.
    """

    point: torch.Tensor
    quantiles: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    router_weights: torch.Tensor
    drift_score: torch.Tensor
    contrastive_loss: torch.Tensor
    prototype_weights: torch.Tensor
    graph_context: torch.Tensor
