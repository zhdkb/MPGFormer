"""Dynamic graph and contrastive context modules for MPGFormer."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """A light residual graph convolution over batch-level workload nodes."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.matmul(adj, x)
        return self.norm(x + self.dropout(F.gelu(self.linear(h))))


class DynamicGraphContrastiveEncoder(nn.Module):
    """Build two adaptive graphs and align their encoded views.

    The workload graph is based on target-history similarity. The context graph
    uses static machine features when available, otherwise compact state
    statistics. A symmetric contrastive loss encourages both graph views to
    represent the same series consistently.
    """

    def __init__(self, d_model: int, static_dim: int, dropout: float, topk: int = 8, temperature: float = 0.5):
        super().__init__()
        if topk < 1:
            raise ValueError("topk must be positive")
        self.topk = topk
        self.temperature = temperature
        self.static_dim = static_dim

        self.static_proj = nn.Linear(static_dim, d_model) if static_dim > 0 else None
        self.state_proj = nn.Linear(4, d_model)
        self.gcn_workload = GraphConvolution(d_model, dropout)
        self.gcn_context = GraphConvolution(d_model, dropout)
        self.output = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def _similarity_graph(self, features: torch.Tensor) -> torch.Tensor:
        """Create a row-normalized top-k cosine graph inside the current batch."""

        bsz = features.size(0)
        sim = torch.matmul(F.normalize(features, dim=-1), F.normalize(features, dim=-1).transpose(0, 1))
        sim = sim.clamp_min(0.0)

        if bsz > 1:
            # Select exactly k other windows. Thresholding includes every tied
            # zero similarity and can turn a sparse graph into a dense one.
            candidates = sim.masked_fill(torch.eye(bsz, device=sim.device, dtype=torch.bool), -1.0)
            neighbors = candidates.topk(k=min(self.topk, bsz - 1), dim=-1).indices
            adj = torch.zeros_like(sim).scatter(1, neighbors, sim.gather(1, neighbors))
        else:
            adj = torch.zeros_like(sim)

        # Add self loops and normalize to make graph aggregation scale-stable.
        adj = adj + torch.eye(bsz, device=features.device, dtype=features.dtype)
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adj

    def _state_features(self, target_history: torch.Tensor) -> torch.Tensor:
        """Fallback context features used when static metadata is unavailable."""

        diff = target_history[:, 1:] - target_history[:, :-1]
        return torch.stack(
            [
                target_history.mean(dim=1),
                target_history.std(dim=1, unbiased=False),
                diff.abs().mean(dim=1),
                target_history[:, -1] - target_history[:, 0],
            ],
            dim=-1,
        )

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Symmetric in-batch contrastive loss between two graph views."""

        bsz = z1.size(0)
        if bsz <= 1:
            return z1.new_zeros(())

        q1 = F.normalize(z1, dim=-1)
        q2 = F.normalize(z2, dim=-1)
        logits = torch.matmul(q1, q2.transpose(0, 1)) / self.temperature
        labels = torch.arange(bsz, device=z1.device)

        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))

    def forward(
        self,
        temporal_summary: torch.Tensor,
        target_history: torch.Tensor,
        static_context: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        workload_adj = self._similarity_graph(target_history)

        if static_context is not None and self.static_proj is not None:
            context_seed = self.static_proj(static_context.float())
        else:
            context_seed = self.state_proj(self._state_features(target_history))

        context_adj = self._similarity_graph(context_seed)
        z1 = self.gcn_workload(temporal_summary, workload_adj)
        z2 = self.gcn_context(temporal_summary, context_adj)

        loss = self._contrastive_loss(z1, z2)
        graph_context = self.output(torch.cat([z1, z2], dim=-1))
        return graph_context, loss, 0.5 * (workload_adj + context_adj)
