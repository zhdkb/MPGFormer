"""Losses and calibration utilities for MPGFormer."""

from collections import deque
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: Sequence[float]) -> torch.Tensor:
    """Pinball loss for ordered quantile forecasts."""

    target = target.unsqueeze(-1)
    losses = []

    for idx, q in enumerate(quantiles):
        err = target - pred[..., idx : idx + 1]
        losses.append(torch.maximum((q - 1.0) * err, q * err))

    return torch.cat(losses, dim=-1).mean()


def peak_underestimate_loss(point: torch.Tensor, target: torch.Tensor, ratio: float = 0.8) -> torch.Tensor:
    """Penalize under-prediction on high-load targets more strongly."""

    if target.numel() == 0:
        return point.new_zeros(())

    threshold = torch.quantile(target.detach().flatten(), ratio)
    peak_mask = (target >= threshold).float()
    under = F.relu(target - point)
    denom = peak_mask.sum().clamp_min(1.0)

    return (under * peak_mask).sum() / denom


def residual_tail_mse_loss(point: torch.Tensor, target: torch.Tensor, ratio: float = 0.9) -> torch.Tensor:
    """MSE over the largest residuals in the batch.

    This term is useful for sparse cloud workloads: MAE can look excellent when
    most targets are near zero, but a few missed bursts dominate raw MSE.
    """

    if target.numel() == 0:
        return point.new_zeros(())

    squared_error = (point - target).pow(2)
    abs_error = squared_error.detach().sqrt()
    threshold = torch.quantile(abs_error.flatten(), ratio)
    mask = (abs_error >= threshold).float()
    denom = mask.sum().clamp_min(1.0)

    return (squared_error * mask).sum() / denom


def target_tail_mse_loss(point: torch.Tensor, target: torch.Tensor, ratio: float = 0.9) -> torch.Tensor:
    """MSE on high-value targets, regardless of over- or under-prediction."""

    if target.numel() == 0:
        return point.new_zeros(())

    threshold = torch.quantile(target.detach().flatten(), ratio)
    mask = (target.detach() >= threshold).float()
    denom = mask.sum().clamp_min(1.0)

    return ((point - target).pow(2) * mask).sum() / denom


class MPGFormerLoss(nn.Module):
    """Composite objective used for point, interval, peak, and graph learning."""

    def __init__(
        self,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        point_weight: float = 0.9,
        quantile_weight: float = 0.3,
        contrastive_weight: float = 0.04,
        peak_weight: float = 0.2,
        mse_weight: float = 0.6,
        tail_mse_weight: float = 0.5,
        target_tail_mse_weight: float = 0.2,
        peak_ratio: float = 0.85,
        tail_mse_ratio: float = 0.9,
        target_tail_mse_ratio: float = 0.9,
    ):
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)
        self.point_weight = point_weight
        self.quantile_weight = quantile_weight
        self.contrastive_weight = contrastive_weight
        self.peak_weight = peak_weight
        self.mse_weight = mse_weight
        self.tail_mse_weight = tail_mse_weight
        self.target_tail_mse_weight = target_tail_mse_weight
        self.peak_ratio = peak_ratio
        self.tail_mse_ratio = tail_mse_ratio
        self.target_tail_mse_ratio = target_tail_mse_ratio

    def forward(self, output, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        point = output.point if hasattr(output, "point") else output["point"]
        quantiles = output.quantiles if hasattr(output, "quantiles") else output["quantiles"]
        contrastive = output.contrastive_loss if hasattr(output, "contrastive_loss") else output.get("contrastive_loss")

        if contrastive is None:
            contrastive = point.new_zeros(())

        # MAE keeps the main forecast robust, while MSE still discourages large misses.
        mae = F.l1_loss(point, target)
        mse = F.mse_loss(point, target)
        qloss = quantile_loss(quantiles, target, self.quantiles)
        ploss = peak_underestimate_loss(point, target, self.peak_ratio)
        tail_mse = residual_tail_mse_loss(point, target, self.tail_mse_ratio)
        target_tail_mse = target_tail_mse_loss(point, target, self.target_tail_mse_ratio)

        total = (
            self.point_weight * mae
            + self.mse_weight * mse
            + self.quantile_weight * qloss
            + self.peak_weight * ploss
            + self.tail_mse_weight * tail_mse
            + self.target_tail_mse_weight * target_tail_mse
            + self.contrastive_weight * contrastive
        )

        parts = {
            "loss": float(total.detach().cpu()),
            "mae": float(mae.detach().cpu()),
            "mse": float(mse.detach().cpu()),
            "quantile": float(qloss.detach().cpu()),
            "peak": float(ploss.detach().cpu()),
            "tail_mse": float(tail_mse.detach().cpu()),
            "target_tail_mse": float(target_tail_mse.detach().cpu()),
            "contrastive": float(contrastive.detach().cpu()),
        }
        return total, parts


class AdaptiveConformalCalibrator:
    """Online conformal interval radius based on recent absolute residuals."""

    def __init__(self, alpha: float = 0.1, window_size: int = 128, gamma: float = 0.005):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.effective_alpha = float(alpha)
        self.residuals = deque(maxlen=int(window_size))

    def reset(self) -> None:
        """Clear residual history and restore the target miscoverage rate."""

        self.effective_alpha = self.alpha
        self.residuals.clear()

    def update(self, point: torch.Tensor, target: torch.Tensor, lower: Optional[torch.Tensor] = None, upper: Optional[torch.Tensor] = None) -> None:
        """Add new residuals and optionally adapt alpha from interval misses."""

        point_cpu = point.detach().flatten().cpu()
        target_cpu = target.detach().flatten().cpu()
        residual = (point_cpu - target_cpu).abs()

        for value in residual.tolist():
            self.residuals.append(float(value))

        if lower is not None and upper is not None:
            missed = ((target < lower) | (target > upper)).float().mean().item()
            self.effective_alpha = min(1.0, max(1e-4, self.effective_alpha + self.gamma * (self.alpha - missed)))

    def radius(self, device=None, dtype=None) -> torch.Tensor:
        """Return the current conformal radius on the requested device/dtype."""

        if not self.residuals:
            return torch.tensor(0.0, device=device, dtype=dtype)

        values = torch.tensor(list(self.residuals), device=device, dtype=dtype)
        q = 1.0 - self.effective_alpha
        return torch.quantile(values, q)

    def interval(self, point: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand point predictions into symmetric conformal intervals."""

        r = self.radius(device=point.device, dtype=point.dtype)
        return point - r, point + r
