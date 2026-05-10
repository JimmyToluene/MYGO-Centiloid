"""Loss functions for Amyloid PET Centiloid Prediction."""

import torch
import torch.nn as nn


class CentiloidLoss(nn.Module):
    """Combined Huber + Pearson correlation loss."""

    def __init__(self, delta: float = 25.0, alpha: float = 0.7):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction="mean")
        self.alpha = alpha

    def pearson_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """1 − Pearson r  (0 = perfect correlation, 2 = perfect anti-correlation)."""
        vx = pred   - pred.mean()
        vy = target - target.mean()
        r  = (vx * vy).sum() / (vx.norm() * vy.norm() + 1e-8)
        return 1.0 - r

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze()
        return (
            self.alpha       * self.huber(pred, target)
            + (1 - self.alpha) * self.pearson_loss(pred, target)
        )


def get_criterion(name: str = "centiloid", **kwargs) -> nn.Module:
    """
    Loss factory.

    Args:
        name: "centiloid" (recommended) | "huber" | "mse" | "mae"
        **kwargs: forwarded to the constructor (e.g. delta=, alpha=).
    """
    registry = {
        "centiloid": CentiloidLoss,
        "huber":     lambda **kw: nn.HuberLoss(delta=kw.get("delta", 25.0)),
        "mse":       lambda **kw: nn.MSELoss(),
        "mae":       lambda **kw: nn.L1Loss(),
    }
    if name not in registry:
        raise ValueError(f"Unknown loss {name!r}. Choose from: {list(registry)}")
    return registry[name](**kwargs)