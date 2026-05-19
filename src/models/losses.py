"""
src/models/losses.py
--------------------
Loss functions for depth estimation.

- SILogLoss: Scale-Invariant Logarithmic Loss (standard for monocular depth)
- BerHuLoss: Reversed Huber Loss (robust to outliers)
"""

import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss.

    Formula:
        L = sqrt( mean(d^2) - lambda * (mean(d))^2 )
    where d = log(pred) - log(target)

    lambda=0.5 is the standard value used in the original paper.
    """
    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = (target > 0).detach()
        if not mask.any():
            return torch.tensor(0.0, device=pred.device)

        pred_m = pred[mask]
        target_m = target[mask]

        eps = 1e-6
        diff = torch.log(pred_m + eps) - torch.log(target_m + eps)
        loss = torch.sqrt(
            torch.mean(diff ** 2) - self.lambd * (torch.mean(diff) ** 2)
        )
        return loss


class BerHuLoss(nn.Module):
    """
    Reversed Huber (BerHu) Loss.

    Used as an alternative to SILog. Less sensitive to outliers.
    Threshold c = 0.2 * max(|pred - target|)
    """
    def __init__(self, threshold: float = 0.2):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = (target > 0).detach()
        if not mask.any():
            return torch.tensor(0.0, device=pred.device)

        diff = torch.abs(pred[mask] - target[mask])
        c = self.threshold * torch.max(diff).detach()

        # L1 for |diff| <= c, L2 for |diff| > c
        loss = torch.where(
            diff <= c,
            diff,
            (diff ** 2 + c ** 2) / (2 * c)
        )
        return loss.mean()