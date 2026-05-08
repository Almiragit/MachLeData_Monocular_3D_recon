"""
src/models/losses.py
--------------------
Custom loss functions for monocular depth estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss (Eigen et al., 2014).
    Standard loss for monocular depth estimation.
    """

    def __init__(self, lambd: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target > 0
        d = torch.log(pred[mask] + self.eps) - torch.log(target[mask] + self.eps)
        return d.pow(2).mean() - self.lambd * d.mean().pow(2)


class BerHuLoss(nn.Module):
    """
    Reverse Huber (BerHu) loss — combines L1 for small errors, L2 for large.
    """

    def __init__(self, threshold: float = 0.2):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target > 0
        diff = torch.abs(pred[mask] - target[mask])
        c = self.threshold * diff.max().detach()
        l1_mask = diff <= c
        loss = torch.where(l1_mask, diff, (diff ** 2 + c ** 2) / (2 * c + 1e-8))
        return loss.mean()
