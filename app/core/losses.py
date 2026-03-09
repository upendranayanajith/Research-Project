"""
app/core/losses.py
==================
[T3.5] Circular Regression — Von Mises Loss & Circular Head

Solves the fundamental 0°/360° wraparound ambiguity in C3's Sigmoid output.

Instead of predicting a scalar angle ∈ [0,1] and multiplying by 360°
(which collapses 0° and 360° into the same point on a linear scale),
we predict TWO scalars: (sin θ, cos θ). This lives on the unit circle,
where 0° and 360° are identical by construction.

Decoding: θ = atan2(sin_pred, cos_pred) * 180/π  (mod 360)

Loss function: Von Mises / Cosine distance
    L = 1 - cos(θ_pred - θ_true)
    = 1 - (sin_pred·sin_true + cos_pred·cos_true)

This is strictly ≥ 0, smooth at the wraparound boundary, and rotationally invariant.

Research reference:
    Gao et al. (2020) "Regression of Instance Boundary by the Relative Distance
    Distribution" — circular loss for angular predictions.
"""

import torch
import torch.nn as nn
import math


class VonMisesLoss(nn.Module):
    """
    [T3.5] Von Mises / Cosine Angular Distance Loss.

    Operates on (sin, cos) predictions — NOT raw angle values.

    Args:
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets_deg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions:  (N, 2) tensor — [sin_pred, cos_pred] from CircularHead.
            targets_deg:  (N,) tensor — ground-truth angle in degrees ∈ [0, 360).

        Returns:
            Loss scalar (or tensor if reduction='none').
        """
        # Convert target degrees → (sin, cos)
        targets_rad = targets_deg * (math.pi / 180.0)
        sin_true = torch.sin(targets_rad)
        cos_true = torch.cos(targets_rad)

        sin_pred = predictions[:, 0]
        cos_pred = predictions[:, 1]

        # Cosine distance: 1 - cos(θ_pred - θ_true)
        # = 1 - (sin_pred·sin_true + cos_pred·cos_true)
        cos_diff = sin_pred * sin_true + cos_pred * cos_true
        loss = 1.0 - cos_diff   # ∈ [0, 2]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss   # (N,)


class CircularHead(nn.Module):
    """
    [T3.5] Drop-in replacement for the Sigmoid head in C3.

    Outputs (sin θ, cos θ) on the unit circle instead of a scalar ∈ [0,1].
    Enables VonMisesLoss and eliminates 0°/360° discontinuity.

    Architecture:
        Linear(512 → 2) → L2-normalise → unit circle constraint

    Usage:
        model = nn.Sequential(resnet18_backbone, CircularHead())
        sin_cos = model(x)          # (N, 2)
        angle_deg = decode_circular(sin_cos)   # (N,) degrees
    """

    def __init__(self, in_features: int = 512):
        super().__init__()
        self.fc = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 512) feature vector from ResNet18 backbone.
        Returns:
            (N, 2) — [sin θ, cos θ], L2-normalised to unit circle.
        """
        raw = self.fc(x)                    # (N, 2)
        norm = raw.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return raw / norm                   # unit circle: sin²+cos²=1


def decode_circular(sin_cos: torch.Tensor) -> torch.Tensor:
    """
    Convert (sin θ, cos θ) predictions back to degrees ∈ [0, 360).

    Args:
        sin_cos: (N, 2) tensor [sin_pred, cos_pred].
    Returns:
        (N,) tensor of angles in degrees ∈ [0, 360).
    """
    sin_pred = sin_cos[:, 0]
    cos_pred = sin_cos[:, 1]
    rad = torch.atan2(sin_pred, cos_pred)           # ∈ (-π, π]
    deg = rad * (180.0 / math.pi)                   # ∈ (-180, 180]
    return (deg % 360.0)                            # ∈ [0, 360)


def decode_circular_numpy(sin_cos_np):
    """
    NumPy version of decode_circular for inference (no grad needed).

    Args:
        sin_cos_np: (2,) or (N, 2) array [sin_pred, cos_pred].
    Returns:
        angle in degrees ∈ [0, 360).
    """
    import numpy as np
    sin_pred = sin_cos_np[..., 0]
    cos_pred = sin_cos_np[..., 1]
    rad = np.arctan2(sin_pred, cos_pred)
    deg = np.degrees(rad)
    return deg % 360.0


class MeanCircularError(nn.Module):
    """
    [T3.5] Mean Circular Error (MCE) — torch metric for eval.

    Computes the circular mean absolute error:
        MCE = mean( arccos( sin_pred·sin_true + cos_pred·cos_true ) * 180/π )

    This is the circular equivalent of MAE — always ∈ [0°, 180°].
    """

    def forward(self, predictions: torch.Tensor, targets_deg: torch.Tensor) -> torch.Tensor:
        targets_rad = targets_deg * (math.pi / 180.0)
        sin_true = torch.sin(targets_rad)
        cos_true = torch.cos(targets_rad)

        cos_diff = (predictions[:, 0] * sin_true + predictions[:, 1] * cos_true).clamp(-1.0, 1.0)
        error_rad = torch.acos(cos_diff)
        return error_rad.mean() * (180.0 / math.pi)   # → degrees
