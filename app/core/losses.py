"""
app/core/losses.py — Loss functions and calibration metrics for C3 angle regression.

Classes
-------
VonMisesLoss      : Circular-safe cosine loss for (sin, cos) head output.
CircularMSELoss   : Scalar-angle MSE that wraps differences to [−180, 180].
ECEMetric         : Expected Calibration Error for uncertainty quantification.
"""

import torch
import torch.nn as nn
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# VonMisesLoss  — for CircularHead (sin,cos) output
# ══════════════════════════════════════════════════════════════════════════════
class VonMisesLoss(nn.Module):
    """
    Cosine-distance loss for circular angle regression.

    Input:
        pred   (B, 2) — L2-normalised [sin θ, cos θ] predictions
        target (B, 2) — L2-normalised [sin θ, cos θ] ground truth

    Output: scalar mean loss in [0, 2].
        0 → perfect alignment (same angle)
        2 → opposite angle (180° apart)

    Advantages over MSE on raw angles:
        - Smooth at the 0°/360° boundary (no discontinuity)
        - Scale-invariant in angle space
        - Reference: Mardia & Jupp, "Directional Statistics" (2000)
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure unit-circle normalisation
        pred   = nn.functional.normalize(pred,   dim=1)
        target = nn.functional.normalize(target, dim=1)
        # 1 − cos(θ_pred − θ_true)  ==  1 − dot(pred, target) for unit vectors
        cosine_similarity = (pred * target).sum(dim=1)   # (B,)
        return (1.0 - cosine_similarity).mean()


# ══════════════════════════════════════════════════════════════════════════════
# CircularMSELoss  — for scalar-angle Sigmoid output
# ══════════════════════════════════════════════════════════════════════════════
class CircularMSELoss(nn.Module):
    """
    MSE loss on scalar angles (°) with circular wrapping.

    Wraps the difference into [−180, 180] before squaring, so
    the distance between 1° and 359° is correctly seen as 2° (not 358°).

    Usage:
        loss_fn = CircularMSELoss()
        loss = loss_fn(pred_deg, target_deg)   # both (B,) tensors in degrees
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        # Wrap to (−180, 180]
        diff = (diff + 180.0) % 360.0 - 180.0
        return (diff ** 2).mean()


# ══════════════════════════════════════════════════════════════════════════════
# ECEMetric  — Expected Calibration Error for MC-Dropout uncertainty
# ══════════════════════════════════════════════════════════════════════════════
class ECEMetric:
    """
    Expected Calibration Error (ECE) for regression uncertainty.

    Bins predictions by predicted σ (uncertainty) and measures whether
    the fraction of samples with error ≤ σ matches the expected fraction
    under a Gaussian model (≈ 68% for 1σ, 95% for 2σ).

    Usage:
        ece = ECEMetric(n_bins=10)
        ece.update(predicted_sigma=5.2, actual_error=3.1)
        ...
        score = ece.compute()   # float in [0, 1]; lower is better
        ece.reset()
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bins: dict[int, list] = {i: [] for i in range(n_bins)}
        self._sigma_min = 0.0
        self._sigma_max = 30.0   # max useful σ in degrees

    def _bin_idx(self, sigma: float) -> int:
        norm = (sigma - self._sigma_min) / (self._sigma_max - self._sigma_min)
        return min(int(norm * self.n_bins), self.n_bins - 1)

    def update(self, predicted_sigma: float, actual_error: float):
        """
        Args:
            predicted_sigma: MC-Dropout circular std in degrees
            actual_error:    |predicted_angle − ground_truth_angle| in degrees (wrapped)
        """
        idx = self._bin_idx(predicted_sigma)
        # 1 if actual error within the predicted sigma, 0 otherwise
        within = 1 if actual_error <= predicted_sigma else 0
        self.bins[idx].append((predicted_sigma, actual_error, within))

    def compute(self) -> float:
        """Return ECE ∈ [0, 1]; 0 = perfectly calibrated."""
        total = sum(len(v) for v in self.bins.values())
        if total == 0:
            return 0.0

        ece = 0.0
        for bin_data in self.bins.values():
            if not bin_data:
                continue
            n    = len(bin_data)
            conf = np.mean([d[0] for d in bin_data])  # mean σ in bin
            acc  = np.mean([d[2] for d in bin_data])  # fraction within σ
            # Expected fraction within σ under Gaussian ≈ erf(1/√2) ≈ 0.6827
            expected = 0.6827
            ece += (n / total) * abs(acc - expected)
        return float(ece)

    def reset(self):
        self.bins = {i: [] for i in range(self.n_bins)}

    def summary(self) -> str:
        ece = self.compute()
        n   = sum(len(v) for v in self.bins.values())
        return f"ECE={ece:.4f}  (n={n} samples, {self.n_bins} bins)"
