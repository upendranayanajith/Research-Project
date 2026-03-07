"""
UncertaintyEstimator
=====================
Quantifies uncertainty over 3D reconstruction results.

Given the posterior distribution (scores over K hypotheses), computes:
  - Per-keypoint credible intervals (90% / 95%)
  - Angle uncertainty (std dev across posterior)
  - Overlap probability: P(hour hand is behind minute hand)
  - Overall reconstruction confidence score [0, 1]

Mathematical basis:
  Treats the K scored hypotheses as an empirical approximation to the
  posterior P(G | I). Weights are computed by softmax of log scores.
"""

import numpy as np
from typing import List, Tuple, Dict
from .graph_prior import GraphStructure3D


class UncertaintyEstimator:
    """
    Compute uncertainty metrics over a set of posterior-weighted 3D hypotheses.

    Parameters
    ----------
    credible_level : float
        Probability mass for credible intervals (default 0.90 = 90%).
    """

    def __init__(self, credible_level: float = 0.90):
        self.credible_level = credible_level

    def compute(
        self,
        structures: List[GraphStructure3D],
        log_scores: np.ndarray,
    ) -> Dict:
        """
        Compute full uncertainty report.

        Parameters
        ----------
        structures  : K candidate 3D structures
        log_scores  : shape (K,), log P(I|G) + log P(G) for each hypothesis

        Returns
        -------
        dict with keys:
          weights, angle_uncertainty_deg, depth_uncertainty,
          overlap_probability, confidence_score, credible_intervals
        """
        K = len(structures)
        if K == 0:
            return self._empty_report()

        # Posterior weights via softmax
        weights = self._softmax(log_scores)                   # shape (K,)

        # --- Collect arrays ---
        tips1 = np.stack([s.tip1 for s in structures])       # (K, 3)
        tips2 = np.stack([s.tip2 for s in structures])       # (K, 3)
        centers = np.stack([s.center for s in structures])   # (K, 3)

        # Angles (from 12 o'clock, clockwise) for each hypothesis
        angles1 = np.array([self._angle_from_12(s.center[:2], s.tip1[:2]) for s in structures])
        angles2 = np.array([self._angle_from_12(s.center[:2], s.tip2[:2]) for s in structures])

        # Weighted mean and std for angles
        mean_a1 = np.average(angles1, weights=weights)
        mean_a2 = np.average(angles2, weights=weights)
        std_a1 = np.sqrt(np.average((angles1 - mean_a1) ** 2, weights=weights))
        std_a2 = np.sqrt(np.average((angles2 - mean_a2) ** 2, weights=weights))

        # Depth uncertainty (std of Z values)
        z1 = tips1[:, 2] - centers[:, 2]
        z2 = tips2[:, 2] - centers[:, 2]
        std_z1 = np.sqrt(np.average((z1 - np.average(z1, weights=weights)) ** 2, weights=weights))
        std_z2 = np.sqrt(np.average((z2 - np.average(z2, weights=weights)) ** 2, weights=weights))

        # Overlap probability: P(tip1 is BEHIND tip2) = P(z1 < z2)
        # i.e., minute hand is in front
        z_diff = z2 - z1   # positive → tip2 is more in front
        overlap_prob = float(np.sum(weights[z_diff > 0]))

        # Credible intervals on 3D positions (weighted quantiles)
        ci_tip1 = self._weighted_credible_interval(tips1, weights, self.credible_level)
        ci_tip2 = self._weighted_credible_interval(tips2, weights, self.credible_level)

        # Confidence score: based on posterior entropy (lower entropy = higher confidence)
        entropy = -np.sum(weights * np.log(weights + 1e-12))
        max_entropy = np.log(K)  # uniform distribution entropy
        confidence = 1.0 - (entropy / (max_entropy + 1e-8))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return {
            "posterior_weights": weights.tolist(),
            "angle_uncertainty": {
                "hand1_mean_deg": round(float(mean_a1), 2),
                "hand1_std_deg": round(float(std_a1), 2),
                "hand2_mean_deg": round(float(mean_a2), 2),
                "hand2_std_deg": round(float(std_a2), 2),
            },
            "depth_uncertainty": {
                "hand1_z_std": round(float(std_z1), 3),
                "hand2_z_std": round(float(std_z2), 3),
            },
            "overlap_probability": round(overlap_prob, 3),
            "confidence_score": round(confidence, 3),
            "credible_intervals": {
                "level": self.credible_level,
                "tip1": ci_tip1,
                "tip2": ci_tip2,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _weighted_credible_interval(
        self, points: np.ndarray, weights: np.ndarray, level: float
    ) -> Dict:
        """Compute per-axis weighted quantile credible intervals."""
        alpha = (1.0 - level) / 2.0
        result = {}
        for axis, name in enumerate(["x", "y", "z"]):
            lo, hi = self._weighted_quantile(points[:, axis], weights, [alpha, 1 - alpha])
            result[name] = {"low": round(float(lo), 2), "high": round(float(hi), 2)}
        return result

    @staticmethod
    def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list) -> list:
        """Compute weighted quantiles."""
        sorter = np.argsort(values)
        sorted_vals = values[sorter]
        sorted_wts = weights[sorter]
        cumulative = np.cumsum(sorted_wts)
        cumulative /= cumulative[-1]   # normalize
        return [float(np.interp(q, cumulative, sorted_vals)) for q in quantiles]

    @staticmethod
    def _softmax(log_scores: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = log_scores - np.max(log_scores)
        exp_s = np.exp(np.clip(shifted, -500, 0))
        return exp_s / (exp_s.sum() + 1e-12)

    @staticmethod
    def _angle_from_12(center: np.ndarray, tip: np.ndarray) -> float:
        """Clockwise angle from 12 o'clock position, in degrees [0, 360)."""
        dx = tip[0] - center[0]
        dy = tip[1] - center[1]
        angle = np.degrees(np.arctan2(dx, -dy))
        return float(angle + 360) if angle < 0 else float(angle)

    @staticmethod
    def _empty_report() -> Dict:
        return {
            "posterior_weights": [],
            "angle_uncertainty": {},
            "depth_uncertainty": {},
            "overlap_probability": 0.5,
            "confidence_score": 0.0,
            "credible_intervals": {},
        }
