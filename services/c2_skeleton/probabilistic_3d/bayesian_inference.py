"""
BayesianGraphInference
=======================
High-level orchestrator for the full Bayesian inference pipeline:

  P(G | I) = P(I | G) × P(G) / P(I)

This module ties together:
  - LearnedGraphPrior  → P(G)
  - TopologyReconstructor → sampling + P(I|G) evaluation + MAP
  - UncertaintyEstimator → posterior uncertainty

Usage
-----
    inference = BayesianGraphInference()
    result = inference.infer(center_2d, tip1_2d, tip2_2d)
    print(result["summary"])
"""

import numpy as np
from typing import Dict, List, Optional
from .topology_reconstructor import TopologyReconstructor
from .uncertainty import UncertaintyEstimator


class BayesianGraphInference:
    """
    End-to-end Bayesian inference for 3D clock-hand graph structure.

    Parameters
    ----------
    k_hypotheses : int
        Number of posterior samples (more = better coverage, slower).
    image_size : int
        Expected image size in pixels.
    seed : int, optional
        For reproducibility.
    """

    def __init__(
        self,
        k_hypotheses: int = 10,
        image_size: int = 500,
        seed: Optional[int] = None,
    ):
        self.reconstructor = TopologyReconstructor(
            k_hypotheses=k_hypotheses,
            image_size=image_size,
            seed=seed,
        )
        self.uncertainty_estimator = UncertaintyEstimator()
        self.k = k_hypotheses

    def infer(
        self,
        center_2d: List[float],
        tip1_2d: List[float],
        tip2_2d: List[float],
    ) -> Dict:
        """
        Run full Bayesian inference and return a human-readable result.

        Parameters
        ----------
        center_2d : [x, y]  — clock center keypoint
        tip1_2d   : [x, y]  — first hand tip keypoint
        tip2_2d   : [x, y]  — second hand tip keypoint

        Returns
        -------
        dict containing:
          map_structure       — best 3D structure [center, tip1, tip2 with Z]
          uncertainty         — credible intervals, std devs, confidence
          hand_assignment     — which hand is hour vs minute
          summary             — human-readable interpretation
        """
        raw = self.reconstructor.infer_3d_structure(center_2d, tip1_2d, tip2_2d)

        map_s = raw["map_structure"]
        uncertainty = raw["uncertainty"]

        # --- Determine hand assignment (hour vs minute by 2D length) ---
        c = np.array(map_s["center"][:2])
        t1 = np.array(map_s["tip1"][:2])
        t2 = np.array(map_s["tip2"][:2])

        len1 = float(np.linalg.norm(t1 - c))
        len2 = float(np.linalg.norm(t2 - c))

        if len1 <= len2:
            hand_assignment = {"hour": "tip1", "minute": "tip2"}
            hour_angle = uncertainty["angle_uncertainty"]["hand1_mean_deg"]
            minute_angle = uncertainty["angle_uncertainty"]["hand2_mean_deg"]
            hour_z = float(map_s["tip1"][2]) - float(map_s["center"][2])
            minute_z = float(map_s["tip2"][2]) - float(map_s["center"][2])
        else:
            hand_assignment = {"hour": "tip2", "minute": "tip1"}
            hour_angle = uncertainty["angle_uncertainty"]["hand2_mean_deg"]
            minute_angle = uncertainty["angle_uncertainty"]["hand1_mean_deg"]
            hour_z = float(map_s["tip2"][2]) - float(map_s["center"][2])
            minute_z = float(map_s["tip1"][2]) - float(map_s["center"][2])

        # --- Overlap / occlusion risk assessment ---
        overlap_prob = uncertainty["overlap_probability"]
        occlusion_risk = self._classify_occlusion_risk(overlap_prob, hour_angle, minute_angle)

        # --- Summary ---
        summary = (
            f"3D reconstruction with {self.k} hypotheses. "
            f"Confidence: {uncertainty['confidence_score']:.2f}. "
            f"Hour hand at {hour_angle:.1f}° (Z={hour_z:+.1f}), "
            f"Minute hand at {minute_angle:.1f}° (Z={minute_z:+.1f}). "
            f"Occlusion risk: {occlusion_risk}."
        )

        return {
            "map_structure": map_s,
            "uncertainty": uncertainty,
            "hand_assignment": hand_assignment,
            "hand_depths": {
                "hour_z_offset": round(hour_z, 3),
                "minute_z_offset": round(minute_z, 3),
                "minute_is_in_front": bool(minute_z > hour_z),
            },
            "occlusion_risk": occlusion_risk,
            "summary": summary,
            "num_hypotheses_used": len(raw["all_hypotheses"]),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_occlusion_risk(
        overlap_probability: float,
        hour_angle: float,
        minute_angle: float,
    ) -> str:
        """
        Estimate occlusion risk based on:
          1. Overlap probability from the posterior
          2. Angular proximity of the two hands
        """
        angular_diff = abs(hour_angle - minute_angle)
        angular_diff = min(angular_diff, 360 - angular_diff)   # wrap-around

        if angular_diff < 10.0:
            base_risk = "HIGH"
        elif angular_diff < 25.0:
            base_risk = "MEDIUM"
        else:
            base_risk = "LOW"

        # Refine with posterior overlap probability
        if overlap_probability > 0.80 and base_risk == "LOW":
            base_risk = "MEDIUM"

        return base_risk
