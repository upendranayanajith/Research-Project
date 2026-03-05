"""
LearnedGraphPrior
=================
Encodes prior beliefs P(G) about plausible 3D clock-hand configurations.

For a clock:
  - Two hands (hour + minute) originate from the same center point
  - Hour hand is shorter than minute hand
  - Both hands lie approximately in the clock face plane (Z ≈ 0)
  - Their angular difference tells the time → all angles are valid
  - Depth (Z) of tip relative to center encodes which hand is "on top"

Prior distributions used:
  - Hand length: Gaussian centered on expected fraction of clock radius
  - Depth offset (Z): Gaussian centered at 0 with small variance
  - Angular spread: Uniform (all times equally likely)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GraphStructure3D:
    """
    Represents a single 3D graph hypothesis for clock hands.

    Attributes
    ----------
    center : np.ndarray, shape (3,)
        3D position of the clock center [x, y, z].
    tip1 : np.ndarray, shape (3,)
        3D position of hand 1 tip (assumed hour hand candidate).
    tip2 : np.ndarray, shape (3,)
        3D position of hand 2 tip (assumed minute hand candidate).
    log_prior : float
        Log prior probability log P(G) of this structure.
    metadata : dict
        Extra info (hypothesis index, sampling parameters, etc.).
    """
    center: np.ndarray
    tip1: np.ndarray
    tip2: np.ndarray
    log_prior: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "tip1": self.tip1.tolist(),
            "tip2": self.tip2.tolist(),
            "log_prior": float(self.log_prior),
            "metadata": self.metadata,
        }


class LearnedGraphPrior:
    """
    Parametric prior P(G) over 3D clock-hand graph structures.

    Clock-specific knowledge encoded:
      1. Hour hand length ≈ 35-50% of image size
      2. Minute hand length ≈ 60-85% of image size
      3. Depth offset (Z) is small; minute hand is typically closer to viewer
      4. Center is at or near the image centroid

    Parameters
    ----------
    image_size : int
        Assumed square image side length in pixels (used to scale priors).
    depth_scale : float
        Scale of the Z-axis relative to the XY plane.
        1.0 means Z is the same unit as pixels.
    """

    def __init__(self, image_size: int = 500, depth_scale: float = 50.0):
        self.image_size = image_size
        self.depth_scale = depth_scale

        # --- Prior parameters (calibrated for clock images) ---
        # Hand length: fraction of half-image (radius)
        self.hour_len_mean = 0.40 * (image_size / 2)
        self.hour_len_std = 0.08 * (image_size / 2)
        self.minute_len_mean = 0.72 * (image_size / 2)
        self.minute_len_std = 0.10 * (image_size / 2)

        # Depth offset: minute hand is typically slightly in front
        # Z > 0 = closer to camera (in front)
        self.depth_mean_minute = 0.2 * depth_scale   # minute slightly in front
        self.depth_mean_hour = -0.2 * depth_scale     # hour slightly behind
        self.depth_std = 0.5 * depth_scale

    def log_prob(self, structure: GraphStructure3D) -> float:
        """
        Compute log P(G) for a given 3D structure.

        Uses Gaussian log-likelihoods for hand lengths and depth offsets.
        """
        c = structure.center
        t1 = structure.tip1
        t2 = structure.tip2

        len1 = float(np.linalg.norm(t1[:2] - c[:2]))  # XY length of hand 1
        len2 = float(np.linalg.norm(t2[:2] - c[:2]))  # XY length of hand 2
        z1 = float(t1[2] - c[2])                       # depth offset hand 1
        z2 = float(t2[2] - c[2])                       # depth offset hand 2

        # Assign hand roles based on length (shorter = hour)
        if len1 <= len2:
            hour_len, minute_len = len1, len2
            z_hour, z_minute = z1, z2
        else:
            hour_len, minute_len = len2, len1
            z_hour, z_minute = z2, z1

        # Gaussian log-prob for lengths
        lp_hour_len = self._gaussian_logp(hour_len, self.hour_len_mean, self.hour_len_std)
        lp_min_len = self._gaussian_logp(minute_len, self.minute_len_mean, self.minute_len_std)

        # Gaussian log-prob for depths
        lp_hour_z = self._gaussian_logp(z_hour, self.depth_mean_hour, self.depth_std)
        lp_min_z = self._gaussian_logp(z_minute, self.depth_mean_minute, self.depth_std)

        return lp_hour_len + lp_min_len + lp_hour_z + lp_min_z

    def sample(self, center_2d: Tuple[float, float], tip1_2d: Tuple[float, float],
               tip2_2d: Tuple[float, float], k: int = 10,
               rng: np.random.Generator = None) -> list:
        """
        Sample K candidate 3D structures by perturbing depth (Z) values.

        The XY positions come from the 2D YOLO detections.
        Only the Z component is sampled from the prior.

        Parameters
        ----------
        center_2d : (x, y)
        tip1_2d   : (x, y)
        tip2_2d   : (x, y)
        k         : number of hypotheses to generate
        rng       : optional numpy random Generator for reproducibility

        Returns
        -------
        list of GraphStructure3D
        """
        if rng is None:
            rng = np.random.default_rng()

        center_3d = np.array([center_2d[0], center_2d[1], 0.0])
        structures = []

        for i in range(k):
            z1 = rng.normal(self.depth_mean_hour, self.depth_std)
            z2 = rng.normal(self.depth_mean_minute, self.depth_std)

            t1 = np.array([tip1_2d[0], tip1_2d[1], z1])
            t2 = np.array([tip2_2d[0], tip2_2d[1], z2])

            s = GraphStructure3D(
                center=center_3d.copy(),
                tip1=t1,
                tip2=t2,
                metadata={"hypothesis_idx": i}
            )
            s.log_prior = self.log_prob(s)
            structures.append(s)

        return structures

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _gaussian_logp(x: float, mean: float, std: float) -> float:
        """Log probability of x under N(mean, std²)."""
        variance = std ** 2 + 1e-8
        return -0.5 * ((x - mean) ** 2 / variance) - 0.5 * np.log(2 * np.pi * variance)
