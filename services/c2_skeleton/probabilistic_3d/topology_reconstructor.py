"""
TopologyReconstructor
=====================
Core engine for GAP 1: Probabilistic 3D Graph Reconstruction.

Algorithm:
  Input  : 2D keypoints from YOLO (center, tip1, tip2)
  Step 1 : Generate K candidate 3D structures via LearnedGraphPrior
  Step 2 : Project each structure back to 2D (geometric rendering)
  Step 3 : Score each hypothesis: log P(I|G) + log P(G)
  Step 4 : Return MAP estimate + full posterior distribution

Rendering (geometric projection):
  Uses weak-perspective (orthographic) projection for simplicity.
  Depth (Z) manifests as slight scaling of hand length in the image.
  A full differentiable renderer (pytorch3d / nvdiffrast) can replace
  the projection step without changing the interface.

Differentiable aspect:
  Gradients are computed numerically (finite differences) over Z to allow
  gradient-based refinement of the MAP estimate.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .graph_prior import LearnedGraphPrior, GraphStructure3D
from .uncertainty import UncertaintyEstimator


class TopologyReconstructor:
    """
    Reconstruct the most likely 3D clock-hand structure from 2D keypoints.

    Parameters
    ----------
    k_hypotheses : int
        Number of 3D candidate structures to sample (default 10).
    image_size : int
        Expected image side length in pixels (for prior scaling).
    focal_length : float
        Simulated camera focal length for weak-perspective projection.
    refine_steps : int
        Gradient-descent refinement steps on the MAP hypothesis.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        k_hypotheses: int = 10,
        image_size: int = 500,
        focal_length: float = 800.0,
        refine_steps: int = 30,
        seed: Optional[int] = None,
    ):
        self.k_hypotheses = k_hypotheses
        self.image_size = image_size
        self.focal_length = focal_length
        self.refine_steps = refine_steps
        self.rng = np.random.default_rng(seed)

        self.prior = LearnedGraphPrior(image_size=image_size)
        self.uncertainty_estimator = UncertaintyEstimator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer_3d_structure(
        self,
        center_2d: List[float],
        tip1_2d: List[float],
        tip2_2d: List[float],
    ) -> Dict:
        """
        Main inference call: Given 2D keypoints, infer best 3D structure.

        Parameters
        ----------
        center_2d : [x, y]
        tip1_2d   : [x, y]
        tip2_2d   : [x, y]

        Returns
        -------
        dict with keys:
          map_structure  — MAP estimate (most likely 3D structure)
          all_hypotheses — list of all K scored hypotheses
          uncertainty    — full uncertainty report
          rendering_scores — per-hypothesis rendering match scores
        """
        # Step 1: Sample K candidate 3D structures from prior
        candidates: List[GraphStructure3D] = self.prior.sample(
            center_2d=tuple(center_2d),
            tip1_2d=tuple(tip1_2d),
            tip2_2d=tuple(tip2_2d),
            k=self.k_hypotheses,
            rng=self.rng,
        )

        # Step 2: Score each hypothesis via rendering likelihood
        log_scores = np.array([
            self._compute_log_score(s, center_2d, tip1_2d, tip2_2d)
            for s in candidates
        ])

        # Step 3: Select MAP (Maximum A Posteriori) hypothesis
        best_idx = int(np.argmax(log_scores))
        map_structure = candidates[best_idx]

        # Step 4: Refine MAP via gradient descent on Z values
        map_structure = self._refine_map(map_structure, center_2d, tip1_2d, tip2_2d)

        # Step 5: Uncertainty quantification
        uncertainty = self.uncertainty_estimator.compute(candidates, log_scores)

        return {
            "map_structure": map_structure.to_dict(),
            "map_hypothesis_idx": best_idx,
            "all_hypotheses": [s.to_dict() for s in candidates],
            "log_scores": log_scores.tolist(),
            "rendering_scores": [float(s) for s in np.exp(log_scores - np.max(log_scores))],
            "uncertainty": uncertainty,
        }

    # ------------------------------------------------------------------
    # Rendering likelihood: P(I | G)
    # ------------------------------------------------------------------

    def _render_to_2d(self, structure: GraphStructure3D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weak-perspective (orthographic + depth scaling) projection.

        Depth (Z) modulates apparent hand length:
          scale = focal / (focal + Z)
          projected_tip = center + (tip - center)_xy * scale

        Returns projected tip1 and tip2 in 2D.
        """
        c = structure.center
        f = self.focal_length

        def project(tip_3d):
            z_offset = tip_3d[2] - c[2]
            scale = f / (f + z_offset + 1e-6)
            delta = tip_3d[:2] - c[:2]
            return c[:2] + delta * scale

        proj_t1 = project(structure.tip1)
        proj_t2 = project(structure.tip2)
        return proj_t1, proj_t2

    def _rendering_log_likelihood(
        self,
        structure: GraphStructure3D,
        obs_tip1: List[float],
        obs_tip2: List[float],
        sigma: float = 8.0,
    ) -> float:
        """
        log P(I | G) — how well the rendered projection matches observed 2D keypoints.

        Modelled as isotropic Gaussian noise around each projected keypoint:
          log P(I|G) = -Σ ||projected_tip_i - observed_tip_i||² / (2σ²)
        """
        proj_t1, proj_t2 = self._render_to_2d(structure)
        obs1 = np.array(obs_tip1)
        obs2 = np.array(obs_tip2)

        # Both assignment orderings (in case YOLO swapped hour/minute)
        err_direct = (np.sum((proj_t1 - obs1) ** 2) + np.sum((proj_t2 - obs2) ** 2))
        err_swapped = (np.sum((proj_t1 - obs2) ** 2) + np.sum((proj_t2 - obs1) ** 2))

        best_err = min(err_direct, err_swapped)
        return -best_err / (2.0 * sigma ** 2)

    def _compute_log_score(
        self,
        structure: GraphStructure3D,
        center_2d, tip1_2d, tip2_2d,
    ) -> float:
        """log P(G|I) ∝ log P(I|G) + log P(G)."""
        log_likelihood = self._rendering_log_likelihood(structure, tip1_2d, tip2_2d)
        log_prior = structure.log_prior
        return log_likelihood + log_prior

    # ------------------------------------------------------------------
    # MAP refinement via numerical gradients (finite differences)
    # ------------------------------------------------------------------

    def _refine_map(
        self,
        structure: GraphStructure3D,
        center_2d, tip1_2d, tip2_2d,
        lr: float = 0.5,
        eps: float = 1e-3,
    ) -> GraphStructure3D:
        """
        Gradient ascent on log P(G|I) w.r.t. Z values of tip1 and tip2.

        Uses numerical finite differences (forward difference).
        """
        s = GraphStructure3D(
            center=structure.center.copy(),
            tip1=structure.tip1.copy(),
            tip2=structure.tip2.copy(),
            metadata={**structure.metadata, "refined": True},
        )

        for _ in range(self.refine_steps):
            base_score = self._compute_log_score(s, center_2d, tip1_2d, tip2_2d)

            # Gradient w.r.t. z of tip1
            s_plus = GraphStructure3D(s.center.copy(), s.tip1.copy(), s.tip2.copy())
            s_plus.tip1[2] += eps
            s_plus.log_prior = self.prior.log_prob(s_plus)
            grad_z1 = (self._compute_log_score(s_plus, center_2d, tip1_2d, tip2_2d) - base_score) / eps

            # Gradient w.r.t. z of tip2
            s_plus2 = GraphStructure3D(s.center.copy(), s.tip1.copy(), s.tip2.copy())
            s_plus2.tip2[2] += eps
            s_plus2.log_prior = self.prior.log_prob(s_plus2)
            grad_z2 = (self._compute_log_score(s_plus2, center_2d, tip1_2d, tip2_2d) - base_score) / eps

            # Ascent step
            s.tip1[2] += lr * grad_z1
            s.tip2[2] += lr * grad_z2
            s.log_prior = self.prior.log_prob(s)

            # Decay learning rate
            lr *= 0.95

        return s
