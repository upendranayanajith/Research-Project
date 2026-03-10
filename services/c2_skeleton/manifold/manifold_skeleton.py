"""
ManifoldSkeletonDetector
=========================
Detects instrument pointer skeleton connectivity using GEODESIC distances
instead of Euclidean distances on the Riemannian manifold.

Pipeline:
  1. Estimate metric tensor field g(x,y) from input image
  2. Given 2D keypoints (center, tip1, tip2 or center, needle_tip), compute pairwise geodesic distances
  3. Build graph where edges connect keypoints below geodesic threshold
  4. Compare with Euclidean graph — quantify curvature effect

Instrument-specific insight:
  On a flat image, Euclidean and geodesic are the same.
  On a curved surface (car dashboard gauge, wrist watch, pipe-mounted gauge):
    - Pointers may appear closer in Euclidean space but far geodesically
      (they're on opposite sides of a curvature ridge)
    - This explains why Euclidean-based trackers fail on curved dials

The curvature ratio (geodesic / Euclidean) is a direct measure of
how non-flat the surface is in the image.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from .metric_estimator import MetricTensorEstimator
from .geodesic import GeodesicDistanceMap


class ManifoldSkeletonDetector:
    """
    Builds instrument-pointer graph using intrinsic (geodesic) distances.

    Parameters
    ----------
    anisotropy : float
        Passed to MetricTensorEstimator — controls boundary strength.
    downsample : int
        Passed to GeodesicDistanceMap — trade-off speed vs accuracy.
    geodesic_threshold_factor : float
        Edge added if geodesic_dist < threshold_factor × image_diagonal.
    """

    def __init__(
        self,
        anisotropy: float = 5.0,
        downsample: int = 8,
        geodesic_threshold_factor: float = 0.9,
    ):
        self.estimator = MetricTensorEstimator(anisotropy=anisotropy)
        self.downsample = downsample
        self.threshold_factor = geodesic_threshold_factor

    def detect(
        self,
        image: np.ndarray,
        center: List[float],
        tip1: List[float],
        tip2: List[float],
    ) -> Dict:
        """
        Run the full manifold skeleton detection pipeline.

        Parameters
        ----------
        image  : np.ndarray, shape (H, W, 3)
        center : [x, y]
        tip1   : [x, y]
        tip2   : [x, y]

        Returns
        -------
        dict with:
          geodesic_distances      — pairwise geodesic distance matrix
          euclidean_distances     — pairwise Euclidean distances
          curvature_ratios        — geodesic / euclidean for each pair
          geodesic_graph          — edges based on geodesic threshold
          euclidean_graph         — edges based on euclidean threshold (standard)
          manifold_analysis       — structured comparison
          metric_magnitude        — per-pixel curvature magnitude (encoded)
        """
        H, W = image.shape[:2]
        diag = float(np.sqrt(H**2 + W**2))
        threshold = self.threshold_factor * diag

        # Step 1: Estimate Riemannian metric
        metric = self.estimator.estimate(image)

        # Step 2: Compute geodesic distance map
        geo_map = GeodesicDistanceMap(metric, downsample=self.downsample)

        # Step 3: Pairwise distances
        # Detect gauge mode: tip1 == tip2 (from _unpack_for_3_point shim)
        is_gauge = (
            abs(tip1[0] - tip2[0]) < 1.0 and abs(tip1[1] - tip2[1]) < 1.0
        )

        if is_gauge:
            keypoints = [tuple(center), tuple(tip1)]
            kp_labels = ["center", "needle_tip"]
        else:
            keypoints = [tuple(center), tuple(tip1), tuple(tip2)]
            kp_labels = ["center", "tip1", "tip2"]

        geo_matrix = geo_map.geodesic_distance_matrix(keypoints)
        euclid_matrix = self._euclidean_matrix(keypoints)

        # Step 4: Build graphs
        geo_edges = self._build_graph(geo_matrix, threshold, kp_labels)
        euclid_edges = self._build_graph(euclid_matrix, threshold, kp_labels)

        # Step 5: Curvature analysis
        if is_gauge:
            pairs = [("center", "needle_tip", 0, 1)]
        else:
            pairs = [("center", "tip1", 0, 1), ("center", "tip2", 0, 2), ("tip1", "tip2", 1, 2)]

        curvature_ratios = {}
        for a_name, b_name, i, j in pairs:
            pair_key = f"{a_name}↔{b_name}"
            geo_d = float(geo_matrix[i, j])
            euc_d = float(euclid_matrix[i, j])
            ratio = geo_d / (euc_d + 1e-6)
            curvature_ratios[pair_key] = {
                "geodesic_px": round(geo_d, 2),
                "euclidean_px": round(euc_d, 2),
                "ratio": round(ratio, 3),
                "surface_curved": bool(ratio > 1.05),
            }

        # Step 6: Metric magnitude visualization (encoded)
        mag = self.estimator.visualize_metric_magnitude(metric)
        import base64
        _, buf = cv2.imencode('.jpg', mag)
        mag_b64 = base64.b64encode(buf).decode('utf-8')

        # Analysis summary
        avg_ratio = np.mean([v["ratio"] for v in curvature_ratios.values()])
        surface_flatness = "FLAT" if avg_ratio < 1.05 else ("MILDLY_CURVED" if avg_ratio < 1.3 else "HIGHLY_CURVED")

        return {
            "geodesic_distances": geo_matrix.tolist(),
            "euclidean_distances": euclid_matrix.tolist(),
            "curvature_ratios": curvature_ratios,
            "geodesic_graph": geo_edges,
            "euclidean_graph": euclid_edges,
            "manifold_analysis": {
                "average_curvature_ratio": round(float(avg_ratio), 3),
                "surface_classification": surface_flatness,
                "recommendation": (
                    "Use Euclidean distances — surface is approximately flat."
                    if surface_flatness == "FLAT"
                    else "Use geodesic distances — surface curvature is significant."
                ),
            },
            "metric_magnitude_b64": mag_b64,
        }

    @staticmethod
    def _euclidean_matrix(keypoints: List[Tuple]) -> np.ndarray:
        n = len(keypoints)
        D = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = float(np.linalg.norm(
                        np.array(keypoints[i]) - np.array(keypoints[j])
                    ))
        return D

    @staticmethod
    def _build_graph(dist_matrix: np.ndarray, threshold: float, labels: List[str]) -> List[Dict]:
        """Build edge list from distance matrix with threshold."""
        n = len(labels)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if dist_matrix[i, j] < threshold:
                    edges.append({
                        "from": labels[i],
                        "to": labels[j],
                        "distance": round(float(dist_matrix[i, j]), 2),
                    })
        return edges
