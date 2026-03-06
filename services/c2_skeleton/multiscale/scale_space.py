"""
GaussianScaleSpace
==================
Builds a Gaussian scale-space pyramid over a clock image and detects
keypoint-like structures at each scale σ.

At each scale:
  1. Smooth image: I_σ = I * G_σ  (Gaussian convolution)
  2. Detect edges / blobs via Laplacian of Gaussian (LoG)
  3. Find local maxima → candidate keypoints at scale σ
  4. Build proximity graph G_σ connecting nearby keypoints

The result is a dict of ScaleGraph objects keyed by σ value.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class ScaleGraph:
    """
    Graph structure detected at a single scale σ.

    Attributes
    ----------
    sigma        : float — the Gaussian smoothing scale
    keypoints    : list of [x, y] positions detected at this scale
    edges        : list of (i, j) index pairs where i,j are keypoint indices
    edge_weights : list of Euclidean distances for each edge
    blob_sizes   : approximate blob radius at each keypoint
    """
    sigma: float
    keypoints: List[List[float]]
    edges: List[Tuple[int, int]]
    edge_weights: List[float]
    blob_sizes: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sigma": self.sigma,
            "num_keypoints": len(self.keypoints),
            "keypoints": self.keypoints,
            "edges": [list(e) for e in self.edges],
            "edge_weights": [round(w, 2) for w in self.edge_weights],
            "blob_sizes": [round(s, 2) for s in self.blob_sizes],
        }


class GaussianScaleSpace:
    """
    Constructs and manages a Gaussian scale-space pyramid.

    Parameters
    ----------
    scales : list of float
        Sigma values for the Gaussian kernels.
        Default [1, 2, 4, 8, 16] covers pixel → hand-shaft → full-hand.
    proximity_factor : float
        Keypoints within proximity_factor * sigma * image_size are connected.
    max_keypoints_per_scale : int
        Cap to avoid noisy over-detections at fine scales.
    """

    def __init__(
        self,
        scales: List[float] = None,
        proximity_factor: float = 0.15,
        max_keypoints_per_scale: int = 20,
    ):
        self.scales = scales if scales is not None else [1.0, 2.0, 4.0, 8.0, 16.0]
        self.proximity_factor = proximity_factor
        self.max_keypoints_per_scale = max_keypoints_per_scale

    def build_pyramid(self, image: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Apply Gaussian blur at each scale.

        Parameters
        ----------
        image : np.ndarray, shape (H, W, 3) or (H, W) in uint8

        Returns
        -------
        dict: {sigma: blurred_image}
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        gray = gray.astype(np.float32) / 255.0
        pyramid = {}
        for sigma in self.scales:
            # Kernel size: 6*sigma+1, rounded to odd
            ksize = int(6 * sigma + 1)
            ksize = ksize if ksize % 2 == 1 else ksize + 1
            blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)
            pyramid[sigma] = blurred
        return pyramid

    def detect_keypoints_at_scale(
        self, smoothed: np.ndarray, sigma: float
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Detect keypoints at a single scale via Laplacian of Gaussian (LoG).

        Steps:
          1. Compute LoG response: −σ² * ∇²I_σ
          2. Find local maxima of |LoG| above a threshold
          3. Return (x, y) positions + estimated blob radii

        Returns
        -------
        keypoints : [[x, y], ...]
        blob_sizes: [radius, ...]
        """
        H, W = smoothed.shape

        # LoG via difference of Gaussians (DoG) approximation
        ksize_outer = int(6 * sigma * 1.6 + 1)
        ksize_outer = ksize_outer if ksize_outer % 2 == 1 else ksize_outer + 1
        outer = cv2.GaussianBlur(smoothed, (ksize_outer, ksize_outer), sigmaX=sigma * 1.6)
        dog = smoothed - outer   # DoG ≈ LoG

        # Normalize and threshold
        dog_abs = np.abs(dog)
        threshold = max(dog_abs.mean() + 1.5 * dog_abs.std(), 0.005)

        # Local maxima via dilate comparison
        ksize_max = max(3, int(2 * sigma + 1))
        ksize_max = ksize_max if ksize_max % 2 == 1 else ksize_max + 1
        dilated = cv2.dilate(dog_abs, np.ones((ksize_max, ksize_max)))
        local_max_mask = (dog_abs == dilated) & (dog_abs > threshold)

        ys, xs = np.where(local_max_mask)
        if len(xs) == 0:
            return [], []

        # Score and cap
        scores = dog_abs[ys, xs]
        top_idx = np.argsort(-scores)[: self.max_keypoints_per_scale]
        keypoints = [[float(xs[i]), float(ys[i])] for i in top_idx]
        blob_sizes = [float(sigma * np.sqrt(2)) for _ in top_idx]

        return keypoints, blob_sizes

    def build_graph(
        self, keypoints: List[List[float]], blob_sizes: List[float],
        sigma: float, image_size: int = 500
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Build a proximity graph: connect keypoints within
        proximity_factor * sigma * image_size distance.
        """
        n = len(keypoints)
        radius = self.proximity_factor * sigma * image_size
        edges, weights = [], []

        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = np.array(keypoints[i]), np.array(keypoints[j])
                dist = float(np.linalg.norm(p1 - p2))
                if dist < radius:
                    edges.append((i, j))
                    weights.append(dist)

        return edges, weights

    def extract_all_scales(self, image: np.ndarray) -> Dict[float, ScaleGraph]:
        """
        Full pipeline: build pyramid → detect keypoints → build graphs.

        Returns
        -------
        dict: {sigma: ScaleGraph}
        """
        H, W = image.shape[:2]
        image_size = max(H, W)
        pyramid = self.build_pyramid(image)
        graphs: Dict[float, ScaleGraph] = {}

        for sigma, smoothed in pyramid.items():
            kpts, blobs = self.detect_keypoints_at_scale(smoothed, sigma)
            edges, weights = self.build_graph(kpts, blobs, sigma, image_size)
            graphs[sigma] = ScaleGraph(
                sigma=sigma,
                keypoints=kpts,
                edges=edges,
                edge_weights=weights,
                blob_sizes=blobs,
            )

        return graphs

    def render_graph_on_image(
        self, base_image: np.ndarray, graph: ScaleGraph,
        color_nodes=(0, 200, 255), color_edges=(0, 128, 255)
    ) -> np.ndarray:
        """Render a ScaleGraph overlay onto the base image (for LVM scoring)."""
        canvas = np.zeros((base_image.shape[0], base_image.shape[1], 3), dtype=np.uint8)
        kpts = graph.keypoints
        r = max(3, int(graph.sigma))

        for i, j in graph.edges:
            p1 = (int(kpts[i][0]), int(kpts[i][1]))
            p2 = (int(kpts[j][0]), int(kpts[j][1]))
            cv2.line(canvas, p1, p2, color_edges, max(1, r // 2))

        for pt in kpts:
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), r, color_nodes, -1)

        return canvas
