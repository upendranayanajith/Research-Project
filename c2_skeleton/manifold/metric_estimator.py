"""
MetricTensorEstimator
======================
Estimates the Riemannian metric tensor field g(x,y) from a single image.

The metric tensor g at pixel (x,y) is a 2×2 positive-definite matrix:
  g = [[g11, g12],
       [g12, g22]]

It encodes "how to measure distance" in the local neighbourhood.
On a flat surface g = I (identity).
On a curved surface g encodes curvature-induced length distortion.

Estimation strategy (from monocular image):
  1. Compute image gradient magnitude ∇I (strong gradients → boundary)
  2. Compute Hessian H_I (second-order curvature of intensity)
  3. Use structure tensor S = ∇I ⊗ ∇I (local orientation field)
  4. Metric tensor: g = I + λ * S / (max(S) + ε)
     → Identity on flat regions, anisotropic near edges/curves

This is an approximation — a learned metric network (MetricNet) would
produce better results with training data.

The resulting metric makes distances LARGER along edges (discourages
crossing object boundaries) and SMALLER along smooth surface regions.
"""

import numpy as np
import cv2
from typing import Tuple


class MetricTensorEstimator:
    """
    Estimates per-pixel Riemannian metric tensors from a grayscale image.

    Parameters
    ----------
    anisotropy : float
        Strength of edge-induced anisotropy (λ). Higher → stronger
        boundary avoidance in geodesic paths. Default 5.0.
    smoothing_sigma : float
        Pre-smoothing sigma to reduce noise before gradient computation.
    """

    def __init__(self, anisotropy: float = 5.0, smoothing_sigma: float = 1.5):
        self.anisotropy = anisotropy
        self.smoothing_sigma = smoothing_sigma

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the per-pixel metric tensor field.

        Parameters
        ----------
        image : np.ndarray, shape (H, W, 3) or (H, W), uint8

        Returns
        -------
        metric_field : np.ndarray, shape (H, W, 2, 2), float32
            metric_field[y, x] = 2×2 positive-definite metric tensor at (x,y).
        """
        gray = self._to_gray(image)
        H, W = gray.shape

        # Pre-smooth
        ksize = int(6 * self.smoothing_sigma + 1) | 1
        smooth = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=self.smoothing_sigma)

        # Image gradients (Sobel)
        Ix = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)

        # Structure tensor components
        Jxx = Ix * Ix   # (H, W)
        Jyy = Iy * Iy
        Jxy = Ix * Iy

        # Smooth structure tensor (local integration window)
        w_ksize = max(3, int(4 * self.smoothing_sigma + 1) | 1)
        Jxx = cv2.GaussianBlur(Jxx, (w_ksize, w_ksize), sigmaX=2 * self.smoothing_sigma)
        Jyy = cv2.GaussianBlur(Jyy, (w_ksize, w_ksize), sigmaX=2 * self.smoothing_sigma)
        Jxy = cv2.GaussianBlur(Jxy, (w_ksize, w_ksize), sigmaX=2 * self.smoothing_sigma)

        # Normalize structure tensor: S = J / (||J||_F + ε)
        J_norm = np.sqrt(Jxx**2 + Jyy**2 + 2 * Jxy**2) + 1e-8
        Sxx = Jxx / J_norm
        Syy = Jyy / J_norm
        Sxy = Jxy / J_norm

        # Metric tensor: g = I + λ * S
        # Shape: (H, W, 2, 2)
        metric = np.zeros((H, W, 2, 2), dtype=np.float32)
        lam = self.anisotropy

        metric[:, :, 0, 0] = 1.0 + lam * Sxx   # g11
        metric[:, :, 1, 1] = 1.0 + lam * Syy   # g22
        metric[:, :, 0, 1] = lam * Sxy          # g12
        metric[:, :, 1, 0] = lam * Sxy          # g21 (symmetric)

        return metric

    def riemannian_distance_element(
        self, metric: np.ndarray, y: int, x: int, dy: float, dx: float
    ) -> float:
        """
        Compute ds = sqrt(v^T g(x,y) v) where v = [dx, dy].

        This is the Riemannian length element at pixel (x,y) for
        displacement vector (dx, dy).
        """
        g = metric[y, x]   # 2×2
        v = np.array([dx, dy], dtype=np.float64)
        val = v @ g @ v
        return float(np.sqrt(max(val, 0.0)))

    def visualize_metric_magnitude(self, metric: np.ndarray) -> np.ndarray:
        """
        Return a (H, W) float32 image of metric determinant magnitude.
        Useful for debugging: bright = high curvature region.
        """
        g11 = metric[:, :, 0, 0]
        g22 = metric[:, :, 1, 1]
        g12 = metric[:, :, 0, 1]
        det = g11 * g22 - g12 * g12
        det = np.clip(det, 0, None)
        det_norm = (det / (det.max() + 1e-8) * 255).astype(np.uint8)
        return det_norm

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        return gray.astype(np.float32) / 255.0
