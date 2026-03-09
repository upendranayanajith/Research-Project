"""
GeodesicDistanceMap
====================
Computes geodesic distances on a pixel grid weighted by a Riemannian
metric tensor field, using Dijkstra's algorithm.

Geodesic distance:
  d(A, B) = min_path ∫ sqrt(v^T g(p) v) dp

  where g(p) is the 2×2 metric tensor at point p,
  and v is the direction of travel.

On a flat surface (g = I): geodesic = Euclidean distance.
On a curved surface: geodesic avoids high-curvature / high-gradient regions.

Implementation:
  - Pixel grid as graph: each pixel connected to 8-neighbors
  - Edge weight between (y1,x1) and (y2,x2):
      w = 0.5 * (ds(y1,x1,dy,dx) + ds(y2,x2,-dy,-dx))
      where ds = sqrt(v^T g(y,x) v) is the Riemannian length element
  - Dijkstra from source point → all-points shortest geodesic distance map

For efficiency only computes distances from a set of source points,
not full all-pairs (which would be O(H²W²)).
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict


class GeodesicDistanceMap:
    """
    Computes geodesic distance maps on Riemannian pixel grids.

    Parameters
    ----------
    metric_field : np.ndarray, shape (H, W, 2, 2)
        Per-pixel Riemannian metric tensor from MetricTensorEstimator.
    downsample : int
        Downsample factor to speed up computation (default 4).
        Distances are scaled back up after computation.
    """

    # 8-neighbour offsets: (dy, dx)
    NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(self, metric_field: np.ndarray, downsample: int = 4):
        self.downsample = downsample
        self.metric_field = self._downsample_metric(metric_field, downsample)
        self.H, self.W = self.metric_field.shape[:2]

    def geodesic_distance(
        self, source: Tuple[int, int], target: Tuple[int, int]
    ) -> float:
        """
        Compute geodesic distance from source to target pixel.

        Parameters
        ----------
        source : (x, y) in original image coordinates
        target : (x, y) in original image coordinates

        Returns
        -------
        Geodesic distance (in downsampled pixel units × downsample factor)
        """
        # Convert to downsampled grid coordinates
        sy, sx = int(source[1] / self.downsample), int(source[0] / self.downsample)
        ty, tx = int(target[1] / self.downsample), int(target[0] / self.downsample)

        sy = np.clip(sy, 0, self.H - 1)
        sx = np.clip(sx, 0, self.W - 1)
        ty = np.clip(ty, 0, self.H - 1)
        tx = np.clip(tx, 0, self.W - 1)

        dist_map = self._dijkstra(sy, sx)
        return float(dist_map[ty, tx]) * self.downsample

    def geodesic_distance_matrix(
        self, points: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Compute pairwise geodesic distances between a set of points.

        Parameters
        ----------
        points : list of (x, y) pairs in original image coordinates

        Returns
        -------
        D : np.ndarray, shape (N, N) — symmetric distance matrix
        """
        N = len(points)
        D = np.zeros((N, N), dtype=np.float32)

        for i, src in enumerate(points):
            sy = np.clip(int(src[1] / self.downsample), 0, self.H - 1)
            sx = np.clip(int(src[0] / self.downsample), 0, self.W - 1)
            dist_map = self._dijkstra(sy, sx)

            for j, tgt in enumerate(points):
                if i == j:
                    continue
                ty = np.clip(int(tgt[1] / self.downsample), 0, self.H - 1)
                tx = np.clip(int(tgt[0] / self.downsample), 0, self.W - 1)
                D[i, j] = float(dist_map[ty, tx]) * self.downsample

        return D

    def euclidean_vs_geodesic_ratio(
        self, point_a: Tuple[int, int], point_b: Tuple[int, int]
    ) -> dict:
        """
        Compare Euclidean vs geodesic distance between two points.
        Ratio > 1 indicates the surface is curved between those points.
        """
        euclid = float(np.linalg.norm(
            np.array(point_a, dtype=float) - np.array(point_b, dtype=float)
        ))
        geo = self.geodesic_distance(point_a, point_b)
        ratio = geo / (euclid + 1e-6)
        return {
            "euclidean": round(euclid, 2),
            "geodesic": round(geo, 2),
            "ratio": round(ratio, 3),
            "surface_is_curved": bool(ratio > 1.05),
        }

    # ------------------------------------------------------------------
    # Internal: Dijkstra
    # ------------------------------------------------------------------

    def _dijkstra(self, start_y: int, start_x: int) -> np.ndarray:
        """
        Dijkstra from (start_y, start_x) → distance map over downsampled grid.
        """
        dist = np.full((self.H, self.W), np.inf, dtype=np.float32)
        dist[start_y, start_x] = 0.0

        # Priority queue: (distance, y, x)
        heap = [(0.0, start_y, start_x)]

        while heap:
            d, y, x = heapq.heappop(heap)
            if d > dist[y, x]:
                continue

            for dy, dx in self.NEIGHBOURS:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= self.H or nx < 0 or nx >= self.W:
                    continue

                # Riemannian edge weight: average ds at both endpoints
                ds_src = self._riemannian_ds(y, x, dy, dx)
                ds_tgt = self._riemannian_ds(ny, nx, -dy, -dx)
                w = 0.5 * (ds_src + ds_tgt)

                new_dist = d + w
                if new_dist < dist[ny, nx]:
                    dist[ny, nx] = new_dist
                    heapq.heappush(heap, (new_dist, ny, nx))

        return dist

    def _riemannian_ds(self, y: int, x: int, dy: int, dx: int) -> float:
        """
        Compute Riemannian length element ds = sqrt([dx,dy]^T g [dx,dy]).
        """
        g = self.metric_field[y, x]    # 2×2
        v = np.array([float(dx), float(dy)])
        val = v @ g @ v
        return float(np.sqrt(max(val, 0.0)))

    @staticmethod
    def _downsample_metric(metric: np.ndarray, factor: int) -> np.ndarray:
        """Downsample metric field by averaging over factor×factor blocks."""
        if factor <= 1:
            return metric
        H, W = metric.shape[:2]
        H_d = H // factor
        W_d = W // factor
        downsampled = np.zeros((H_d, W_d, 2, 2), dtype=np.float32)
        for i in range(H_d):
            for j in range(W_d):
                block = metric[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
                downsampled[i, j] = block.mean(axis=(0, 1))
        return downsampled
