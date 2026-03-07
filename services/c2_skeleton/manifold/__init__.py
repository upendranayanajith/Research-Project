"""
Non-Euclidean Manifold Skeleton — GAP 4
========================================
"Intrinsic Geometry Learning for Skeleton Extraction on Non-Euclidean Manifolds"

Core Idea:
  Euclidean distance is WRONG on curved surfaces.
  A clock on a curved dashboard, or a gauge on a cylindrical tank, has
  a non-flat image domain.

  We learn a Riemannian metric tensor field g(x,y) from visual cues
  (image gradients, shading, texture distortion) and compute
  GEODESIC distances instead of straight-line Euclidean distances.

  dist_geodesic(A, B) = min_path ∫ sqrt(dx^T g(x) dx)

Modules:
  metric_estimator   — MetricTensorEstimator: 2×2 g(x,y) from image
  geodesic           — GeodesicDistanceMap: Dijkstra on Riemannian grid
  manifold_skeleton  — ManifoldSkeletonDetector: graph via geodesic edges
"""

from .metric_estimator import MetricTensorEstimator
from .geodesic import GeodesicDistanceMap
from .manifold_skeleton import ManifoldSkeletonDetector

__all__ = [
    "MetricTensorEstimator",
    "GeodesicDistanceMap",
    "ManifoldSkeletonDetector",
]
