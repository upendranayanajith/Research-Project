"""
Multi-Scale Skeleton Extraction — GAP 3
========================================
"Scale-Space Graph Theory for Multi-Resolution Skeleton Extraction"

Core Idea:
  Structure exists at MULTIPLE scales simultaneously.
  A clock hand is a line at σ=20, a rectangle at σ=5, pixels at σ=1.
  At each scale the graph G_σ = {V_σ, E_σ} has different connectivity.

  The semantic scale σ* is the one where the structure matches the
  visual intent — determined by an LVM-style similarity oracle.

Mathematical framework:
  I_σ(x,y) = I(x,y) * G_σ(x,y)       [Gaussian convolution]
  G_σ = detect_graph(I_σ)              [keypoints at scale σ]
  σ* = argmax_σ LVM_score(I, render(G_σ))

Modules:
  scale_space          — GaussianScaleSpace: pyramid construction + keypoints
  lvm_scale_selector   — LVMScaleSelector: structural similarity scoring
  multi_scale_extractor — MultiScaleSkeletonExtractor: orchestrator
"""

from .scale_space import GaussianScaleSpace, ScaleGraph
from .lvm_scale_selector import LVMScaleSelector
from .multi_scale_extractor import MultiScaleSkeletonExtractor

__all__ = [
    "GaussianScaleSpace",
    "ScaleGraph",
    "LVMScaleSelector",
    "MultiScaleSkeletonExtractor",
]
