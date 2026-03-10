"""
LVM-Guided Multi-Scale Detector — Recommended Combination Pipeline
===================================================================
"LVM-Guided Multi-Scale Skeleton Detection with Uncertainty Quantification"

This is the RECOMMENDED publication-ready pipeline that combines:
  1. MultiScaleSkeletonExtractor (GAP 3) — optimal scale σ*
  2. BayesianGraphInference (GAP 1) — probabilistic 3D + uncertainty
  3. LVMTemporalSmoother — jitter-robust temporal consistency

The Pitch:
  "We discover that optimal detection scale varies by clock design
   (ornate vs minimal). We use LVM embeddings as a learned scale oracle,
   achieving improved accuracy over fixed-scale methods, with statistical
   confidence bounds via posterior uncertainty quantification."

Modules:
  lvm_multiscale_detector — LVMMultiScaleDetector: full pipeline
"""

from .lvm_multiscale_detector import LVMMultiScaleDetector, CombinedResult

__all__ = [
    "LVMMultiScaleDetector",
    "CombinedResult",
]
