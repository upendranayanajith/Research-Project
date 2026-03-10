"""
Temporal Graph Dynamics Sub-Package
=====================================
GAP 2 Research Innovation:
  "Persistent Homology Tracking for Dynamic Graph Structures in Video"

Core Insight from Algebraic Topology:
  - Graph structure has a "topological signature" (Betti numbers)
  - β₀ = number of connected components
  - β₁ = number of independent loops / cycles
  - Signature should change SLOWLY over time for smooth physical motion
  - Sudden changes → likely OCCLUSION, not real topology change

Modules:
  persistent_homology   — Betti number tracking, birth/death events
  temporal_tracker      — Frame-by-frame state management
  occlusion_detector    — Classify: real topology change vs occlusion
"""

from .persistent_homology import PersistentHomologyTracker
from .temporal_tracker import TemporalGraphTracker
from .occlusion_detector import OcclusionDetector

__all__ = [
    "PersistentHomologyTracker",
    "TemporalGraphTracker",
    "OcclusionDetector",
]
