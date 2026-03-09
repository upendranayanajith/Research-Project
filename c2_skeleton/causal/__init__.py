"""
Causal Structure Discovery — GAP 5
=====================================
"Causal Graph Discovery from Visual Dynamics Using Granger Causality
 in Latent Space"

Core Idea:
  In a mechanical system (clock, linkage, gear train) we observe
  CORRELATED motion. But which element DRIVES which?

  Granger Causality Test:
    X Granger-causes Y if:
      Predicting Y from {Y_past + X_past} is significantly better
      than predicting from {Y_past} alone.

  Applied to keypoint trajectories:
    1. Extract per-keypoint position time series
    2. Fit VAR model (restricted: Y only, full: X + Y)
    3. F-test on residual improvement
    4. Build directed causal graph where edges = significant causality

Modules:
  granger           — GrangerCausalityTest: VAR + F-test
  causal_discovery  — CausalSkeletonDiscovery: pairwise tests → graph
"""

from .granger import GrangerCausalityTest, GrangerResult
from .causal_discovery import CausalSkeletonDiscovery

__all__ = [
    "GrangerCausalityTest",
    "GrangerResult",
    "CausalSkeletonDiscovery",
]
