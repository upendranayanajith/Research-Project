"""
Probabilistic 3D Graph Reconstruction Sub-Package
==================================================
GAP 1 Research Innovation:
  "Probabilistic 3D Graph Reconstruction from Monocular Images
   via Differentiable Rendering"

Bayesian Formulation:
  P(G | I) ∝ P(I | G) × P(G)

  Where:
    G = 3D graph structure (clock hands in 3D space)
    I = 2D observed image / keypoints
    P(I | G) = rendering likelihood (geometric projection)
    P(G) = prior over plausible clock hand structures

Modules:
  graph_prior           — Learned/parametric prior over 3D structures
  topology_reconstructor — Core K-hypothesis sampling + MAP inference
  uncertainty           — Credible intervals and confidence scoring
  bayesian_inference    — Orchestrates the full P(G|I) computation
"""

from .graph_prior import LearnedGraphPrior
from .topology_reconstructor import TopologyReconstructor
from .uncertainty import UncertaintyEstimator
from .bayesian_inference import BayesianGraphInference

__all__ = [
    "LearnedGraphPrior",
    "TopologyReconstructor",
    "UncertaintyEstimator",
    "BayesianGraphInference",
]
