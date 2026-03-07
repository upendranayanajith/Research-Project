"""
PersistentHomologyTracker
==========================
Tracks topological features (Betti numbers) of the clock-hand graph
over time to distinguish occlusion from real topology changes.

Algebraic Topology Background
-------------------------------
Betti numbers describe topological structure:
  β₀ = number of connected components
       (2 when both hands detected; 3 if one breaks; 1 if merged)
  β₁ = number of independent loops/cycles
       (clock hands form no loops → β₁ = 0 always)
  β₂ = voids (3D concept, not relevant here)

Key Principle:
  For smooth physical motion, Betti numbers should be CONSTANT or
  change SLOWLY. A sudden jump in β₀ (components appearing/disappearing)
  with no gradual transition signals OCCLUSION rather than true topology
  change (e.g., a hand breaking off is physically impossible for a clock).

Persistence:
  A topological feature "born" at frame t_birth and "dies" at t_death
  has lifetime = t_death - t_birth.
  Short-lived features → noise / occlusion.
  Long-lived features → real structural changes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class TopologicalFeature:
    """
    A topological feature (connected component, loop) across time.

    Attributes
    ----------
    feature_type : str
        "component" (β₀) or "loop" (β₁).
    birth_frame : int
        Frame index where this feature appeared.
    death_frame : Optional[int]
        Frame index where this feature disappeared (None if still alive).
    betti_value_change : int
        +1 for birth, -1 for death.
    """
    feature_type: str
    birth_frame: int
    death_frame: Optional[int] = None
    betti_value_change: int = 1
    metadata: dict = field(default_factory=dict)

    @property
    def lifetime(self) -> Optional[int]:
        if self.death_frame is None:
            return None
        return self.death_frame - self.birth_frame

    @property
    def is_alive(self) -> bool:
        return self.death_frame is None

    def to_dict(self) -> dict:
        return {
            "feature_type": self.feature_type,
            "birth_frame": self.birth_frame,
            "death_frame": self.death_frame,
            "lifetime": self.lifetime,
            "is_alive": self.is_alive,
        }


@dataclass
class BettiNumbers:
    """Betti numbers for a graph at a single frame."""
    beta0: int    # connected components
    beta1: int    # loops (cycles)
    frame: int = 0

    def to_dict(self) -> dict:
        return {"beta0": self.beta0, "beta1": self.beta1, "frame": self.frame}


class PersistentHomologyTracker:
    """
    Computes and tracks Betti numbers of the clock-hand graph over time.

    The clock skeleton is a simple graph:
      Nodes: {center, tip1, tip2}  — always 3 nodes
      Edges: {(center, tip1), (center, tip2)}  — when both hands detected

    Betti numbers:
      β₀: 1 (fully connected) when both hands detected
          2 when one hand missing
          3 when center is isolated too
      β₁: always 0 (a tree has no cycles)

    The tracker records birth/death events and flags sudden changes.

    Parameters
    ----------
    persistence_threshold : int
        Minimum lifetime required for a feature to count as "real".
        Features shorter than this → classified as noise/occlusion.
    """

    def __init__(self, persistence_threshold: int = 3):
        self.persistence_threshold = persistence_threshold
        self.history: List[BettiNumbers] = []
        self.features: List[TopologicalFeature] = []
        self._prev_beta0: Optional[int] = None
        self._frame_count: int = 0

    def add_frame(
        self,
        center_detected: bool,
        tip1_detected: bool,
        tip2_detected: bool,
    ) -> Dict:
        """
        Add a new frame and compute topological events.

        Parameters
        ----------
        center_detected : bool  — was the center keypoint found?
        tip1_detected   : bool  — was tip1 found?
        tip2_detected   : bool  — was tip2 found?

        Returns
        -------
        dict with frame Betti numbers and any topological events
        """
        betti = self._compute_betti(center_detected, tip1_detected, tip2_detected)
        betti.frame = self._frame_count
        self.history.append(betti)

        events = self._detect_events(betti)

        report = {
            "frame": self._frame_count,
            "betti_numbers": betti.to_dict(),
            "events": events,
            "features": [f.to_dict() for f in self.features],
            "topology_stable": len(events) == 0,
        }

        self._prev_beta0 = betti.beta0
        self._frame_count += 1
        return report

    def get_persistence_diagram(self) -> List[Dict]:
        """
        Returns all feature birth/death pairs (persistence diagram).
        Short-lifetime features indicate noise or occlusion.
        """
        return [f.to_dict() for f in self.features]

    def get_betti_series(self) -> List[Dict]:
        """Time series of Betti numbers across all frames."""
        return [b.to_dict() for b in self.history]

    def is_topology_stable(self, window: int = 5) -> bool:
        """
        Check if Betti numbers have been stable over the last `window` frames.
        """
        if len(self.history) < 2:
            return True
        recent = self.history[-min(window, len(self.history)):]
        b0_values = [b.beta0 for b in recent]
        return len(set(b0_values)) == 1

    def reset(self):
        """Reset all tracked state."""
        self.history.clear()
        self.features.clear()
        self._prev_beta0 = None
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_betti(center: bool, tip1: bool, tip2: bool) -> BettiNumbers:
        """
        Compute Betti numbers for the clock-hand graph.

        Graph topology:
          - If all 3 nodes present + 2 edges → β₀=1, β₁=0
          - If center + one tip → β₀=1, β₁=0
          - If only tips (no center) → β₀=2, β₁=0
          - If only one node → β₀=1, β₁=0
          - If no nodes → β₀=0, β₁=0
        """
        nodes = int(center) + int(tip1) + int(tip2)
        edges = int(center and tip1) + int(center and tip2)

        if nodes == 0:
            return BettiNumbers(beta0=0, beta1=0)

        # β₀ = nodes - edges (for a forest/tree, β₁=0)
        beta0 = nodes - edges
        return BettiNumbers(beta0=beta0, beta1=0)

    def _detect_events(self, current: BettiNumbers) -> List[Dict]:
        """Detect birth/death of topological features between frames."""
        events = []

        if self._prev_beta0 is None:
            # First frame: birth of initial components
            for _ in range(current.beta0):
                f = TopologicalFeature(
                    feature_type="component",
                    birth_frame=self._frame_count,
                )
                self.features.append(f)
            return events

        delta = current.beta0 - self._prev_beta0

        if delta > 0:
            # New components born (connectivity loss — could be occlusion)
            for _ in range(delta):
                f = TopologicalFeature(
                    feature_type="component",
                    birth_frame=self._frame_count,
                    betti_value_change=+1,
                )
                self.features.append(f)
                events.append({
                    "type": "COMPONENT_BIRTH",
                    "frame": self._frame_count,
                    "description": "Graph connectivity decreased — possible occlusion or detection loss",
                })

        elif delta < 0:
            # Components merged (connectivity restored)
            alive = [f for f in self.features if f.is_alive and f.feature_type == "component"]
            killed = min(abs(delta), len(alive))
            for i in range(killed):
                alive[-(i + 1)].death_frame = self._frame_count
                events.append({
                    "type": "COMPONENT_DEATH",
                    "frame": self._frame_count,
                    "lifetime": alive[-(i + 1)].lifetime,
                    "description": "Graph connectivity restored — occlusion ended or detection recovered",
                })

        return events
