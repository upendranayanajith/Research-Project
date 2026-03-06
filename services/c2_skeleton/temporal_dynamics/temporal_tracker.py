"""
TemporalGraphTracker
=====================
Maintains per-session state for tracking clock-hand graph topology
across a sequence of video frames.

Integrates:
  - PersistentHomologyTracker  → Betti number history + birth/death events
  - OcclusionDetector          → classifies connectivity changes
  - Smoothed motion state      → velocity + acceleration per hand tip

This is the main class called by the FastAPI /track-temporal endpoint.
Each POST request adds a frame and returns the full temporal analysis.

Session lifecycle:
  - State persists in memory between requests
  - Call DELETE /reset-tracker to start fresh (e.g., for a new video)

Thread safety:
  - Not thread-safe by default (single-service assumption).
  - For production multi-session use, replace with a session store.
"""

import numpy as np
import math
from typing import List, Dict, Optional, Deque
from collections import deque
from .persistent_homology import PersistentHomologyTracker
from .occlusion_detector import OcclusionDetector, FrameState


class TemporalGraphTracker:
    """
    Frame-by-frame tracker for clock-hand graph topology.

    Parameters
    ----------
    max_history : int
        Maximum number of frames to keep in memory.
    homology_persistence_threshold : int
        Passed to PersistentHomologyTracker.
    occlusion_angle_threshold : float
        Passed to OcclusionDetector.
    """

    def __init__(
        self,
        max_history: int = 100,
        homology_persistence_threshold: int = 3,
        occlusion_angle_threshold: float = 20.0,
    ):
        self.max_history = max_history
        self.frame_count = 0

        self.homology_tracker = PersistentHomologyTracker(
            persistence_threshold=homology_persistence_threshold
        )
        self.occlusion_detector = OcclusionDetector(
            occlusion_angle_threshold=occlusion_angle_threshold
        )

        # Ring buffer of FrameState objects
        self.history: Deque[FrameState] = deque(maxlen=max_history)
        # Recent connectivity events for summary
        self.events_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_frame(
        self,
        center: Optional[List[float]],
        tip1: Optional[List[float]],
        tip2: Optional[List[float]],
    ) -> Dict:
        """
        Process a new frame and return temporal analysis.

        Parameters
        ----------
        center : [x, y] or None
        tip1   : [x, y] or None
        tip2   : [x, y] or None

        Returns
        -------
        dict with:
          frame_idx, betti_numbers, topology_events, motion_analysis,
          occlusion_analysis, summary_status
        """
        # Compute angles
        angle1 = self._angle_from_12(center, tip1) if (center and tip1) else None
        angle2 = self._angle_from_12(center, tip2) if (center and tip2) else None

        state = FrameState(
            frame_idx=self.frame_count,
            center=center,
            tip1=tip1,
            tip2=tip2,
            angle1=angle1,
            angle2=angle2,
            tip1_visible=tip1 is not None,
            tip2_visible=tip2 is not None,
        )
        self.history.append(state)

        # Update homology tracker
        homology_report = self.homology_tracker.add_frame(
            center_detected=center is not None,
            tip1_detected=tip1 is not None,
            tip2_detected=tip2 is not None,
        )

        # Motion analysis (velocity + acceleration)
        motion = self._compute_motion()

        # Occlusion analysis (only if connectivity event occurred)
        occlusion_analysis = None
        topology_events = homology_report.get("events", [])
        if topology_events:
            self.events_log.extend(topology_events)
            missing = self._identify_missing_hand(state)
            if missing:
                occlusion_analysis = self.occlusion_detector.classify(
                    history=list(self.history),
                    event_frame=self.frame_count,
                    missing_hand=missing,
                )

        # Summary status
        status = self._compute_status(homology_report, occlusion_analysis)

        self.frame_count += 1

        return {
            "frame_idx": state.frame_idx,
            "betti_numbers": homology_report["betti_numbers"],
            "topology_stable": homology_report["topology_stable"],
            "topology_events": topology_events,
            "motion_analysis": motion,
            "occlusion_analysis": occlusion_analysis,
            "angles": {"hand1": angle1, "hand2": angle2},
            "summary_status": status,
        }

    def get_session_summary(self) -> Dict:
        """
        Return a summary of the entire tracked session.
        """
        betti_series = self.homology_tracker.get_betti_series()
        persistence_diagram = self.homology_tracker.get_persistence_diagram()

        beta0_values = [b["beta0"] for b in betti_series]
        stable_pct = (sum(1 for v in beta0_values if v == 1) / max(len(beta0_values), 1)) * 100

        return {
            "total_frames": self.frame_count,
            "topology_stable_percentage": round(stable_pct, 1),
            "total_topology_events": len(self.events_log),
            "persistence_diagram": persistence_diagram,
            "betti_series": betti_series,
            "events_log": self.events_log,
        }

    def reset(self):
        """Reset the tracker for a new video sequence."""
        self.frame_count = 0
        self.history.clear()
        self.events_log.clear()
        self.homology_tracker.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_motion(self) -> Dict:
        """
        Compute velocity and acceleration for both hand tips.
        Uses the last 3 frames.
        """
        hist = list(self.history)
        result = {"tip1": {}, "tip2": {}}

        for hand in ["tip1", "tip2"]:
            positions = []
            for s in hist[-3:]:
                pos = s.tip1 if hand == "tip1" else s.tip2
                if pos is not None:
                    positions.append(np.array(pos[:2]))

            if len(positions) >= 2:
                vel = positions[-1] - positions[-2]
                speed = float(np.linalg.norm(vel))
                result[hand]["velocity_px_per_frame"] = vel.tolist()
                result[hand]["speed_px"] = round(speed, 2)

                if len(positions) >= 3:
                    vel_prev = positions[-2] - positions[-3]
                    accel = vel - vel_prev
                    result[hand]["acceleration"] = accel.tolist()
                    result[hand]["accel_magnitude"] = round(float(np.linalg.norm(accel)), 2)
            else:
                result[hand] = {"note": "insufficient history"}

        return result

    @staticmethod
    def _angle_from_12(center: List[float], tip: List[float]) -> float:
        """Clockwise angle from 12 o'clock, in degrees [0, 360)."""
        dx = tip[0] - center[0]
        dy = tip[1] - center[1]
        angle = math.degrees(math.atan2(dx, -dy))
        return (angle + 360) % 360

    @staticmethod
    def _identify_missing_hand(state: FrameState) -> Optional[str]:
        """Return which hand is missing in this frame."""
        if not state.tip1_visible:
            return "tip1"
        if not state.tip2_visible:
            return "tip2"
        return None

    @staticmethod
    def _compute_status(homology_report: Dict, occlusion_analysis: Optional[Dict]) -> str:
        """Generate a concise status string for this frame."""
        beta0 = homology_report["betti_numbers"]["beta0"]

        if beta0 == 1:
            return "NOMINAL"   # Both hands fully connected

        if occlusion_analysis:
            cls = occlusion_analysis["classification"]
            conf = occlusion_analysis["confidence"]
            return f"{cls} (confidence={conf:.2f})"

        if beta0 == 2:
            return "ONE_HAND_MISSING"
        if beta0 >= 3:
            return "MULTIPLE_COMPONENTS_DETECTED"

        return "UNKNOWN"
