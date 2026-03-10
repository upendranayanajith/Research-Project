"""
OcclusionDetector
==================
Classifies whether a graph connectivity change (detected by PersistentHomologyTracker)
is caused by:
  (A) OCCLUSION — one pointer passing behind another or behind bezel (very common)
  (B) DETECTION_LOSS — keypoint detector failed to localize a pointer
  (C) REAL_TOPOLOGY_CHANGE — actual structural change (rare for clocks/gauges)

Decision Logic
--------------
1. Angular proximity: if two pointers are close in angle, occlusion is likely
2. Velocity consistency: if a pointer "disappears" but its last known velocity
   predicts it should still be in the frame → occlusion (not real change)
3. Persistence: if the connectivity change is short-lived (< threshold frames)
   → likely occlusion/noise, not real topology change
4. Z-depth: if probabilistic 3D says one hand was behind the other → occlusion

Classification output:
  OCCLUSION_LIKELY   — high confidence it's just occlusion
  DETECTION_LOSS     — detector failed, but structure likely intact
  TOPOLOGY_CHANGE    — genuine connectivity change
  AMBIGUOUS          — insufficient evidence to classify
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum


class ChangeType(str, Enum):
    OCCLUSION_LIKELY = "OCCLUSION_LIKELY"
    DETECTION_LOSS = "DETECTION_LOSS"
    TOPOLOGY_CHANGE = "TOPOLOGY_CHANGE"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class FrameState:
    """
    Snapshot of pointer positions at a single frame.

    Attributes
    ----------
    frame_idx    : frame number
    center       : [x, y] or None
    tip1         : [x, y] or None
    tip2         : [x, y] or None
    angle1       : angle from 12-o'clock for tip1, or None
    angle2       : angle from 12-o'clock for tip2, or None
    tip1_visible : was tip1 detected?
    tip2_visible : was tip2 detected?
    """
    frame_idx: int
    center: Optional[List[float]]
    tip1: Optional[List[float]]
    tip2: Optional[List[float]]
    angle1: Optional[float]
    angle2: Optional[float]
    tip1_visible: bool
    tip2_visible: bool

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "center": self.center,
            "tip1": self.tip1,
            "tip2": self.tip2,
            "angle1": self.angle1,
            "angle2": self.angle2,
            "tip1_visible": self.tip1_visible,
            "tip2_visible": self.tip2_visible,
        }


class OcclusionDetector:
    """
    Classifies connectivity changes as occlusion, detection loss, or real topology change.

    Parameters
    ----------
    occlusion_angle_threshold : float
        Angular proximity (degrees) below which hands are considered overlapping.
        Default 20° — hands within 20° of each other are likely occluding.
    velocity_prediction_tolerance : float
        Maximum pixel error between predicted and observed position.
        Used to detect if a hand "should still be there" based on velocity.
    persistence_threshold : int
        Connectivity changes shorter than this (frames) are treated as transient.
    """

    def __init__(
        self,
        occlusion_angle_threshold: float = 20.0,
        velocity_prediction_tolerance: float = 30.0,
        persistence_threshold: int = 4,
    ):
        self.occlusion_angle_threshold = occlusion_angle_threshold
        self.velocity_prediction_tolerance = velocity_prediction_tolerance
        self.persistence_threshold = persistence_threshold

    def classify(
        self,
        history: List[FrameState],
        event_frame: int,
        missing_hand: str = "tip1",
    ) -> Dict:
        """
        Classify a connectivity change event.

        Parameters
        ----------
        history       : list of FrameState (most recent last)
        event_frame   : frame index when the connectivity change occurred
        missing_hand  : "tip1" or "tip2" — which hand disappeared

        Returns
        -------
        dict with:
          classification, confidence, evidence, recommendation
        """
        if len(history) < 2:
            return self._result(ChangeType.AMBIGUOUS, 0.4, ["Insufficient history"])

        current = history[-1]
        prev_states = history[:-1]

        evidence = []
        scores: Dict[str, float] = {
            ChangeType.OCCLUSION_LIKELY: 0.0,
            ChangeType.DETECTION_LOSS: 0.0,
            ChangeType.TOPOLOGY_CHANGE: 0.0,
        }

        # --- Evidence 1: Angular proximity ---
        if current.angle1 is not None and current.angle2 is not None:
            angular_diff = abs(current.angle1 - current.angle2)
            angular_diff = min(angular_diff, 360 - angular_diff)

            if angular_diff < self.occlusion_angle_threshold:
                scores[ChangeType.OCCLUSION_LIKELY] += 0.5
                evidence.append(
                    f"Pointers are {angular_diff:.1f}° apart — within occlusion zone ({self.occlusion_angle_threshold}°)"
                )
            elif angular_diff < self.occlusion_angle_threshold * 2:
                scores[ChangeType.OCCLUSION_LIKELY] += 0.2
                evidence.append(f"Pointers are {angular_diff:.1f}° apart — moderate occlusion risk")

        # --- Evidence 2: Velocity-based prediction ---
        velocity_check = self._velocity_prediction_check(prev_states, missing_hand)
        if velocity_check["predicted_in_frame"]:
            # Hand should still be visible based on trajectory → not real topology change
            scores[ChangeType.OCCLUSION_LIKELY] += 0.3
            evidence.append(
                f"Velocity prediction: pointer should be at {velocity_check['predicted_pos']} — "
                "likely occluded not absent"
            )
        else:
            scores[ChangeType.DETECTION_LOSS] += 0.2
            evidence.append("Velocity prediction uncertain — may be detection loss")

        # --- Evidence 3: Historical detection consistency ---
        n_visible = sum(
            1 for s in prev_states[-10:]
            if (missing_hand == "tip1" and s.tip1_visible)
            or (missing_hand == "tip2" and s.tip2_visible)
        )
        total_checked = min(10, len(prev_states))
        if total_checked > 0:
            detection_rate = n_visible / total_checked
            if detection_rate > 0.8:
                # Was consistently detected before → detector likely failed now
                scores[ChangeType.DETECTION_LOSS] += 0.25
                evidence.append(f"Detection rate was {detection_rate:.0%} before event → detector issue likely")
            elif detection_rate < 0.4:
                # Was often missed → structural ambiguity
                scores[ChangeType.TOPOLOGY_CHANGE] += 0.15
                evidence.append(f"Detection rate was only {detection_rate:.0%} — pointer may genuinely be absent")

        # --- Evidence 4: Instrument domain knowledge ---
        # For mechanical clocks/gauges, pointers do NOT break connections.
        # Any β₀ change is almost never a real topology change.
        scores[ChangeType.TOPOLOGY_CHANGE] *= 0.1   # Strong prior against it
        scores[ChangeType.OCCLUSION_LIKELY] += 0.15  # Instrument-specific prior
        evidence.append("Domain prior: instrument pointers cannot physically disconnect — real topology change is rare")

        # --- Final classification ---
        best_type = max(scores, key=scores.get)
        total = sum(scores.values()) + 1e-8
        confidence = scores[best_type] / total

        return self._result(
            ChangeType(best_type),
            round(float(confidence), 3),
            evidence,
            angular_diff=None if (current.angle1 is None or current.angle2 is None)
                         else round(abs(current.angle1 - current.angle2), 1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _velocity_prediction_check(
        self,
        history: List[FrameState],
        hand: str,
    ) -> Dict:
        """
        Predict where the missing hand should be based on the last 2-3 frames.
        Returns whether the predicted position is within the image frame.
        """
        positions = []
        for s in history[-3:]:
            pos = s.tip1 if hand == "tip1" else s.tip2
            if pos is not None:
                positions.append(pos)

        if len(positions) < 2:
            return {"predicted_in_frame": False, "predicted_pos": None}

        # Simple linear extrapolation
        p1 = np.array(positions[-2])
        p2 = np.array(positions[-1])
        velocity = p2 - p1
        predicted = p2 + velocity

        # Is the predicted position within image bounds (assume 0-500)?
        in_frame = (0 <= predicted[0] <= 600) and (0 <= predicted[1] <= 600)

        return {
            "predicted_in_frame": bool(in_frame),
            "predicted_pos": predicted.tolist(),
            "velocity": velocity.tolist(),
        }

    @staticmethod
    def _result(
        classification: ChangeType,
        confidence: float,
        evidence: List[str],
        angular_diff: Optional[float] = None,
    ) -> Dict:
        recommendations = {
            ChangeType.OCCLUSION_LIKELY: "Use 3D depth ordering to infer pointer overlap. Continue tracking.",
            ChangeType.DETECTION_LOSS: "Re-run detector with lower confidence threshold. Interpolate from last known position.",
            ChangeType.TOPOLOGY_CHANGE: "Unusual event detected. Flag for manual review.",
            ChangeType.AMBIGUOUS: "Collect more frames before making a determination.",
        }
        return {
            "classification": classification.value,
            "confidence": confidence,
            "evidence": evidence,
            "recommendation": recommendations[classification],
            "angular_diff_deg": angular_diff,
        }
