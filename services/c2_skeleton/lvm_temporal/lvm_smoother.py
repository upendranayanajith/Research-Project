"""
LVMTemporalSmoother
====================
Applies LVM-based temporal gating to smooth jittery skeleton detections.

Core Idea:
  Raw detector output is noisy frame-to-frame. Traditional approaches:
    - Kalman filter: only tracks positions (scalar noise model)
    - Optical flow: pixel-level, not structure-aware
    - Moving average: blurs genuine structure changes too

  Our approach: validate each new skeleton against the PREVIOUS using
  an LVM-proxy embedding distance.

  Algorithm:
    For each new frame t with skeleton S_t:
      1. Compute embedding e_t = encode(S_t)
      2. Compute cosine distance d = dist(e_t, e_{t-1})
      3. If d < threshold:           → accept S_t (smooth motion)
         elif d < 2 * threshold:     → interpolate (large but plausible)
         else:                       → reject (likely detection error)
      4. Store smoothed skeleton S_t_smooth

Decision thresholds (calibrated for clock hands):
  ACCEPT_THRESHOLD  = 0.15   (< 15% embedding distance = smooth)
  BLEND_THRESHOLD   = 0.40   (15-40% = interpolate)
  > BLEND_THRESHOLD = reject  (> 40% = detection likely wrong)

Novel contribution:
  LVM as structural temporal validator — not just position smoothing
  but SEMANTIC consistency check on the skeleton configuration.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import deque
from .skeleton_encoder import SkeletonEncoder


@dataclass
class SmoothedSkeleton:
    """
    Output of LVMTemporalSmoother for a single frame.

    Attributes
    ----------
    frame_idx         : int
    center, tip1, tip2 : smoothed keypoint positions
    raw_center, ...   : original raw keypoints before smoothing
    action            : "ACCEPTED", "INTERPOLATED", "REJECTED_PREV_USED"
    embedding_distance : cosine distance from previous frame's embedding
    confidence        : 1 - normalized_distance — how confident in acceptance
    """
    frame_idx: int
    center: List[float]
    tip1: List[float]
    tip2: List[float]
    raw_center: Optional[List[float]]
    raw_tip1: Optional[List[float]]
    raw_tip2: Optional[List[float]]
    action: str
    embedding_distance: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "center": self.center,
            "tip1": self.tip1,
            "tip2": self.tip2,
            "action": self.action,
            "embedding_distance": round(self.embedding_distance, 4),
            "confidence": round(self.confidence, 3),
        }


class LVMTemporalSmoother:
    """
    Smooths skeleton detections using LVM-proxy embedding distance gating.

    Parameters
    ----------
    accept_threshold : float
        Cosine distance below which skeletons are directly accepted.
    blend_threshold  : float
        Cosine distance below which skeletons are interpolated.
        Above this → reject (use previous).
    blend_alpha      : float
        Blending weight for interpolation: (1-α)*prev + α*curr.
        Default 0.7 → favours new detection but smoothed.
    history_size     : int
        Number of past smoothed skeletons to keep.
    """

    def __init__(
        self,
        accept_threshold: float = 0.15,
        blend_threshold: float = 0.40,
        blend_alpha: float = 0.70,
        history_size: int = 50,
    ):
        self.accept_threshold = accept_threshold
        self.blend_threshold = blend_threshold
        self.blend_alpha = blend_alpha
        self.encoder = SkeletonEncoder()

        self._history: deque = deque(maxlen=history_size)
        self._prev_embedding: Optional[np.ndarray] = None
        self._frame_count = 0
        self._stats = {"accepted": 0, "interpolated": 0, "rejected": 0}

    def add_frame(
        self,
        center: List[float],
        tip1: List[float],
        tip2: List[float],
        original_size: int = 500,
    ) -> SmoothedSkeleton:
        """
        Process a new raw skeleton detection.

        Returns the smoothed skeleton for this frame.
        """
        # Compute embedding for current frame
        curr_embedding = self.encoder.encode(center, tip1, tip2, original_size)

        if self._prev_embedding is None:
            # First frame — accept unconditionally
            result = SmoothedSkeleton(
                frame_idx=self._frame_count,
                center=center, tip1=tip1, tip2=tip2,
                raw_center=center, raw_tip1=tip1, raw_tip2=tip2,
                action="ACCEPTED",
                embedding_distance=0.0,
                confidence=1.0,
            )
            self._stats["accepted"] += 1
        else:
            dist = SkeletonEncoder.cosine_distance(curr_embedding, self._prev_embedding)
            confidence = max(0.0, 1.0 - dist / (self.blend_threshold + 1e-8))

            if dist <= self.accept_threshold:
                # Smooth motion: accept as-is
                result = SmoothedSkeleton(
                    frame_idx=self._frame_count,
                    center=center, tip1=tip1, tip2=tip2,
                    raw_center=center, raw_tip1=tip1, raw_tip2=tip2,
                    action="ACCEPTED",
                    embedding_distance=dist,
                    confidence=min(1.0, confidence),
                )
                self._stats["accepted"] += 1

            elif dist <= self.blend_threshold:
                # Moderate change: interpolate with previous
                prev = self._history[-1]
                alpha = self.blend_alpha
                s_center = self._lerp(prev.center, center, alpha)
                s_tip1 = self._lerp(prev.tip1, tip1, alpha)
                s_tip2 = self._lerp(prev.tip2, tip2, alpha)
                result = SmoothedSkeleton(
                    frame_idx=self._frame_count,
                    center=s_center, tip1=s_tip1, tip2=s_tip2,
                    raw_center=center, raw_tip1=tip1, raw_tip2=tip2,
                    action="INTERPOLATED",
                    embedding_distance=dist,
                    confidence=confidence,
                )
                self._stats["interpolated"] += 1

            else:
                # Too different: use previous smoothed skeleton
                prev = self._history[-1]
                result = SmoothedSkeleton(
                    frame_idx=self._frame_count,
                    center=prev.center, tip1=prev.tip1, tip2=prev.tip2,
                    raw_center=center, raw_tip1=tip1, raw_tip2=tip2,
                    action="REJECTED_PREV_USED",
                    embedding_distance=dist,
                    confidence=0.0,
                )
                self._stats["rejected"] += 1

        # Update state
        self._prev_embedding = self.encoder.encode(
            result.center, result.tip1, result.tip2, original_size
        )
        self._history.append(result)
        self._frame_count += 1
        return result

    def process_sequence(
        self,
        skeletons: List[Dict],
        original_size: int = 500,
    ) -> List[Dict]:
        """
        Batch-process a list of raw skeleton dicts.

        Each dict: {"center": [x,y], "tip1": [x,y], "tip2": [x,y]}

        Returns list of SmoothedSkeleton dicts.
        """
        self.reset()
        results = []
        for skel in skeletons:
            smoothed = self.add_frame(
                center=skel["center"],
                tip1=skel["tip1"],
                tip2=skel["tip2"],
                original_size=original_size,
            )
            results.append(smoothed.to_dict())
        return results

    def get_smoothing_stats(self) -> Dict:
        """Return summary statistics about smoothing decisions."""
        total = sum(self._stats.values()) or 1
        return {
            "total_frames": self._frame_count,
            "accepted": self._stats["accepted"],
            "interpolated": self._stats["interpolated"],
            "rejected": self._stats["rejected"],
            "acceptance_rate": round(self._stats["accepted"] / total, 3),
            "rejection_rate": round(self._stats["rejected"] / total, 3),
        }

    def reset(self):
        """Reset smoother state for a new video sequence."""
        self._history.clear()
        self._prev_embedding = None
        self._frame_count = 0
        self._stats = {"accepted": 0, "interpolated": 0, "rejected": 0}

    @staticmethod
    def _lerp(a: List[float], b: List[float], alpha: float) -> List[float]:
        """Linear interpolation: (1-α)*a + α*b."""
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        return ((1 - alpha) * a_arr + alpha * b_arr).tolist()
