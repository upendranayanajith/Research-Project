"""
LVMMultiScaleDetector — Recommended Combination Pipeline
==========================================================
"LVM-Guided Multi-Scale Skeleton Detection with Uncertainty Quantification"

This is the publication-ready pipeline combining:

  Step 1 [GAP 3] MultiScaleSkeletonExtractor
    → Select optimal scale σ* using LVM-proxy oracle
    → Returns scale-optimal keypoints + confidence

  Step 2 [GAP 1] BayesianGraphInference
    → Probabilistic 3D reconstruction of best keypoints
    → Returns MAP structure + posterior uncertainty

  Step 3 [LVM Temporal] LVMTemporalSmoother
    → Validate skeleton against previous frame embedding
    → Returns ACCEPTED / INTERPOLATED / REJECTED decision

  Output: Combined result with all layers of analysis.

The Research Story:
  "Optimal detection scale varies by clock design (ornate vs minimal).
   We use LVM embeddings as a learned scale oracle.
   Combined with Bayesian uncertainty quantification,
   we produce spatially and temporally consistent 3D skeletons."

Novel claims:
  ✅ First LVM-as-scale-oracle for skeleton detection
  ✅ Scale + depth + temporal uncertainty all quantified
  ✅ No re-training required — works with existing YOLO-Pose
"""

import numpy as np
import cv2
import base64
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from ..multiscale import MultiScaleSkeletonExtractor
from ..lvm_temporal import LVMTemporalSmoother, SmoothedSkeleton

# Lazy import to avoid circular dependency issues
def _get_bayesian_engine():
    from ..probabilistic_3d import BayesianGraphInference
    return BayesianGraphInference(k_hypotheses=10, image_size=500)


@dataclass
class CombinedResult:
    """
    Full output of the LVMMultiScaleDetector pipeline.

    Attributes
    ----------
    frame_idx           : int
    keypoints_2d        : original YOLO 2D keypoints
    scale_analysis      : multi-scale LVM scoring results
    reconstruction_3d   : Bayesian 3D inference result
    temporal_smoothing  : LVM-gated smoothing decision
    final_skeleton      : the FINAL recommended skeleton (smoothed if needed)
    pipeline_confidence : geometric mean of all sub-module confidences
    summary             : human-readable pipeline narrative
    """
    frame_idx: int
    keypoints_2d: Dict
    scale_analysis: Dict
    reconstruction_3d: Dict
    temporal_smoothing: Dict
    final_skeleton: Dict
    pipeline_confidence: float
    summary: str

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "keypoints_2d": self.keypoints_2d,
            "scale_analysis": {
                "best_sigma": self.scale_analysis.get("best_sigma"),
                "confidence": self.scale_analysis.get("confidence"),
                "lvm_scores": self.scale_analysis.get("scale_scores"),
                "interpretation": self.scale_analysis.get("interpretation"),
            },
            "reconstruction_3d": {
                "hand_assignment": self.reconstruction_3d.get("hand_assignment"),
                "hand_depths": self.reconstruction_3d.get("hand_depths"),
                "uncertainty": self.reconstruction_3d.get("uncertainty"),
                "occlusion_risk": self.reconstruction_3d.get("occlusion_risk"),
            },
            "temporal_smoothing": self.temporal_smoothing,
            "final_skeleton": self.final_skeleton,
            "pipeline_confidence": round(self.pipeline_confidence, 3),
            "summary": self.summary,
        }


class LVMMultiScaleDetector:
    """
    Publication-ready combined pipeline detector.

    Parameters
    ----------
    scales : list of float
        Sigma levels for multi-scale extraction.
    smooth_accept_threshold : float
        LVM cosine distance threshold for direct acceptance.
    smooth_blend_threshold  : float
        LVM cosine distance threshold below which interpolation applies.
    image_size : int
        Expected image side length.
    """

    def __init__(
        self,
        scales: List[float] = None,
        smooth_accept_threshold: float = 0.15,
        smooth_blend_threshold: float = 0.40,
        image_size: int = 500,
    ):
        self.image_size = image_size
        self.scale_extractor = MultiScaleSkeletonExtractor(scales=scales)
        self.smoother = LVMTemporalSmoother(
            accept_threshold=smooth_accept_threshold,
            blend_threshold=smooth_blend_threshold,
        )
        self._bayesian = None   # lazy init (avoids import overhead)
        self._frame_count = 0

    def process(
        self,
        image: np.ndarray,
        center: List[float],
        tip1: List[float],
        tip2: List[float],
    ) -> CombinedResult:
        """
        Run the full combined pipeline for one frame.

        Parameters
        ----------
        image  : np.ndarray — original clock image (H, W, 3)
        center : [x, y] — YOLO keypoint
        tip1   : [x, y] — YOLO keypoint
        tip2   : [x, y] — YOLO keypoint

        Returns
        -------
        CombinedResult
        """
        if self._bayesian is None:
            self._bayesian = _get_bayesian_engine()

        # ── Step 1: Multi-scale LVM analysis ─────────────────────────
        scale_result = self.scale_extractor.extract_with_yolo_keypoints(
            image, center, tip1, tip2
        )
        scale_confidence = float(scale_result.get("confidence", 0.0))

        # ── Step 2: Bayesian 3D reconstruction ────────────────────────
        bayes_result = self._bayesian.infer(center, tip1, tip2)
        bayes_confidence = float(
            bayes_result.get("uncertainty", {}).get("confidence_score", 0.5)
        )

        # ── Step 3: LVM temporal smoothing ───────────────────────────
        smoothed: SmoothedSkeleton = self.smoother.add_frame(
            center=center, tip1=tip1, tip2=tip2,
            original_size=self.image_size,
        )
        smooth_confidence = smoothed.confidence

        # ── Step 4: Combine confidences ───────────────────────────────
        # Geometric mean of the three sub-module confidence values
        pipeline_confidence = float(
            (scale_confidence * bayes_confidence * smooth_confidence + 1e-8) ** (1/3)
        )

        # ── Step 5: Final skeleton (use smoothed positions) ──────────
        final_skeleton = {
            "center": smoothed.center,
            "tip1":   smoothed.tip1,
            "tip2":   smoothed.tip2,
            "angles": {
                "hand1": round(self._angle(smoothed.center, smoothed.tip1), 2),
                "hand2": round(self._angle(smoothed.center, smoothed.tip2), 2),
            },
        }

        # ── Step 6: Build summary narrative ──────────────────────────
        summary = self._build_summary(
            scale_result, bayes_result, smoothed, pipeline_confidence
        )

        result = CombinedResult(
            frame_idx=self._frame_count,
            keypoints_2d={"center": center, "tip1": tip1, "tip2": tip2},
            scale_analysis=scale_result,
            reconstruction_3d=bayes_result,
            temporal_smoothing=smoothed.to_dict(),
            final_skeleton=final_skeleton,
            pipeline_confidence=pipeline_confidence,
            summary=summary,
        )

        self._frame_count += 1
        return result

    def get_session_stats(self) -> Dict:
        """Return combined pipeline statistics."""
        return {
            "frames_processed": self._frame_count,
            "smoothing_stats": self.smoother.get_smoothing_stats(),
        }

    def reset(self):
        """Reset temporal state for a new video sequence."""
        self.smoother.reset()
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _angle(center: List[float], tip: List[float]) -> float:
        """Clockwise angle from 12-o'clock."""
        dx = tip[0] - center[0]
        dy = tip[1] - center[1]
        angle = math.degrees(math.atan2(dx, -dy))
        return (angle + 360) % 360

    @staticmethod
    def _build_summary(scale_res, bayes_res, smoothed, conf) -> str:
        sigma = scale_res.get("best_sigma", "?")
        occ_risk = bayes_res.get("occlusion_risk", "?")
        action = smoothed.action
        return (
            f"[Combined Pipeline] "
            f"Scale: σ*={sigma} (LVM-selected). "
            f"3D: occlusion_risk={occ_risk}, "
            f"confidence={bayes_res.get('uncertainty', {}).get('confidence_score', 0):.2f}. "
            f"Temporal: {action} (dist={smoothed.embedding_distance:.3f}). "
            f"Pipeline confidence: {conf:.3f}."
        )
