"""
LVM Temporal Smoothing
========================
"LVM-Guided Temporal Consistency for Skeleton Detection"

Core Idea:
  Raw keypoint detections are JITTERY frame-to-frame due to:
    - Detector noise
    - Slight image variation
    - Occlusion artifacts

  Instead of Kalman filtering (position only) or Optical flow (pixels),
  we validate each new skeleton against the previous one using an
  LVM-style EMBEDDING DISTANCE.

  If cosine_distance(embed(skeleton_t), embed(skeleton_{t-1})) > threshold:
    → Too different → likely detection error
    → Interpolate between previous and current

LVM Proxy:
  Instead of calling CLIP at runtime, we compute a structural embedding:
    - Render skeleton to small (64×64) image
    - Extract HOG features (shape-describing descriptor)
    - L2-normalize → embedding vector
  Interface is identical to a real LVM — swap in CLIP by replacing encode().

Modules:
  skeleton_encoder  — SkeletonEncoder: render + HOG embedding
  lvm_smoother      — LVMTemporalSmoother: distance gating + interpolation
"""

from .skeleton_encoder import SkeletonEncoder
from .lvm_smoother import LVMTemporalSmoother, SmoothedSkeleton

__all__ = [
    "SkeletonEncoder",
    "LVMTemporalSmoother",
    "SmoothedSkeleton",
]
