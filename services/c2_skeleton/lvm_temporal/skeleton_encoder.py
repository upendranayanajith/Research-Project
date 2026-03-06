"""
SkeletonEncoder
================
Computes a structural embedding vector for a clock skeleton,
acting as the LVM proxy for temporal consistency validation.

Role in the LVM Temporal Smoothing pipeline:
  encode(skeleton_t) → embedding vector e_t
  cosine_distance(e_t, e_{t-1}) → similarity score
  if distance > threshold → jitter → interpolate

LVM Proxy Strategy:
  1. Render skeleton to a 64×64 patch (center, tip1, tip2 as lines + dots)
  2. Convert to grayscale
  3. Compute HOG descriptor — captures edge orientations (shape-aware)
  4. L2-normalize → unit embedding vector

Why HOG?
  - Rotation-sensitive: different clock-hand configurations → different HOG
  - Scale-invariant within the patch: hand length doesn't dominate
  - Fast (< 1 ms on CPU)
  - Captures gross skeleton changes (detector failure) vs fine motion

Swap-in for real CLIP:
  Replace encode() with:
      img_tensor = preprocess(render_skeleton(...)).unsqueeze(0)
      return clip_model.encode_image(img_tensor).detach().numpy()
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple


class SkeletonEncoder:
    """
    Renders and encodes a clock skeleton as an LVM-proxy embedding.

    Parameters
    ----------
    patch_size : int
        Size (H=W) of the rendered patch used for embedding (default 64).
    embed_dim : int or None
        If set, truncate or pad HOG features to this dimension.
    """

    def __init__(self, patch_size: int = 64, embed_dim: Optional[int] = None):
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self._hog = cv2.HOGDescriptor(
            _winSize=(patch_size, patch_size),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )

    def render_skeleton(
        self,
        center: List[float],
        tip1: List[float],
        tip2: List[float],
        original_size: int = 500,
    ) -> np.ndarray:
        """
        Render the skeleton as a binary image patch of size patch_size×patch_size.

        Lines: center→tip1 and center→tip2.
        Circles at each keypoint.

        Returns
        -------
        np.ndarray, shape (patch_size, patch_size), uint8
        """
        canvas = np.zeros((original_size, original_size), dtype=np.uint8)
        cx, cy = int(center[0]), int(center[1])
        t1x, t1y = int(tip1[0]), int(tip1[1])
        t2x, t2y = int(tip2[0]), int(tip2[1])

        cv2.line(canvas, (cx, cy), (t1x, t1y), 200, 6)
        cv2.line(canvas, (cx, cy), (t2x, t2y), 200, 6)
        cv2.circle(canvas, (cx, cy), 10, 255, -1)
        cv2.circle(canvas, (t1x, t1y), 8, 180, -1)
        cv2.circle(canvas, (t2x, t2y), 8, 180, -1)

        # Resize to patch_size
        patch = cv2.resize(canvas, (self.patch_size, self.patch_size))
        return patch

    def encode(
        self,
        center: List[float],
        tip1: List[float],
        tip2: List[float],
        original_size: int = 500,
    ) -> np.ndarray:
        """
        Render skeleton and compute its HOG embedding.

        Parameters
        ----------
        center, tip1, tip2 : [x, y] keypoint positions
        original_size : size of the coordinate space (to scale render)

        Returns
        -------
        embedding : np.ndarray, shape (D,), float32, L2-normalized
        """
        patch = self.render_skeleton(center, tip1, tip2, original_size)
        hog_feat = self._hog.compute(patch)

        if hog_feat is None:
            hog_feat = np.zeros((324,), dtype=np.float32)   # default HOG dim

        embedding = hog_feat.flatten().astype(np.float32)

        if self.embed_dim is not None:
            if len(embedding) >= self.embed_dim:
                embedding = embedding[:self.embed_dim]
            else:
                embedding = np.pad(embedding, (0, self.embed_dim - len(embedding)))

        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance in [0, 2] (0 = identical, 2 = opposite)."""
        similarity = float(np.clip(np.dot(a, b), -1.0, 1.0))
        return 1.0 - similarity

    def encode_raw(self, patch: np.ndarray) -> np.ndarray:
        """Encode a pre-rendered (patch_size × patch_size) uint8 patch directly."""
        hog_feat = self._hog.compute(patch)
        if hog_feat is None:
            return np.zeros((324,), dtype=np.float32)
        embedding = hog_feat.flatten().astype(np.float32)
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm
