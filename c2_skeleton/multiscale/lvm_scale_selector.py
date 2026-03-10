"""
LVMScaleSelector
=================
Selects the optimal spatial scale σ* for skeleton detection by acting
as an LVM (Large Vision Model) oracle.

Core Idea (from GAP 3):
  The "right" scale is the one where the detected skeleton structure
  best matches the visual content of the original image.

  LVM similarity score: how well does render(G_σ) look like the input?

  σ* = argmax_σ sim(encode(original_image), encode(render(G_σ)))

LVM Proxy Implementation:
  Instead of CLIP at runtime, we compute:
    1. HOG (Histogram of Oriented Gradients) descriptor — captures shape
    2. Edge density overlap — how much skeleton aligns with image edges
    3. Structural SSIM — pixel-level structural similarity
  Combined as a weighted similarity score.

  Interface is swap-in compatible with CLIP:
    Replace encode() with clip.encode_image() to use real LVM.

Outputs:
  - best_sigma: float — the selected optimal scale
  - scale_scores: dict {sigma: score} — LVM score at each scale
  - confidence: float — spread between best and second-best score
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .scale_space import ScaleGraph


class LVMScaleSelector:
    """
    Selects optimal scale σ* using structural similarity as LVM proxy.

    Parameters
    ----------
    embed_size : int
        Size of the patch used for embedding (default 64×64).
    hog_weight : float
        Weight of HOG cosine similarity in the combined score.
    edge_weight : float
        Weight of edge overlap score.
    ssim_weight : float
        Weight of structural SSIM score.
    """

    def __init__(
        self,
        embed_size: int = 64,
        hog_weight: float = 0.5,
        edge_weight: float = 0.3,
        ssim_weight: float = 0.2,
    ):
        self.embed_size = embed_size
        self.hog_weight = hog_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight

        # HOG descriptor parameters
        self._hog = cv2.HOGDescriptor(
            _winSize=(embed_size, embed_size),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )

    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Compute LVM-proxy embedding of an image.

        Parameters
        ----------
        image : np.ndarray, shape (H, W, 3) or (H, W), uint8

        Returns
        -------
        embedding : np.ndarray, shape (D,), float32, L2-normalized
        """
        resized = cv2.resize(image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                             (self.embed_size, self.embed_size))
        hog_feat = self._hog.compute(resized)
        if hog_feat is None:
            hog_feat = np.zeros((1,), dtype=np.float32)
        embedding = hog_feat.flatten().astype(np.float32)
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized embeddings."""
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    def edge_overlap_score(
        self, original: np.ndarray, rendered: np.ndarray
    ) -> float:
        """
        Fraction of rendered skeleton pixels that coincide with original image edges.
        High overlap = skeleton aligns with real structure.
        """
        gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if original.ndim == 3 else original
        gray_rend = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY) if rendered.ndim == 3 else rendered

        edges_orig = cv2.Canny(gray_orig, 30, 100).astype(bool)
        edges_rend = (gray_rend > 20).astype(bool)

        if not edges_rend.any():
            return 0.0

        intersection = (edges_orig & edges_rend).sum()
        return float(intersection) / float(edges_rend.sum() + 1e-6)

    def ssim_score(self, original: np.ndarray, rendered: np.ndarray) -> float:
        """Structural SSIM between original and rendered skeleton patch."""
        size = (self.embed_size, self.embed_size)
        a = cv2.resize(
            cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if original.ndim == 3 else original,
            size
        ).astype(np.float32) / 255.0
        b = cv2.resize(
            cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY) if rendered.ndim == 3 else rendered,
            size
        ).astype(np.float32) / 255.0

        mu_a, mu_b = a.mean(), b.mean()
        sig_a = a.std() + 1e-8
        sig_b = b.std() + 1e-8
        sig_ab = float(np.mean((a - mu_a) * (b - mu_b)))

        c1, c2 = (0.01 ** 2), (0.03 ** 2)
        ssim = ((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)) / \
               ((mu_a**2 + mu_b**2 + c1) * (sig_a**2 + sig_b**2 + c2))
        return float(np.clip(ssim, 0.0, 1.0))

    def score_scale(
        self,
        original_image: np.ndarray,
        rendered_graph: np.ndarray,
        original_embedding: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute combined LVM-proxy similarity score for one scale.

        Parameters
        ----------
        original_image  : raw input image
        rendered_graph  : graph overlay rendered at this scale
        original_embedding : pre-computed embedding of original (optional, for speed)

        Returns
        -------
        Combined score in [0, 1]
        """
        if original_embedding is None:
            original_embedding = self.encode(original_image)

        rendered_embedding = self.encode(rendered_graph)
        hog_sim = self.cosine_similarity(original_embedding, rendered_embedding)
        hog_sim = (hog_sim + 1) / 2.0   # shift [-1,1] → [0,1]

        edge_sim = self.edge_overlap_score(original_image, rendered_graph)
        ssim = self.ssim_score(original_image, rendered_graph)

        combined = (self.hog_weight * hog_sim
                    + self.edge_weight * edge_sim
                    + self.ssim_weight * ssim)
        return float(np.clip(combined, 0.0, 1.0))

    def select_best_scale(
        self,
        original_image: np.ndarray,
        scale_graphs: Dict,    # {sigma: ScaleGraph}
        scale_space,           # GaussianScaleSpace instance
    ) -> dict:
        """
        Score all scale graphs and return the best σ*.

        Returns
        -------
        dict with:
          best_sigma, scale_scores, confidence, best_graph
        """
        orig_embed = self.encode(original_image)
        scale_scores: Dict[float, float] = {}

        for sigma, graph in scale_graphs.items():
            rendered = scale_space.render_graph_on_image(original_image, graph)
            score = self.score_scale(original_image, rendered, orig_embed)
            scale_scores[sigma] = round(score, 4)

        if not scale_scores:
            return {"best_sigma": 1.0, "scale_scores": {}, "confidence": 0.0, "best_graph": None}

        sorted_scores = sorted(scale_scores.items(), key=lambda x: x[1], reverse=True)
        best_sigma = sorted_scores[0][0]
        best_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        confidence = round(float(best_score - second_score), 4)

        return {
            "best_sigma": best_sigma,
            "scale_scores": scale_scores,
            "confidence": confidence,
            "best_graph": scale_graphs[best_sigma].to_dict(),
        }
