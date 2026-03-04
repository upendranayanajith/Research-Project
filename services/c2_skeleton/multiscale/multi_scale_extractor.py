"""
MultiScaleSkeletonExtractor
============================
Orchestrates the full GAP 3 multi-scale extraction pipeline:

  1. Build Gaussian scale pyramid at σ = {1, 2, 4, 8, 16}
  2. Detect keypoints and build proximity graph G_σ at each scale
  3. Score each G_σ using LVM-proxy similarity (LVMScaleSelector)
  4. Select σ* = argmax similarity score
  5. Return optimal graph + all scale results + confidence

The key insight:
  - At σ=1 (fine scale): detects pixel-level noise + fine texture edges
  - At σ=4-8 (hand scale): detects clock hand shaft structure
  - At σ=16 (coarse scale): detects whole clock as single blob
  - σ* is where semantic content (hand connectivity) is clearest

This module integrates GaussianScaleSpace and LVMScaleSelector
into a single .extract() call for use by the FastAPI endpoint.
"""

import numpy as np
import cv2
import base64
from typing import Dict, List, Optional
from .scale_space import GaussianScaleSpace, ScaleGraph
from .lvm_scale_selector import LVMScaleSelector


class MultiScaleSkeletonExtractor:
    """
    Full GAP 3 pipeline: multi-scale graph extraction with LVM scale oracle.

    Parameters
    ----------
    scales : list of float
        Sigma values for Gaussian pyramid. Default [1, 2, 4, 8, 16].
    max_keypoints : int
        Max keypoints per scale (for speed).
    embed_size : int
        LVMScaleSelector patch size.
    """

    def __init__(
        self,
        scales: List[float] = None,
        max_keypoints: int = 15,
        embed_size: int = 64,
    ):
        self.scales = scales or [1.0, 2.0, 4.0, 8.0, 16.0]
        self.scale_space = GaussianScaleSpace(
            scales=self.scales,
            max_keypoints_per_scale=max_keypoints,
        )
        self.selector = LVMScaleSelector(embed_size=embed_size)

    def extract(self, image: np.ndarray) -> Dict:
        """
        Run multi-scale extraction on an input image.

        Parameters
        ----------
        image : np.ndarray, shape (H, W, 3) or (H, W)

        Returns
        -------
        dict with:
          best_sigma      — selected optimal scale σ*
          scale_scores    — LVM score at each σ
          confidence      — margin between top-2 scores
          best_graph      — ScaleGraph dict at σ*
          all_graphs      — ScaleGraph dicts at all σ
          interpretation  — text description
          visualization   — base64 JPEG of best-scale overlay
        """
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Step 1-2: Build all scale graphs
        scale_graphs: Dict[float, ScaleGraph] = self.scale_space.extract_all_scales(image)

        # Step 3-4: LVM selection
        selection = self.selector.select_best_scale(image, scale_graphs, self.scale_space)

        best_sigma = selection["best_sigma"]
        best_graph = scale_graphs[best_sigma]

        # Step 5: Render best-scale graph overlay
        overlay = self.scale_space.render_graph_on_image(image, best_graph)
        viz = cv2.addWeighted(image, 0.6, overlay, 0.8, 0)
        _, buf = cv2.imencode('.jpg', cv2.resize(viz, (500, 500)))
        viz_b64 = base64.b64encode(buf).decode('utf-8')

        # Step 6: Interpretation
        interpretation = self._interpret(best_sigma, selection["confidence"], best_graph)

        return {
            "best_sigma": best_sigma,
            "scale_scores": selection["scale_scores"],
            "confidence": selection["confidence"],
            "best_graph": selection["best_graph"],
            "all_graphs": {str(s): g.to_dict() for s, g in scale_graphs.items()},
            "interpretation": interpretation,
            "visualization": viz_b64,
        }

    def extract_with_yolo_keypoints(
        self,
        image: np.ndarray,
        yolo_center: List[float],
        yolo_tip1: List[float],
        yolo_tip2: List[float],
    ) -> Dict:
        """
        Run multi-scale extraction AND integrate YOLO keypoints.

        At the selected scale σ*, check if YOLO keypoints align with
        the detected graph nodes. If not → YOLO may be detecting at
        the wrong scale.

        Parameters
        ----------
        image        : input image
        yolo_center, yolo_tip1, yolo_tip2 : YOLO-detected keypoints [x, y]

        Returns
        -------
        dict from extract() + additional:
          yolo_scale_alignment — which sigma best matches YOLO output
          scale_mismatch       — True if YOLO and LVM disagree on scale
        """
        result = self.extract(image)
        scale_graphs = {float(k): None for k in result["all_graphs"]}

        # Re-extract (or use stored) graphs for YOLO alignment
        graphs_full = self.scale_space.extract_all_scales(image)

        yolo_pts = [yolo_center, yolo_tip1, yolo_tip2]
        alignment_scores: Dict[str, float] = {}

        for sigma, graph in graphs_full.items():
            if not graph.keypoints:
                alignment_scores[str(sigma)] = 0.0
                continue
            # Average min-distance from each YOLO keypoint to any graph node
            kpt_arr = np.array(graph.keypoints)
            min_dists = []
            for pt in yolo_pts:
                dists = np.linalg.norm(kpt_arr - np.array(pt), axis=1)
                min_dists.append(float(dists.min()))
            alignment_scores[str(sigma)] = round(1.0 / (np.mean(min_dists) + 1.0), 4)

        best_yolo_sigma = max(alignment_scores, key=alignment_scores.get)
        scale_mismatch = (str(result["best_sigma"]) != best_yolo_sigma)

        result["yolo_scale_alignment"] = {
            "per_sigma": alignment_scores,
            "best_yolo_sigma": float(best_yolo_sigma),
            "lvm_selected_sigma": result["best_sigma"],
            "scale_mismatch": scale_mismatch,
        }
        return result

    @staticmethod
    def _interpret(sigma: float, confidence: float, graph: ScaleGraph) -> str:
        scale_label = {
            1.0: "pixel-level (fine texture)",
            2.0: "fine structure",
            4.0: "hand shaft scale",
            8.0: "full hand scale",
            16.0: "whole clock scale",
        }.get(sigma, f"σ={sigma}")

        n_kpts = len(graph.keypoints)
        n_edges = len(graph.edges)

        return (
            f"Optimal scale σ*={sigma} ({scale_label}). "
            f"LVM confidence margin: {confidence:.3f}. "
            f"Graph: {n_kpts} keypoints, {n_edges} edges. "
            f"{'High confidence selection.' if confidence > 0.1 else 'Low confidence — scales similar in quality.'}"
        )
