"""
ResearchVisualizer
===================
Generates publication-quality base64 images for the C2 research outputs,
displayed inside the main pipeline's Structure tab.

All rendering uses OpenCV + NumPy only (no matplotlib/plotly dependency).
Each method returns a base64-encoded JPEG string.
"""

import numpy as np
import cv2
import base64
import math
from typing import Dict, List, Optional, Tuple


class ResearchVisualizer:
    """Renders visual chart images for C2 enhanced analysis."""

    # ── Color palette ────────────────────────────────────────────────────
    BG         = (30,  30,  35)
    WHITE      = (255, 255, 255)
    GRAY       = (140, 140, 145)
    GREEN      = (100, 220, 120)
    BLUE       = (220, 160,  60)
    RED        = (80,  80,  240)
    GOLD       = (50,  200, 255)
    CYAN       = (230, 200, 80)
    PANEL_BG   = (42,  42,  48)
    BAR_BG     = (60,  60,  65)

    @classmethod
    def _encode(cls, img: np.ndarray) -> str:
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return base64.b64encode(buf).decode('utf-8')

    # ─────────────────────────────────────────────────────────────────────
    # 1.  Scale Pyramid Grid
    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def render_scale_pyramid(
        cls,
        original: np.ndarray,
        scale_scores: Dict[str, float],
        best_sigma: float,
    ) -> str:
        """
        Renders a horizontal strip of blurred images at each scale σ,
        with an LVM score bar below each.
        """
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if original.ndim == 3 else original
        H_thumb, W_thumb = 120, 120
        sigmas = sorted([float(s) for s in scale_scores.keys()])
        n = len(sigmas)
        if n == 0:
            return cls._encode(np.zeros((100, 300, 3), dtype=np.uint8))

        pad = 8
        bar_h = 40
        label_h = 22
        total_w = n * (W_thumb + pad) + pad
        total_h = label_h + H_thumb + bar_h + pad * 3
        canvas = np.full((total_h, total_w, 3), cls.BG, dtype=np.uint8)

        for i, sigma in enumerate(sigmas):
            x0 = pad + i * (W_thumb + pad)
            y0 = label_h + pad

            # Blurred thumbnail
            ksize = int(6 * sigma + 1) | 1
            blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)
            thumb = cv2.resize(blurred, (W_thumb, H_thumb))
            thumb_color = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

            # Highlight best sigma
            is_best = abs(sigma - best_sigma) < 0.01
            if is_best:
                cv2.rectangle(thumb_color, (0, 0), (W_thumb-1, H_thumb-1), cls.GOLD, 3)

            canvas[y0:y0+H_thumb, x0:x0+W_thumb] = thumb_color

            # σ label
            label = f"σ={sigma:.0f}" if sigma == int(sigma) else f"σ={sigma}"
            color = cls.GOLD if is_best else cls.WHITE
            cv2.putText(canvas, label, (x0 + 10, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Score bar
            score = scale_scores.get(str(sigma), scale_scores.get(sigma, 0.0))
            bar_y = y0 + H_thumb + pad
            bar_w = int(W_thumb * min(score, 1.0))
            cv2.rectangle(canvas, (x0, bar_y), (x0 + W_thumb, bar_y + 18), cls.BAR_BG, -1)
            bar_color = cls.GOLD if is_best else cls.CYAN
            cv2.rectangle(canvas, (x0, bar_y), (x0 + bar_w, bar_y + 18), bar_color, -1)
            cv2.putText(canvas, f"{score:.2f}", (x0 + 4, bar_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, cls.BG, 1)

        return cls._encode(canvas)

    # ─────────────────────────────────────────────────────────────────────
    # 2.  Confidence Gauge
    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def render_confidence_gauge(
        cls,
        confidence: float,
        occlusion_risk: str = "LOW",
        hand_assignment: Optional[Dict] = None,
    ) -> str:
        """
        Renders an arc gauge for confidence [0–1] plus occlusion risk badge.
        """
        W, H = 320, 200
        canvas = np.full((H, W, 3), cls.BG, dtype=np.uint8)
        cx, cy = W // 2, 150
        r = 90

        # Background arc (180° → 0°, bottom half hidden)
        cv2.ellipse(canvas, (cx, cy), (r, r), 0, 180, 360, cls.BAR_BG, 12)

        # Confidence arc
        end_angle = 180 + int(180 * min(confidence, 1.0))
        if confidence > 0.7:
            color = cls.GREEN
        elif confidence > 0.4:
            color = cls.GOLD
        else:
            color = cls.RED
        cv2.ellipse(canvas, (cx, cy), (r, r), 0, 180, end_angle, color, 12)

        # Needle
        needle_angle = math.radians(180 + 180 * min(confidence, 1.0))
        nx = int(cx + (r - 20) * math.cos(needle_angle))
        ny = int(cy + (r - 20) * math.sin(needle_angle))
        cv2.line(canvas, (cx, cy), (nx, ny), cls.WHITE, 2)
        cv2.circle(canvas, (cx, cy), 6, cls.WHITE, -1)

        # Value text
        cv2.putText(canvas, f"{confidence:.2f}", (cx - 30, cy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cls.WHITE, 2)
        cv2.putText(canvas, "Confidence", (cx - 42, cy - r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cls.GRAY, 1)

        # Occlusion risk badge
        risk_colors = {"LOW": cls.GREEN, "MEDIUM": cls.GOLD, "HIGH": cls.RED}
        badge_color = risk_colors.get(occlusion_risk, cls.GRAY)
        cv2.rectangle(canvas, (10, 10), (150, 35), badge_color, -1)
        cv2.putText(canvas, f"Occlusion: {occlusion_risk}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls.BG, 1)

        # Hand assignment
        if hand_assignment:
            y_ha = 50
            for label, val in hand_assignment.items():
                cv2.putText(canvas, f"{label}: {val}", (15, y_ha), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls.WHITE, 1)
                y_ha += 18

        return cls._encode(canvas)

    # ─────────────────────────────────────────────────────────────────────
    # 3.  Curvature Heatmap Overlay
    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def render_curvature_heatmap(
        cls,
        original: np.ndarray,
        curvature_ratios: Dict,
        surface_class: str,
    ) -> str:
        """
        Overlays curvature information on the clock image with
        geodesic vs euclidean annotations per pair.
        """
        H_out, W_out = 300, 400
        canvas = np.full((H_out, W_out, 3), cls.BG, dtype=np.uint8)

        # Resize original into left half
        thumb = cv2.resize(original, (180, 180))
        canvas[10:190, 10:190] = thumb

        # Surface classification badge
        sc_colors = {"FLAT": cls.GREEN, "MILDLY_CURVED": cls.GOLD, "HIGHLY_CURVED": cls.RED}
        badge_c = sc_colors.get(surface_class, cls.GRAY)
        cv2.rectangle(canvas, (10, 200), (190, 225), badge_c, -1)
        cv2.putText(canvas, surface_class, (20, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cls.BG, 1)

        # Curvature ratio table on right side
        x_table = 205
        y = 20
        cv2.putText(canvas, "Curvature Analysis", (x_table, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cls.WHITE, 1)
        y += 8
        cv2.line(canvas, (x_table, y), (W_out - 10, y), cls.GRAY, 1)
        y += 20

        for pair_name, info in curvature_ratios.items():
            geo = info.get("geodesic_px", 0)
            euc = info.get("euclidean_px", 0)
            ratio = info.get("ratio", 1.0)
            curved = info.get("surface_curved", False)

            # Pair label
            label = pair_name.replace("↔", "⇔")
            cv2.putText(canvas, label, (x_table, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, cls.CYAN, 1)
            y += 18

            # Euclidean bar
            cv2.putText(canvas, f"Euclid: {euc:.0f}px", (x_table, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, cls.GRAY, 1)
            y += 16
            cv2.putText(canvas, f"Geodesic: {geo:.0f}px", (x_table, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, cls.GRAY, 1)
            y += 16

            # Ratio indicator
            r_color = cls.RED if curved else cls.GREEN
            cv2.putText(canvas, f"Ratio: {ratio:.3f}", (x_table, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, r_color, 1)
            y += 25

        return cls._encode(canvas)

    # ─────────────────────────────────────────────────────────────────────
    # 4.  Before / After Comparison Panel
    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def render_comparison(
        cls,
        original: np.ndarray,
        center: List[float],
        tip1: List[float],
        tip2: List[float],
        confidence: float,
        occlusion_risk: str,
        best_sigma: float,
    ) -> str:
        """
        Side-by-side: Left = basic YOLO skeleton, Right = enhanced with
        confidence-colored lines and metadata overlay.
        """
        size = 220
        pad = 10
        W = size * 2 + pad * 3
        H = size + 80
        canvas = np.full((H, W, 3), cls.BG, dtype=np.uint8)

        # Detect gauge mode: tip1 == tip2 (from _unpack_for_3_point shim)
        is_gauge = (
            abs(tip1[0] - tip2[0]) < 1.0 and abs(tip1[1] - tip2[1]) < 1.0
        )

        def draw_skel(img, color_h1, color_h2, thickness):
            out = cv2.resize(img.copy(), (size, size))
            sc = size / max(img.shape[:2])
            c = (int(center[0]*sc), int(center[1]*sc))
            t1 = (int(tip1[0]*sc), int(tip1[1]*sc))
            cv2.line(out, c, t1, color_h1, thickness)
            cv2.circle(out, c, 6, cls.WHITE, -1)
            cv2.circle(out, t1, 5, color_h1, -1)
            if not is_gauge:
                t2 = (int(tip2[0]*sc), int(tip2[1]*sc))
                cv2.line(out, c, t2, color_h2, thickness)
                cv2.circle(out, t2, 5, color_h2, -1)
            return out

        # Left: basic YOLO (plain green lines)
        left = draw_skel(original, (0, 200, 0), (0, 0, 200), 2)
        canvas[40:40+size, pad:pad+size] = left
        cv2.putText(canvas, "Basic YOLO", (pad + 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cls.GRAY, 1)

        # Right: enhanced (confidence-colored, thicker)
        if confidence > 0.7:
            c1, c2 = cls.GREEN, (120, 255, 120)
        elif confidence > 0.4:
            c1, c2 = cls.GOLD, (80, 220, 255)
        else:
            c1, c2 = cls.RED, (100, 100, 255)
        right = draw_skel(original, c1, c2, 4)

        # Overlay confidence badge on enhanced
        cv2.rectangle(right, (5, 5), (110, 28), (0, 0, 0), -1)
        cv2.putText(right, f"Conf: {confidence:.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c1, 1)
        x_r = pad * 2 + size
        canvas[40:40+size, x_r:x_r+size] = right
        cv2.putText(canvas, "Enhanced (Research)", (x_r + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cls.GOLD, 1)

        # Bottom info strip
        y_info = 40 + size + 20
        info_text = f"Scale: sigma*={best_sigma}   |   Occlusion: {occlusion_risk}   |   Confidence: {confidence:.3f}"
        cv2.putText(canvas, info_text, (pad, y_info), cv2.FONT_HERSHEY_SIMPLEX, 0.38, cls.WHITE, 1)

        return cls._encode(canvas)

    # ─────────────────────────────────────────────────────────────────────
    # 5.  Betti Number Badge
    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def render_betti_badge(
        cls,
        beta0: int,
        beta1: int,
        topology_status: str,
    ) -> str:
        """
        Compact badge showing Betti numbers and topology status.
        """
        W, H = 260, 80
        canvas = np.full((H, W, 3), cls.BG, dtype=np.uint8)

        # β₀ box
        cv2.rectangle(canvas, (10, 10), (80, 60), cls.PANEL_BG, -1)
        cv2.putText(canvas, "B0", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls.GRAY, 1)
        cv2.putText(canvas, str(beta0), (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cls.CYAN, 2)

        # β₁ box
        cv2.rectangle(canvas, (90, 10), (160, 60), cls.PANEL_BG, -1)
        cv2.putText(canvas, "B1", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls.GRAY, 1)
        cv2.putText(canvas, str(beta1), (115, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cls.CYAN, 2)

        # Status badge
        status_color = cls.GREEN if topology_status == "NOMINAL" else cls.GOLD
        cv2.rectangle(canvas, (170, 15), (250, 55), status_color, -1)
        cv2.putText(canvas, topology_status[:8], (175, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, cls.BG, 1)

        return cls._encode(canvas)

    # ─────────────────────────────────────────────────────────────────────
    # 6.  Impact Summary KPI Strip
    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def render_impact_kpis(
        cls,
        confidence: float,
        best_sigma: float,
        occlusion_risk: str,
        surface_class: str,
        beta0: int,
    ) -> str:
        """
        Horizontal KPI strip with 5 metric cards.
        """
        card_w, card_h = 110, 70
        pad = 6
        n_cards = 5
        W = n_cards * (card_w + pad) + pad
        H = card_h + 2 * pad
        canvas = np.full((H, W, 3), cls.BG, dtype=np.uint8)

        kpis = [
            ("Confidence", f"{confidence:.2f}", cls.GREEN if confidence > 0.6 else cls.RED),
            ("Best Scale", f"σ={best_sigma}", cls.CYAN),
            ("Occlusion", occlusion_risk, {"LOW": cls.GREEN, "MEDIUM": cls.GOLD, "HIGH": cls.RED}.get(occlusion_risk, cls.GRAY)),
            ("Surface", surface_class[:10], {"FLAT": cls.GREEN, "MILDLY_CURVED": cls.GOLD, "HIGHLY_CURVED": cls.RED}.get(surface_class, cls.GRAY)),
            ("Betti β₀", str(beta0), cls.CYAN),
        ]

        for i, (label, value, color) in enumerate(kpis):
            x = pad + i * (card_w + pad)
            y = pad
            cv2.rectangle(canvas, (x, y), (x + card_w, y + card_h), cls.PANEL_BG, -1)
            cv2.putText(canvas, label, (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, cls.GRAY, 1)
            cv2.putText(canvas, value, (x + 8, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return cls._encode(canvas)
