"""
C2 Semantic Shadow Filter — GAP 5
Uses an LVM (Claude / Gemini) as a vision oracle to distinguish
real instrument hands from shadow artifacts.

Includes a geometric-heuristic fallback when the API key is unavailable.
"""
import cv2
import numpy as np
import math
import json
import base64
import os
from dataclasses import dataclass, asdict
from typing import Optional, List


@dataclass
class ValidationResult:
    """Result of validating one candidate keypoint."""
    keypoint: tuple
    weighted_score: float
    decision: str           # "REAL" | "SHADOW" | "UNCERTAIN"
    confidence: float
    scores: dict
    reasoning: str
    accepted: bool          # Final binary decision

    def to_dict(self):
        return {
            'keypoint': list(self.keypoint),
            'weighted_score': self.weighted_score,
            'decision': self.decision,
            'confidence': self.confidence,
            'scores': self.scores,
            'reasoning': self.reasoning,
            'accepted': self.accepted,
        }


class SemanticShadowFilter:
    """
    C2 Shadow Filter — uses an LVM vision oracle to score
    whether candidate keypoints are real hands or shadow artifacts.

    Falls back to a geometric heuristic when no API key is available.
    """

    THRESHOLD_ACCEPT = 0.72   # Above → REAL
    THRESHOLD_REJECT = 0.42   # Below → SHADOW

    WEIGHTS = {
        "origin_alignment":      0.35,
        "geometry_coherence":    0.30,
        "length_plausibility":   0.20,
        "shadow_offset_penalty": 0.15,
    }

    def __init__(self):
        # Load system prompt
        prompt_path = os.path.join(os.path.dirname(__file__), "c2_system_prompt.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = ""

        # Build a cached Gemini client (avoids per-call instantiation)
        self._gemini_client = None
        _api_key = os.getenv("GEMINI_API_KEY")
        if _api_key and self.system_prompt:
            try:
                from google import genai as _genai
                self._gemini_client = _genai.Client(api_key=_api_key)
            except Exception:
                pass
        self.use_lvm = self._gemini_client is not None

    # ──────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────

    def filter_keypoints(
        self,
        image: np.ndarray,
        center: tuple,
        candidates: list,
        face_radius: float = None,
    ) -> List[ValidationResult]:
        """
        Validate all candidate keypoints against shadow detection.

        Args:
            image: BGR numpy array (cropped instrument face)
            center: (cx, cy) center of instrument
            candidates: [(x, y, conf), ...] candidate tip points
            face_radius: estimated radius of instrument face (auto-detected if None)

        Returns:
            List of ValidationResult for each candidate
        """
        if face_radius is None:
            h, w = image.shape[:2]
            face_radius = min(h, w) / 2.0

        if not candidates:
            return []

        if len(candidates) == 1 or not self.use_lvm:
            return [
                self._validate_single(image, center, (float(kp[0]), float(kp[1])), face_radius, candidates)
                for kp in candidates
            ]

        # Parallel Gemini calls — each candidate is independent I/O
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
            futures = [
                ex.submit(self._validate_single, image, center,
                          (float(kp[0]), float(kp[1])), face_radius, candidates)
                for kp in candidates
            ]
            results = [f.result() for f in futures]
        return results

    def get_real_keypoints(self, image, center, candidates, face_radius=None):
        """Convenience — returns only accepted (non-shadow) keypoints."""
        results = self.filter_keypoints(image, center, candidates, face_radius)
        return [r for r in results if r.accepted]

    # ──────────────────────────────────────────────
    # CORE VALIDATION
    # ──────────────────────────────────────────────

    def _validate_single(self, image, center, tip, face_radius, all_candidates):
        """Validate one candidate keypoint."""
        if self.use_lvm:
            try:
                return self._validate_lvm(image, center, tip, face_radius)
            except Exception as e:
                # Fallback to geometric if LVM fails
                return self._validate_geometric(image, center, tip, face_radius, all_candidates,
                                                fallback_reason=f"LVM error: {str(e)[:50]}")
        else:
            return self._validate_geometric(image, center, tip, face_radius, all_candidates)

    # ──────────────────────────────────────────────
    # METHOD 1: LVM ORACLE (Gemini Vision)
    # ──────────────────────────────────────────────

    def _validate_lvm(self, image, center, tip, face_radius):
        """Use Gemini Vision API as the coherence oracle."""
        from PIL import Image as PILImage

        hypothesis_img = self._render_hypothesis(image, center, tip)

        orig_pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        hypo_pil = PILImage.fromarray(cv2.cvtColor(hypothesis_img, cv2.COLOR_BGR2RGB))

        prompt_text = (
            "IMAGE 1 is the original instrument face. "
            "IMAGE 2 has a candidate pointer line drawn on it. "
            "Evaluate this hypothesis. Respond ONLY with valid JSON as specified in your instructions."
        )

        response = self._gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[self.system_prompt, "\n\n", prompt_text, orig_pil, hypo_pil]
        )
        raw = response.text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        return self._parse_response(raw, tip)

    # ──────────────────────────────────────────────
    # METHOD 2: GEOMETRIC HEURISTIC (Fallback)
    # ──────────────────────────────────────────────

    def _validate_geometric(self, image, center, tip, face_radius, all_candidates,
                            fallback_reason=None):
        """
        Pure geometric heuristic — no API call needed.
        Computes the same 4 scores using image analysis.
        """
        cx, cy = center
        tx, ty = tip
        dist = math.hypot(tx - cx, ty - cy)

        # 1. ORIGIN_ALIGNMENT: How close is center to image center?
        h, w = image.shape[:2]
        img_center_x, img_center_y = w / 2.0, h / 2.0
        center_offset = math.hypot(cx - img_center_x, cy - img_center_y)
        origin_score = max(0.0, 1.0 - (center_offset / (face_radius * 0.5)))
        origin_score = min(1.0, origin_score)

        # 2. GEOMETRY_COHERENCE: Edge density along the hand line
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        line_mask = np.zeros_like(edges)
        cv2.line(line_mask, (int(cx), int(cy)), (int(tx), int(ty)), 255, 5)
        overlap = cv2.bitwise_and(edges, line_mask)
        line_pixels = max(1, np.count_nonzero(line_mask))
        geometry_score = min(1.0, np.count_nonzero(overlap) / (line_pixels * 0.3))

        # 3. LENGTH_PLAUSIBILITY: Is length 30%-90% of face radius?
        ratio = dist / face_radius if face_radius > 0 else 0
        if 0.30 <= ratio <= 0.90:
            length_score = 1.0
        elif ratio < 0.30:
            length_score = max(0.0, ratio / 0.30)
        else:
            length_score = max(0.0, 1.0 - (ratio - 0.90) / 0.30)

        # 4. SHADOW_OFFSET_PENALTY: Check if this line is parallel & offset from another
        shadow_score = 1.0  # No shadow evidence by default
        angle_self = math.atan2(ty - cy, tx - cx)
        for other_kp in all_candidates:
            ox, oy = float(other_kp[0]), float(other_kp[1])
            if abs(ox - tx) < 3 and abs(oy - ty) < 3:
                continue  # Skip self
            angle_other = math.atan2(oy - cy, ox - cx)
            angle_diff = abs(angle_self - angle_other)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
            if angle_diff < 0.15:  # Nearly parallel
                # This might be a shadow of the other hand
                other_dist = math.hypot(ox - cx, oy - cy)
                if dist < other_dist * 0.85:
                    shadow_score = 0.3  # Shorter & parallel → likely shadow
                elif abs(dist - other_dist) / max(dist, other_dist) < 0.15:
                    shadow_score = 0.5  # Similar length & parallel → suspicious

        # Weighted score
        ws = (
            origin_score      * self.WEIGHTS["origin_alignment"] +
            geometry_score    * self.WEIGHTS["geometry_coherence"] +
            length_score      * self.WEIGHTS["length_plausibility"] +
            shadow_score      * self.WEIGHTS["shadow_offset_penalty"]
        )
        ws = round(ws, 4)

        # Decision
        if ws >= self.THRESHOLD_ACCEPT:
            decision = "REAL"
            accepted = True
        elif ws <= self.THRESHOLD_REJECT:
            decision = "SHADOW"
            accepted = False
        else:
            decision = "UNCERTAIN"
            accepted = True  # In fallback mode, be permissive with uncertain

        scores = {
            "origin_alignment": round(origin_score, 3),
            "geometry_coherence": round(geometry_score, 3),
            "length_plausibility": round(length_score, 3),
            "shadow_offset_penalty": round(shadow_score, 3),
        }

        reasoning = fallback_reason or (
            f"Geometric heuristic: origin={origin_score:.2f}, "
            f"geometry={geometry_score:.2f}, length={length_score:.2f}, "
            f"shadow={shadow_score:.2f}"
        )

        return ValidationResult(
            keypoint=(tx, ty),
            weighted_score=ws,
            decision=decision,
            confidence=round(ws * 0.9, 3),  # Slightly lower confidence for heuristic
            scores=scores,
            reasoning=reasoning,
            accepted=accepted,
        )

    # ──────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────

    def _render_hypothesis(self, image, center, tip,
                           line_color=(0, 120, 255), thickness=3):
        """Draw candidate pointer line on a copy of the image."""
        out = image.copy()
        cx, cy = int(center[0]), int(center[1])
        tx, ty = int(tip[0]), int(tip[1])
        cv2.line(out, (cx, cy), (tx, ty), line_color, thickness, cv2.LINE_AA)
        cv2.circle(out, (tx, ty), 6, line_color, -1)
        cv2.circle(out, (cx, cy), 5, (255, 255, 0), -1)
        return out

    def _encode_image(self, image):
        """Convert BGR numpy array → base64 JPEG string."""
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return base64.b64encode(buffer).decode("utf-8")

    def _parse_response(self, raw, tip):
        """Parse LVM JSON response into ValidationResult."""
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            return ValidationResult(
                keypoint=tip,
                weighted_score=0.5,
                decision="UNCERTAIN",
                confidence=0.0,
                scores={},
                reasoning="Parse error — LVM response malformed",
                accepted=False,
            )

        ws = float(data.get("weighted_score", 0.5))

        # Override decision with our thresholds (don't trust LVM blindly)
        if ws >= self.THRESHOLD_ACCEPT:
            decision = "REAL"
            accepted = True
        elif ws <= self.THRESHOLD_REJECT:
            decision = "SHADOW"
            accepted = False
        else:
            decision = "UNCERTAIN"
            accepted = False  # Conservative: uncertain = reject

        return ValidationResult(
            keypoint=tip,
            weighted_score=round(ws, 4),
            decision=decision,
            confidence=float(data.get("confidence", 0.5)),
            scores=data.get("scores", {}),
            reasoning=data.get("reasoning", ""),
            accepted=accepted,
        )

    def render_validation_image(self, image, center, results):
        """
        Draw all validated keypoints on image with color-coded results.
        Green = REAL, Red = SHADOW, Yellow = UNCERTAIN.
        Returns annotated image (for UI display).
        """
        out = image.copy()
        cx, cy = int(center[0]), int(center[1])

        color_map = {
            "REAL": (0, 200, 0),
            "SHADOW": (0, 0, 200),
            "UNCERTAIN": (0, 200, 200),
        }

        for r in results:
            tx, ty = int(r.keypoint[0]), int(r.keypoint[1])
            color = color_map.get(r.decision, (128, 128, 128))
            thickness = 3 if r.accepted else 1
            line_style = cv2.LINE_AA if r.accepted else cv2.LINE_4

            cv2.line(out, (cx, cy), (tx, ty), color, thickness, line_style)
            cv2.circle(out, (tx, ty), 7, color, -1)

            # Label
            label = f"{r.decision} {r.weighted_score:.2f}"
            cv2.putText(out, label, (tx + 10, ty - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Center dot
        cv2.circle(out, (cx, cy), 6, (255, 255, 0), -1)
        cv2.putText(out, "CENTER", (cx + 10, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        return cv2.resize(out, (500, 500), interpolation=cv2.INTER_LINEAR)
