import os
import cv2
import numpy as np
import re
import math
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
from app.core.xai import (
    XaiVisualizer, SemanticExplainer, LocalExplainer,
    AdaptiveSemanticRouter, ContrastiveExplainer,
    LimeExplainer, ShapExplainer, compute_entropy,
    IntegratedGradientsExplainer, ROARFidelityScorer,
)
from app.core.metrics import calculate_gauge_reading, calculate_gauge_reading_advanced
from app.core.c2_research import C2ResearchAnalyzer
from app.core.c2_shadow_filter import SemanticShadowFilter
from google import genai
import easyocr
from dotenv import load_dotenv
from datetime import datetime
from app.core.c4_confidence import C4ConfidenceAnalyzer

load_dotenv()
class HARPEngine:
    def __init__(self, base_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- [C1] LOCALIZATION (Gatekeepers) ---
        self.c1_clock_path = os.path.join(base_dir, "models", "c1_localization", "best.pt")
        self.c1_clock_model = self._load_yolo(self.c1_clock_path, "C1_Clock")

        self.c1_gauge_path = os.path.join(base_dir, "models", "c1_gauge_localization", "best.pt")
        self.c1_gauge_model = self._load_yolo(self.c1_gauge_path, "C1_Gauge")

        # --- [C2] STRUCTURE POSE (Specialists) ---
        self.c2_clock_path = os.path.join(base_dir, "models", "c2_hands_skeleton", "best.pt")
        self.c2_clock_model = self._load_yolo(self.c2_clock_path, "C2_Clock")

        self.c2_gauge_path = os.path.join(base_dir, "models", "c2_gauge_skeleton", "best.pt")
        self.c2_gauge_model = self._load_yolo(self.c2_gauge_path, "C2_Gauge")
        
        # --- [C3] ANGLE PREDICTION (Clock Only) ---
        self.c3_path = os.path.join(base_dir, "models", "c3_angle_regression", "best.pth")
        print(f"Loading C3: {self.c3_path}...")
        self.c3_model = self._get_c3_arch().to(self.device)
        
        # --- [C4] PHYSICS LOGIC TABLES (Clock) ---
        self.possible_minutes = np.arange(0, 720)
        self.theory_h = (self.possible_minutes * 0.5) % 360
        self.theory_m = (self.possible_minutes * 6.0) % 360

        # --- [C3] PREPROCESSING ---
        self.c3_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if os.path.exists(self.c3_path):
            raw_sd = torch.load(self.c3_path, map_location=self.device)
            key_map = {"0.fc.weight": "0.fc.1.weight", "0.fc.bias": "0.fc.1.bias"}
            remapped_sd = {key_map.get(k, k): v for k, v in raw_sd.items()}
            missing, unexpected = self.c3_model.load_state_dict(remapped_sd, strict=False)
            if missing:
                print(f"⚠️  C3 load: missing keys: {missing}")
            if unexpected:
                print(f"⚠️  C3 load: unexpected keys: {unexpected}")
            self.c3_model.eval()

            # ── XAI Stack (all 7 layers + IG + ROAR) ──────────────────────────
            self.xai             = XaiVisualizer(self.c3_model[0])        # L1  GradCAM++
            self.ig_explainer    = IntegratedGradientsExplainer(n_steps=20) # L1.5 IG
            self.roar_scorer     = ROARFidelityScorer(mask_fraction=0.20)   # ROAR AFS
            self.local_explainer = LocalExplainer()                         # L2  heuristic
            self.explainer       = SemanticExplainer()                      # L3  Gemini VLM
            self.adaptive_router = AdaptiveSemanticRouter(self.explainer)   # L4  entropy gate
            self.contrastive_xai = ContrastiveExplainer()                   # L5  "Why not?"
            self.lime_explainer  = LimeExplainer()                          # L6  superpixel
            self.shap_explainer  = ShapExplainer()                          # L7  attribution
            self._c3_kalman: list[dict | None] = [None, None]    # per-hand Kalman (clock)
            self._gauge_kalman: dict | None = None                # [T8] gauge needle Kalman
            self._uncertainty_temperature: float = 1.0            # temp scaling T
            print("✅ XAI Stack loaded:")
            print(f"   L1  GradCAM++ (multi-layer): {'✅' if self.xai.available else '⚠️ unavailable'}")
            print("   L1.5 Integrated Gradients (20 steps): ✅")
            print("   ROAR Fidelity Scorer (top-20%): ✅")
            print(f"   L3  Gemini VLM: {'✅' if self.explainer.available else '⚠️ no API key'}")
            print(f"   L4  Entropy router: threshold={AdaptiveSemanticRouter.ENTROPY_THRESHOLD}")
            print(f"   L6  LIME: {'✅' if self.lime_explainer.available else '⚠️ pip install lime'}")
            print(f"   L7  SHAP: {'✅' if self.shap_explainer.available else '⚠️ pip install shap'}")
        else:
            print("⚠️ WARNING: C3 weights not found.")
            self.c3_model = None
            self.xai = self.explainer = self.local_explainer = None
            self.adaptive_router = self.contrastive_xai = None
            self.lime_explainer  = self.shap_explainer  = None
            self.ig_explainer    = self.roar_scorer      = None
            self._c3_kalman = [None, None]
            self._uncertainty_temperature = 1.0
        
        # --- [C2] Research Analyzer ---
        self.c2_research = C2ResearchAnalyzer()
        self.c2_shadow_filter = SemanticShadowFilter()

        print("Loading EasyOCR Fallback...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

        # --- [C4+] CONFIDENCE ANALYZER (Research Extension) ---
        self.c4_confidence = C4ConfidenceAnalyzer()

        # --- Cached Gemini client (avoids re-instantiation on every API call) ---
        self._gemini_client = None
        _api_key = os.getenv("GEMINI_API_KEY")
        if _api_key:
            try:
                self._gemini_client = genai.Client(api_key=_api_key)
            except Exception:
                pass

    def _load_yolo(self, path, name):
        try:
            print(f"Loading {name}: {path}...")
            return YOLO(path)
        except Exception as e:
            print(f"⚠️ {name} Failed: {e}")
            return None

    def _get_c3_arch(self):
        """
        ResNet-18 with Dropout + 2-output (sin θ, cos θ) head for MC-Dropout
        uncertainty estimation.

        Using (sin, cos) instead of a single scalar avoids the 0°/360°
        discontinuity that breaks MSE-based regression near the wrap-around
        boundary.  At inference: residual = atan2(sin_pred, cos_pred).

        Wrapped in nn.Sequential so model[0] gives the backbone directly
        to the XAI hooks (GradCAM++, IG, LIME, SHAP).

        Must match get_c3_model() in scripts/train_c3.py exactly.
        """
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(backbone.fc.in_features, 2)   # (sin θ, cos θ)
        )
        return nn.Sequential(backbone)   # model[0] = backbone for XAI hooks

    # ─────────────────────────────────────────────────────────────────────────────
    def _apply_temperature_scaling(self, sigma: float) -> float:
        """
        Apply temperature scaling to MC-Dropout uncertainty.
        T > 1.0 → inflate uncertainty (model is over-confident)
        T < 1.0 → shrink uncertainty (model is under-confident)
        T = 1.0 → uncalibrated (default)
        """
        return sigma * self._uncertainty_temperature

    def _smooth_c3_angle(self, hand_idx: int, raw_angle: float) -> float:
        """
        Circular Kalman filter applied PER HAND on the C3 output angle (degrees).
        Handles 359°/1° wraparound correctly using circular difference.

        State: [angle, angular_velocity]
        Measurement: raw_angle
        Process noise Q = 5.0, Measurement noise R = 10.0
        """
        Q = 5.0    # process noise (expected max hand movement between frames)
        R = 10.0   # measurement noise (C3 typical error in degrees)

        state = self._c3_kalman[hand_idx]

        if state is None:
            # First observation — initialise
            self._c3_kalman[hand_idx] = {
                "angle": raw_angle,
                "vel":   0.0,
                "P":     [[10.0, 0.0], [0.0, 5.0]]  # covariance
            }
            return raw_angle

        # Predict
        angle_pred = (state["angle"] + state["vel"]) % 360.0
        P = state["P"]
        P_pred = [
            [P[0][0] + Q, P[0][1]],
            [P[1][0],     P[1][1] + Q]
        ]

        # Circular innovation (wrap to ±180)
        innov = raw_angle - angle_pred
        innov = (innov + 180.0) % 360.0 - 180.0

        # Kalman gain (scalar observation of angle only)
        K0 = P_pred[0][0] / (P_pred[0][0] + R)
        K1 = P_pred[1][0] / (P_pred[0][0] + R)

        # Update
        angle_upd = (angle_pred + K0 * innov) % 360.0
        vel_upd   = state["vel"] + K1 * innov

        self._c3_kalman[hand_idx] = {
            "angle": angle_upd,
            "vel":   float(np.clip(vel_upd, -15.0, 15.0)),  # cap velocity
            "P":     [
                [P_pred[0][0] * (1 - K0), P_pred[0][1]],
                [P_pred[1][0] * (1 - K1), P_pred[1][1]]
            ]
        }
        return angle_upd

    def _enable_dropout(self):
        """Set all Dropout layers to train mode so they remain stochastic during inference."""
        if self.c3_model is not None:
            for m in self.c3_model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

    def _predict_with_uncertainty(self, tensor: torch.Tensor, n_passes: int = 8):
        """
        Monte Carlo Dropout Uncertainty Estimation.

        Temporarily enables dropout during N stochastic forward passes to estimate
        epistemic uncertainty. Uses circular statistics to correctly handle the
        0°/360° wraparound in angle predictions.

        Args:
            tensor:   (1, 3, 64, 64) normalised input tensor.
            n_passes: Number of stochastic forward passes (default: 20).

        Returns:
            mean_angle_deg (float): Circular mean of all stochastic predictions.
            std_deg        (float): Circular standard deviation — uncertainty proxy.
        """
        if self.c3_model is None:
            return 0.0, 0.0

        # Keep BatchNorm in eval mode (stable stats), Dropout in train mode (random masking)
        self.c3_model.eval()
        self._enable_dropout()

        preds_deg = []
        with torch.no_grad():
            for _ in range(n_passes):
                raw = self.c3_model(tensor)[0]       # shape [2]: (sin θ, cos θ)
                sin_p = raw[0].item()
                cos_p = raw[1].item()
                angle_deg = math.degrees(math.atan2(sin_p, cos_p)) % 360.0
                preds_deg.append(angle_deg)

        # Restore full eval mode (no stochasticity)
        self.c3_model.eval()

        # Circular statistics — pure numpy, no scipy dependency.
        # Essential for angles because 1° and 359° are only 2° apart.
        preds_rad = np.radians(preds_deg)
        mean_sin  = float(np.mean(np.sin(preds_rad)))
        mean_cos  = float(np.mean(np.cos(preds_rad)))
        mean_deg  = float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0)
        # Mardia & Jupp circular std: σ = sqrt(-2 ln R),  R = resultant length ∈ [0,1]
        R         = math.sqrt(mean_sin ** 2 + mean_cos ** 2)
        std_deg   = float(np.degrees(math.sqrt(max(0.0, -2.0 * math.log(R + 1e-9)))))

        # Cache per-call uncertainties for surface in result payload
        if not hasattr(self, '_last_uncertainties'):
            self._last_uncertainties = []
        self._last_uncertainties.append(std_deg)

        return mean_deg, std_deg

    def _get_angle(self, center, point):
        dx, dy = point[0] - center[0], point[1] - center[1]
        angle = math.degrees(math.atan2(dx, -dy))
        return angle + 360 if angle < 0 else angle

    # --- CLOCK PHYSICS SOLVER (Restored from main branch) ---
    def _solve_physics(self, a1, a2, length1=None, length2=None, top_n=3, sigma=15.0):
        telemetry_log = "[C4 Physics Initialized]"
        
        # Correct individual hand difference logic
        diff_h1_a = np.abs(a1 - self.theory_h)
        diff_m1_a = np.abs(a2 - self.theory_m)
        err_a = np.minimum(diff_h1_a, 360 - diff_h1_a) + np.minimum(diff_m1_a, 360 - diff_m1_a)
        
        diff_h1_b = np.abs(a2 - self.theory_h)
        diff_m1_b = np.abs(a1 - self.theory_m)
        err_b = np.minimum(diff_h1_b, 360 - diff_h1_b) + np.minimum(diff_m1_b, 360 - diff_m1_b)
        
        # --- UPSTREAM HEURISTIC DATA FUSION: HAND MORPHOLOGY WEIGHTING ---
        if length1 is not None and length2 is not None and length1 > 0 and length2 > 0:
            ratio = length1 / length2
            penalty = 30.0 
            if ratio < 0.9: 
                err_b += penalty 
                telemetry_log += f" | Hand 1 ({length1:.1f}px) < Hand 2 ({length2:.1f}px) -> Rewarded Hypothesis A (Ang1=Hour), Applied {penalty}deg penalty to B."
            elif ratio > 1.1:
                err_a += penalty 
                telemetry_log += f" | Hand 1 ({length1:.1f}px) > Hand 2 ({length2:.1f}px) -> Rewarded Hypothesis B (Ang2=Hour), Applied {penalty}deg penalty to A."
        
        candidates = []
        for i in range(720):
            h = int(i // 60)
            if h == 0: h = 12
            m = int(i % 60)
            
            conf_a = np.exp(-0.5 * (err_a[i] / sigma)**2) * 100
            conf_b = np.exp(-0.5 * (err_b[i] / sigma)**2) * 100
            
            candidates.append({'error': err_a[i], 'confidence': conf_a, 'hour': h, 'minute': m, 'hypothesis': 'A'})
            candidates.append({'error': err_b[i], 'confidence': conf_b, 'hour': h, 'minute': m, 'hypothesis': 'B'})
            
        candidates.sort(key=lambda x: x['error'])
        
        top_candidates = []
        seen_times = set()
        for cand in candidates:
            time_key = f"{cand['hour']}:{cand['minute']}"
            if time_key not in seen_times:
                top_candidates.append(cand)
                seen_times.add(time_key)
            if len(top_candidates) >= top_n:
                break
                
        best = top_candidates[0]
        telemetry_log += f" | Search Complete. Minimum Error {best['error']:.2f}deg -> Winning Hypothesis: {best['hypothesis']} ({best['confidence']:.1f}% Confidence)."
        return best['hour'], best['minute'], best['error'], best['confidence'], top_candidates, telemetry_log

    def _get_crop(self, img, center, angle, box_size=128):
        """
        Rotate image CCW by angle so the hand is approximately vertical, then
        crop box_size×box_size around the clock centre.

        FIX: the original version returned an empty array whenever the crop
        window touched an image boundary, silently skipping C3 for hands near
        the edge.  This version pads symmetrically instead, matching the
        behaviour of rotate_and_crop() in generate_c3_dataset.py.

        Padding is tracked per-side so the crop origin is only shifted by
        the padding actually added on that side (the old merged pad_w/pad_h
        approach would shift the crop incorrectly when only one side overflowed).
        """
        h, w = img.shape[:2]
        cx, cy = float(center[0]), float(center[1])

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

        half = box_size // 2
        x1 = int(cx - half)
        y1 = int(cy - half)

        pad_left   = max(0, -x1)
        pad_top    = max(0, -y1)
        pad_right  = max(0, (x1 + box_size) - w)
        pad_bottom = max(0, (y1 + box_size) - h)

        if pad_left or pad_top or pad_right or pad_bottom:
            rotated = cv2.copyMakeBorder(
                rotated, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            x1 += pad_left
            y1 += pad_top

        return rotated[y1 : y1 + box_size, x1 : x1 + box_size]

    def _resize_small(self, img):
        """Helper to force 500x500px output for dashboard efficiency"""
        return cv2.resize(img, (500, 500), interpolation=cv2.INTER_LINEAR)

    def _infer_ampm(self, crop):
        if crop is None or crop.size == 0: return "Unknown", 0.0
        if self._gemini_client is None:
            return "Unknown (Missing API Key)", 0.0
        try:
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            prompt = "Look at this cropped image of a clock. Based on the lighting, shadows, colors, and overall ambiance, guess whether this photo was taken during the day (AM) or night (PM). Return ONLY 'AM' or 'PM'."
            response = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash", contents=[prompt, pil_img]
            )
            text = response.text.strip().upper()
            if "AM" in text:
                return "AM", 0.9
            elif "PM" in text:
                return "PM", 0.9
            else:
                return "Unknown", 0.5
        except Exception:
            return "Unknown", 0.0

    def _resolve_ambiguity(self, a1, a2, h, m):
        diff = min((a1 - a2) % 360, (a2 - a1) % 360)
        warning = None
        if diff < 10.0:
            if h == 12 and m == 0:
                warning = "Perfect overlap at 12:00. Time is unambiguous."
            elif h == 6 and m == 30:
                warning = "Ambiguity Handled: Hands overlap near 6 position. Resolved securely as 6:30 based on physics constraints."
            else:
                warning = f"Ambiguity Warning: Hands overlap at {h}:{m:02d}. Reading may be difficult."
        return warning

    def _calculate_drift(self, detected_h, detected_m, device_time_str=None):
        if device_time_str:
            try:
                device_time = datetime.fromisoformat(device_time_str.replace("Z", "+00:00"))
            except ValueError:
                device_time = datetime.now()
        else:
            device_time = datetime.now()
            
        real_h = device_time.hour % 12
        real_h = 12 if real_h == 0 else real_h
        real_m = device_time.minute
        
        det_total_m = (detected_h % 12) * 60 + detected_m
        real_total_m = (real_h % 12) * 60 + real_m
        
        diff = det_total_m - real_total_m
        if diff > 360: diff -= 720
        elif diff < -360: diff += 720
            
        if diff == 0: return "Perfectly accurate (0 min drift)"
        elif diff > 0: return f"Fast by {abs(diff)} minutes"
        else: return f"Slow by {abs(diff)} minutes"

    def _extract_gauge_scale_gemini(self, img, min_roi=None, max_roi=None):
        """
        T5: Accept optional pre-cropped ROIs near the min/max keypoints.
        When ROIs are supplied, stitch them and ask Gemini which label is min and which is max,
        giving much tighter focus than the full gauge crop.
        Falls back to full-crop prompt when ROIs are unavailable.
        """
        if self._gemini_client is None:
            return None, None, "Missing GEMINI_API_KEY in .env"

        try:
            if min_roi is not None and max_roi is not None:
                # --- [T5] ROI-targeted prompt ---
                # Resize both patches to same height for side-by-side stitch
                h = max(min_roi.shape[0], max_roi.shape[1], 80)
                min_patch = cv2.resize(min_roi, (h, h))
                max_patch = cv2.resize(max_roi, (h, h))
                stitched = np.hstack([min_patch, max_patch])
                pil_img = Image.fromarray(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
                prompt = (
                    "This image shows two small patches from an analog gauge: "
                    "LEFT patch is near the scale minimum mark, RIGHT patch is near the scale maximum mark. "
                    "Read the numeric value from EACH patch and return them as: <left_value>, <right_value>. "
                    "Return ONLY two numbers separated by a comma. Do not include units or other text."
                )
            else:
                # --- Fallback: full-crop prompt ---
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                prompt = (
                    "Look at this analog gauge. Return ONLY the minimum scale reading and the maximum "
                    "scale reading printed on it, separated by a comma. "
                    "Do not include units or any other text. For example: 0, 100 or -10, 50."
                )

            response = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash", contents=[prompt, pil_img]
            )
            text = response.text.strip()

            parts = [p.strip() for p in text.split(',')]
            if len(parts) >= 2:
                min_match = re.findall(r"[-+]?\d*\.\d+|\d+", parts[0])
                max_match = re.findall(r"[-+]?\d*\.\d+|\d+", parts[1])
                min_val = float(min_match[0]) if min_match else None
                max_val = float(max_match[0]) if max_match else None
                return min_val, max_val, None
            return None, None, f"Could not parse format: {text}"
        except Exception as e:
            return None, None, str(e)


    def _get_roi_inward(self, img, center_pt, target_pt, patch_size=60, shift_pixels=25):
        """Calculates a vector from target_pt toward center_pt, and shifts the crop box inward."""
        h, w = img.shape[:2]
        cx, cy = center_pt[0], center_pt[1]
        tx, ty = target_pt[0], target_pt[1]
        
        # Vector from target toward center
        dx, dy = cx - tx, cy - ty
        length = math.hypot(dx, dy)
        
        if length > 0:
            ux, uy = dx/length, dy/length
            # Shift the center of the crop inward
            crop_x = int(tx + ux * shift_pixels)
            crop_y = int(ty + uy * shift_pixels)
        else:
            crop_x, crop_y = int(tx), int(ty)
            
        half = patch_size // 2
        y1, y2 = max(0, crop_y - half), min(h, crop_y + half)
        x1, x2 = max(0, crop_x - half), min(w, crop_x + half)
        
        if x2 - x1 == 0 or y2 - y1 == 0: return None
        return img[y1:y2, x1:x2]

    def _extract_number(self, roi_img):
        """Local EasyOCR extraction on a small image crop."""
        if roi_img is None or roi_img.size == 0: return None
        
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        results = self.reader.readtext(thresh)
        if not results: return None
        
        full_text = "".join([res[1] for res in results])
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", full_text)
        if matches:
            try: return float(matches[0])
            except: return None
        return None

    def _get_crop_from_box(self, img, box, pad=30):
        if box is None: return None, None
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w = img.shape[:2]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        return img[y1:y2, x1:x2], (x1, y1, x2, y2)

    # ─────────────────────────────────────────────────────────────────────────
    # [GAUGE XAI] Gemini Vision semantic explainer — equivalent of clock L3
    # ─────────────────────────────────────────────────────────────────────────
    def _explain_gauge_xai(self, crop, reading, min_val, max_val, span_deg: float, needle_deg: float) -> str:
        """
        Gauge-specific XAI via Gemini Vision (L3 semantic explanation equivalent).
        Asks Gemini to describe where the needle visually points and whether the
        computed reading is consistent with what it sees in the image.

        Returns a plain-language explanation string, or a fallback on error.
        """
        if self._gemini_client is None:
            return "Gemini XAI unavailable (missing GEMINI_API_KEY)."
        try:
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            scale_info = (
                f"Scale range: {min_val} to {max_val}"
                if min_val is not None and max_val is not None
                else "Scale range: unknown"
            )
            prompt = (
                f"You are reviewing an analog gauge image. The automated HARP system detected:\n"
                f"  • {scale_info}\n"
                f"  • Total scale span: {span_deg:.1f}°\n"
                f"  • Needle angular offset from minimum mark: {needle_deg:.1f}°\n"
                f"  • Computed reading: {reading}\n\n"
                f"In 2–3 concise sentences, describe where the needle visually appears to be "
                f"pointing and whether this is consistent with the computed reading. "
                f"Mention any visual uncertainty (glare, shadow, small tick marks) you observe."
            )
            response = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash", contents=[prompt, pil_img]
            )
            return response.text.strip()
        except Exception as e:
            return f"Gemini XAI error: {e}"

    def _compute_gauge_confidence(self, c2_kpt_conf: float, scale_stage: str, quality_score: float):
        """
        Compute a real confidence score for gauge readings, replacing the hardcoded 'High'.

        Weights:
          - c2_kpt_conf  (0–1)    → 45%  C2 keypoint detection quality
          - scale_stage  weight   → 35%  scale extraction quality
          - image quality (0–100) → 20%  blur/brightness/contrast

        Returns:
            confidence_pct (float): 0–100
            tier (str): 'HIGH' | 'MEDIUM' | 'LOW' | 'UNRELIABLE'
        """
        stage_weights = {"manual": 1.0, "gemini": 0.90, "ocr": 0.70, "failed": 0.15}
        sw = stage_weights.get(scale_stage, 0.50)
        score = float(
            max(0.0, min(100.0,
                (c2_kpt_conf * 0.45 + sw * 0.35 + (quality_score / 100.0) * 0.20) * 100.0
            ))
        )
        if score >= 80:   tier = "HIGH"
        elif score >= 55: tier = "MEDIUM"
        elif score >= 30: tier = "LOW"
        else:             tier = "UNRELIABLE"
        return round(score, 1), tier

    def _score_quality(self, img):
        if img is None or img.size == 0: return {"blur": 0, "brightness": 0, "contrast": 0, "overall": 0}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)
        blur_score = min(blur / 500.0 * 100, 100) 
        bright_score = 100 - (abs(brightness - 127) / 127 * 100)
        contrast_score = min(contrast / 60.0 * 100, 100)
        overall = (blur_score * 0.4) + (bright_score * 0.3) + (contrast_score * 0.3)
        return {"blur": blur, "brightness": brightness, "contrast": contrast, "overall": max(0, min(100, overall))}

    def _localize_all(self, img):
        def _run_clock():
            return self.c1_clock_model(img, verbose=False)[0] if self.c1_clock_model else None
        def _run_gauge():
            return self.c1_gauge_model(img, verbose=False)[0] if self.c1_gauge_model else None

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_c = ex.submit(_run_clock)
            f_g = ex.submit(_run_gauge)
            clock_res = f_c.result()
            gauge_res = f_g.result()

        c_box = clock_res.boxes[0] if clock_res and len(clock_res.boxes) > 0 else None
        g_box = gauge_res.boxes[0] if gauge_res and len(gauge_res.boxes) > 0 else None

        c_crop, c_bbox = self._get_crop_from_box(img, c_box)
        g_crop, g_bbox = self._get_crop_from_box(img, g_box)

        c_conf = c_box.conf.item() if c_box else -1.0
        g_conf = g_box.conf.item() if g_box else -1.0

        return c_crop, c_bbox, c_conf, g_crop, g_bbox, g_conf

    def _draw_bbox(self, img, bbox, detected_type):
        img_copy = img.copy()
        x1, y1, x2, y2 = bbox
        label = f"{detected_type.capitalize()} Detected"
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(img_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
     
        return self._resize_small(img_copy)

    # --- [C2] SKELETON VISUALIZATION (No Text) ---
    def _draw_skeleton(self, img, center, tip1, tip2):
        img_copy = img.copy()
        center_pt = (int(center[0]), int(center[1]))
        tip1_pt = (int(tip1[0]), int(tip1[1]))
        tip2_pt = (int(tip2[0]), int(tip2[1]))
        
        cv2.line(img_copy, center_pt, tip1_pt, (0, 255, 0), 4)
        cv2.line(img_copy, center_pt, tip2_pt, (0, 0, 255), 4)
        cv2.circle(img_copy, center_pt, 8, (255, 0, 0), -1)
        cv2.circle(img_copy, tip1_pt, 8, (0, 255, 0), -1)
        cv2.circle(img_copy, tip2_pt, 8, (0, 0, 255), -1)
        
        return self._resize_small(img_copy)

    # --- [C3] ANGLE VISUALIZATION (With Text) ---
    def _draw_angles_on_img(self, img, center, tip1, tip2, a1, a2):
        img_copy = img.copy()
        center_pt = (int(center[0]), int(center[1]))
        tip1_pt = (int(tip1[0]), int(tip1[1]))
        tip2_pt = (int(tip2[0]), int(tip2[1]))
        
        cv2.line(img_copy, center_pt, tip1_pt, (0, 255, 0), 4)
        cv2.line(img_copy, center_pt, tip2_pt, (0, 0, 255), 4)
        cv2.circle(img_copy, center_pt, 8, (255, 0, 0), -1)
        cv2.circle(img_copy, tip1_pt, 8, (0, 255, 0), -1)
        cv2.circle(img_copy, tip2_pt, 8, (0, 0, 255), -1)
        
        # TEXT
        cv2.putText(img_copy, f"H: {a1:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img_copy, f"M: {a2:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return self._resize_small(img_copy)

    def _draw_gauge_skeleton(self, img, center, min_pt, max_pt, tip, parsed_min="?", parsed_max="?"):
        img_copy = img.copy()
        center_pt = (int(center[0]), int(center[1]))
        min_p = (int(min_pt[0]), int(min_pt[1]))
        max_p = (int(max_pt[0]), int(max_pt[1]))
        tip_p = (int(tip[0]), int(tip[1]))

        # Draw Scale Limits & Needle
        cv2.line(img_copy, center_pt, min_p, (255, 100, 0), 2) # Blue-ish (Start)
        cv2.line(img_copy, center_pt, max_p, (0, 0, 255), 2)   # Red (End)
        cv2.line(img_copy, center_pt, tip_p, (0, 255, 0), 3)   # Green (Needle)
        cv2.circle(img_copy, center_pt, 6, (255, 255, 255), -1)
        
        # Draw min/max values
        cv2.putText(img_copy, f"Min: {parsed_min}", (min_p[0] - 20, min_p[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        cv2.putText(img_copy, f"Max: {parsed_max}", (max_p[0] - 20, max_p[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return self._resize_small(img_copy)

    def _draw_gauge_angles(self, img, center, min_pt, max_pt, tip, span, needle):
        img_copy = img.copy()
        center_pt = (int(center[0]), int(center[1]))
        min_p = (int(min_pt[0]), int(min_pt[1]))
        max_p = (int(max_pt[0]), int(max_pt[1]))
        tip_p = (int(tip[0]), int(tip[1]))

        # Draw Scale Limits & Needle
        cv2.line(img_copy, center_pt, min_p, (255, 100, 0), 2)
        cv2.line(img_copy, center_pt, max_p, (0, 0, 255), 2)
        cv2.line(img_copy, center_pt, tip_p, (0, 255, 0), 3)
        cv2.circle(img_copy, center_pt, 6, (255, 255, 255), -1)

        cv2.putText(img_copy, f"Span: {span:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img_copy, f"Needle: {needle:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return self._resize_small(img_copy)

    def analyze(self, img_array, force_expert=False, manual_min_val="", manual_max_val="", device_time_str=None):
        debug_info = []
        visualizations = {}

        # --- [C1] LOCALIZATION (Runs dynamically for both modes now) ---
        c_crop, c_bbox, c_conf, g_crop, g_bbox, g_conf = self._localize_all(img_array)
        
        if c_conf == -1.0 and g_conf == -1.0:
            debug_info.append(f"C1: No Object Found - Stopping")
            visualizations['c1_detection'] = self._resize_small(img_array.copy())
            return {
                "time": "N/A", 
                "method": f"no clock face or gauge in the uploaded image",
                "confidence": "0.0",
                "heatmap": None,
                "debug": debug_info,
                "visualizations": visualizations,
                "angles": {"hand1": 0.0, "hand2": 0.0},
                "reasoning": f"No valid clock or gauge detected.",
                "error": f"no clock face or gauge in the uploaded image"
            }

        c1_detected_type = 'clock' if c_conf > g_conf else 'gauge'
        debug_info.append(f"C1: Initial guess is {c1_detected_type.capitalize()}")

        # --- [C2] CROSS-VALIDATION (Parallel) ---
        def _run_c2_clock():
            if self.c2_clock_model and c_crop is not None:
                res = self.c2_clock_model(c_crop, verbose=False)[0]
                if res.keypoints and len(res.keypoints.data) > 0:
                    kpts = res.keypoints.data[0].cpu().numpy()
                    if len(kpts) >= 3:
                        return float(np.mean(kpts[:3, 2])), kpts
            return 0.0, None

        def _run_c2_gauge():
            if self.c2_gauge_model and g_crop is not None:
                res = self.c2_gauge_model(g_crop, verbose=False)[0]
                if res.keypoints and len(res.keypoints.data) > 0:
                    kpts = res.keypoints.data[0].cpu().numpy()
                    if len(kpts) >= 4:
                        return float(np.mean(kpts[:4, 2])), kpts
            return 0.0, None

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_ck = ex.submit(_run_c2_clock)
            f_gk = ex.submit(_run_c2_gauge)
            c2_clock_conf, clock_kpts = f_ck.result()
            c2_gauge_conf, gauge_kpts = f_gk.result()

        if c2_clock_conf == 0.0 and c2_gauge_conf == 0.0:
             return {"error": "no clock face or gauge in the uploaded image (C2 verification failed)"}
             
        # The true type is whichever C2 model is more confident about its keypoints
        detected_type = 'clock' if c2_clock_conf > c2_gauge_conf else 'gauge'
        target_crop = c_crop if detected_type == 'clock' else g_crop
        bbox = c_bbox if detected_type == 'clock' else g_bbox
        conf = c_conf if detected_type == 'clock' else g_conf
        
        debug_info.append(f"C2 Validation: Chose {detected_type.capitalize()} (Clock KP Conf: {c2_clock_conf:.2f}, Gauge KP Conf: {c2_gauge_conf:.2f})")

        quality = self._score_quality(target_crop)
        c_quality = self._score_quality(c_crop) if c_crop is not None else {"blur": 0, "brightness": 0, "contrast": 0, "overall": 0}
        g_quality = self._score_quality(g_crop) if g_crop is not None else {"blur": 0, "brightness": 0, "contrast": 0, "overall": 0}
        
        visualizations['c1_detection'] = self._draw_bbox(img_array, bbox, detected_type)

        # ==========================================
        # GAUGE LOGIC PIPELINE
        # ==========================================
        if detected_type == 'gauge':
            
            kpts = gauge_kpts
            
            center, min_pt, max_pt, tip = kpts[0][:2], kpts[1][:2], kpts[2][:2], kpts[3][:2]
            
            # --- STAGE 1: MANUAL OVERRIDE ---
            min_val, max_val = None, None
            override_active = False
            scale_stage = "failed"  # tracks which extraction stage succeeded
            
            if manual_min_val and manual_max_val:
                try:
                    min_val = float(manual_min_val)
                    max_val = float(manual_max_val)
                    override_active = True
                    scale_stage = "manual"
                    debug_info.append(f"Stage 1 (Manual Override): Min={min_val}, Max={max_val}")
                except Exception as e:
                    debug_info.append(f"Stage 1 (Manual Override) Failed to parse: {e}")

            # --- STAGE 2: GEMINI API (T5: pass ROI crops for targeted reading) ---
            err_str = None
            if not override_active:
                # Build ROI crops from keypoints for T5
                _min_roi = self._get_roi_inward(target_crop, center, min_pt)
                _max_roi = self._get_roi_inward(target_crop, center, max_pt)
                min_val, max_val, err_str = self._extract_gauge_scale_gemini(
                    target_crop, min_roi=_min_roi, max_roi=_max_roi
                )
                if err_str:
                    debug_info.append(f"Stage 2 (Gemini API) Failed: {err_str}")
                elif min_val is not None and max_val is not None:
                    scale_stage = "Completed"
                    debug_info.append(f"Stage 2 (Gemini API): Min={min_val}, Max={max_val}")
            
            # --- STAGE 3: LOCAL FALLBACK (INWARD SHIFT OCR) ---
            if not override_active and (min_val is None or max_val is None):
                debug_info.append("Stage 3 (Local API Fallback): Using Inward-Shift OCR.")
                min_roi = self._get_roi_inward(target_crop, center, min_pt)
                max_roi = self._get_roi_inward(target_crop, center, max_pt)
                min_val = self._extract_number(min_roi)
                max_val = self._extract_number(max_roi)
                if min_val is not None and max_val is not None:
                    scale_stage = "ocr"
                debug_info.append(f"Stage 3 (Local API Fallback): Min={min_val}, Max={max_val}")
            
            parsed_min = min_val if min_val is not None else "Failed"
            parsed_max = max_val if max_val is not None else "Failed"
            debug_info.append(f"Scale Extraction Stage: {scale_stage}")
            
            a_min = self._get_angle(center, min_pt)
            a_max = self._get_angle(center, max_pt)
            a_tip = self._get_angle(center, tip)
            span = (a_max - a_min + 360) % 360
            needle = (a_tip - a_min + 360) % 360
            
            units_per_deg = 0.0
            # C4 Physics Logic 
            if min_val is not None and max_val is not None and min_val <= max_val:
                reading, units_per_deg = calculate_gauge_reading_advanced(span, needle, min_val, max_val)
                time_str = str(reading)
                method_str = "Advanced Gauge Reading (C1+C2+OCR+C4)"
                reasoning_str = f"Gauge Logic: 1° = {units_per_deg:.4f} units. Formula: {min_val} + ({needle:.1f}° * {units_per_deg:.4f}) = {reading}"
            else:
                if min_val is not None and max_val is not None and min_val > max_val:
                    debug_info.append(f"Scale Reversed: min={min_val} > max={max_val}. Falling back to raw percentage.")
                reading = calculate_gauge_reading([center, min_pt, max_pt, tip])
                time_str = f"{reading}%"
                method_str = "Gauge Reading - Fallback (C1+C2+C4)"
                reasoning_str = "Gauge Logic: OCR extraction failed. Interpreted raw percentage. Please use 'Manual Gauge Scale Overrides' in settings."

            visualizations['c2_skeleton'] = self._draw_gauge_skeleton(target_crop, center, min_pt, max_pt, tip, parsed_min, parsed_max)
            visualizations['c3_angles'] = self._draw_gauge_angles(target_crop, center, min_pt, max_pt, tip, span, needle)

            # --- C2 Shadow Filter + Research (expert path only — LVM calls are expensive) ---
            shadow_results = []
            c2_research_data = None
            if force_expert:
                try:
                    candidates = [kpts[i] for i in range(1, min(4, len(kpts)))]
                    shadow_results = self.c2_shadow_filter.filter_keypoints(target_crop, center, candidates)
                    shadow_viz = self.c2_shadow_filter.render_validation_image(target_crop, center, shadow_results)
                    visualizations['c2_shadow'] = shadow_viz
                    debug_info.append(f"Shadow Filter: {sum(1 for r in shadow_results if r.accepted)}/{len(shadow_results)} accepted")
                except Exception as e:
                    debug_info.append(f"Shadow Filter Error: {e}")

                try:
                    c2_research_data = self.c2_research.analyze(target_crop, kpts, detected_type='gauge',
                                                                shadow_results=shadow_results)
                except Exception as e:
                    debug_info.append(f"C2 Research Error: {e}")

            # --- [T1] REAL CONFIDENCE SCORE (replaces hardcoded 'High') ---
            gauge_conf_score, gauge_conf_tier = self._compute_gauge_confidence(
                c2_kpt_conf=c2_gauge_conf,
                scale_stage=scale_stage,
                quality_score=quality.get("overall", 0)
            )
            gauge_confidence = {"score": gauge_conf_score, "tier": gauge_conf_tier}
            debug_info.append(f"Gauge Confidence: {gauge_conf_tier} ({gauge_conf_score:.1f}/100) — stage={scale_stage}, kpt_conf={c2_gauge_conf:.3f}")

            # --- [T1] GAUGE XAI — Gemini Vision explanation (force_expert only) ---
            gauge_xai_text = ""
            if force_expert:
                debug_info.append("Gauge XAI: Requesting Gemini Vision explanation...")
                try:
                    gauge_xai_text = self._explain_gauge_xai(
                        target_crop, time_str, min_val, max_val, span, needle
                    )
                    debug_info.append(f"Gauge XAI: {gauge_xai_text[:80]}..." if len(gauge_xai_text) > 80 else f"Gauge XAI: {gauge_xai_text}")
                except Exception as e:
                    debug_info.append(f"Gauge XAI Error: {e}")

            # --- [T3] C4 CONFIDENCE ANALYZER — gauge-adapted call ---
            # Reuse C4ConfidenceAnalyzer with gauge semantics:
            #   a1 = needle angle from min (span start), a2 = span end (fixed = span)
            #   error = angular "error proxy" inferred from confidence gap (lower score → higher error)
            #   h, m = 0, 0  (unused clock fields — pass neutral values)
            try:
                _error_proxy = max(0.0, (1.0 - c2_gauge_conf) * 20.0)  # 0° at conf=1.0, 20° at conf=0.0
                c4_gauge_result = self.c4_confidence.analyze(
                    a1=needle, a2=span, error=_error_proxy, h=0, m=0
                )
                c4_gauge_dict = c4_gauge_result.to_dict()
                debug_info.append(f"C4 Gauge Confidence: {c4_gauge_dict['tier']} ({c4_gauge_dict['score']:.1f}/100)")
            except Exception as _e:
                c4_gauge_dict = None
                debug_info.append(f"C4 Gauge Confidence Error: {_e}")

            # --- [T4] GAUGE UNCERTAINTY QUANTIFICATION ---
            # Propagates three independent uncertainty sources into a single estimate
            #   kpt_unc:   how much keypoint jitter could shift the needle (std proxy)
            #   scale_unc: uncertainty from scale extraction method
            #   quality_unc: image quality degradation factor
            _scale_unc_map = {"manual": 0.0, "gemini": 2.0, "ocr": 5.0, "failed": 15.0}
            _kpt_unc    = (1.0 - c2_gauge_conf) * 10.0           # 0–10° equivalent
            _scale_unc  = _scale_unc_map.get(scale_stage, 8.0)   # reading error from method
            _qual_unc   = max(0.0, (100 - quality.get("overall", 50)) / 20.0)  # 0–5°
            _total_unc  = round((_kpt_unc**2 + _scale_unc**2 + _qual_unc**2) ** 0.5, 2)
            gauge_uncertainty = {
                "total_deg":    _total_unc,
                "kpt_deg":      round(_kpt_unc, 2),
                "scale_deg":    _scale_unc,
                "quality_deg":  round(_qual_unc, 2),
                "summary":      f"±{_total_unc:.1f}° total (kpt={_kpt_unc:.1f}°, scale={_scale_unc:.1f}°, qual={_qual_unc:.1f}°)",
            }
            debug_info.append(f"Gauge Uncertainty: {gauge_uncertainty['summary']}")

            return {
                "time": time_str,
                "method": method_str,
                "confidence": f"{gauge_conf_score:.1f}%",
                "heatmap": None,
                "debug": debug_info + [f"Final Reading: {time_str}"],
                "visualizations": visualizations,
                "angles": {"span": span, "needle": needle, "units_per_deg": units_per_deg},
                "scale": {"min": parsed_min, "max": parsed_max},
                "scale_stage": scale_stage,
                "reasoning": reasoning_str,
                "error": "",
                "c1_conf": float(conf),
                "c1_quality": quality,
                "c1_clock_conf": float(c_conf) if c_conf != -1.0 else 0.0,
                "c1_gauge_conf": float(g_conf) if g_conf != -1.0 else 0.0,
                "c1_clock_quality": c_quality,
                "c1_gauge_quality": g_quality,
                "c2_research": c2_research_data,
                "gauge_confidence": gauge_confidence,
                "keypoint_confidence": round(float(c2_gauge_conf), 4),
                "gauge_xai": gauge_xai_text,
                "c4_confidence": c4_gauge_dict,       # [T3] gauge-adapted C4 confidence
                "gauge_uncertainty": gauge_uncertainty, # [T4] uncertainty breakdown
            }


        # ==========================================
        # CLOCK LOGIC PIPELINE
        # ==========================================
        elif detected_type == 'clock':
            kpts = clock_kpts
            center, tip1, tip2 = kpts[0][:2], kpts[1][:2], kpts[2][:2]
            
            a1 = self._get_angle(center, tip1)
            a2 = self._get_angle(center, tip2)
            
            len1 = math.hypot(center[0]-tip1[0], center[1]-tip1[1])
            len2 = math.hypot(center[0]-tip2[0], center[1]-tip2[1])
            
            visualizations['c2_skeleton'] = self._draw_skeleton(target_crop, center, tip1, tip2)
            visualizations['c3_angles'] = self._draw_angles_on_img(target_crop, center, tip1, tip2, a1, a2)

            h, m, error, conf, cands, telemetry_log = self._solve_physics(a1, a2, length1=len1, length2=len2)

            ambiguity_warning = self._resolve_ambiguity(a1, a2, h, m)
            drift_str = self._calculate_drift(h, m, device_time_str)

            # [C4+] Physics Confidence Assessment (Research Extension)
            c4_conf_result = self.c4_confidence.analyze(a1, a2, error, h, m)
            debug_info.append(f"C4 Confidence: {c4_conf_result.reason}")
            debug_info.append(f"C4 Telemetry Trace: {telemetry_log}")
            
            if len1 > 0 and len2 > 0:
                ratio = len1 / len2
                if ratio < 0.9:
                    debug_info.append(f"Heuristics: Hand 1 (Px: {len1:.1f}) shorter than Hand 2 (Px: {len2:.1f}). Rewarded Hypothesis A.")
                elif ratio > 1.1:
                    debug_info.append(f"Heuristics: Hand 1 (Px: {len1:.1f}) longer than Hand 2 (Px: {len2:.1f}). Rewarded Hypothesis B.")
            
            cand_strings = [f"{c['hour']}:{c['minute']:02d} ({c['confidence']:.1f}%)" for c in cands]
            debug_info.append(f"Top 3 Candidates: {', '.join(cand_strings)}")
            
            if ambiguity_warning: debug_info.append(ambiguity_warning)
            debug_info.append(f"Accuracy: {drift_str}")
            
            if error < 40.0 and not force_expert:
                return {
                    "time": f"{h}:{m:02d}",
                    "method": "Fast Path (C1+C2+C4)",
                    "confidence": f"{conf:.1f}%",
                    "heatmap": None,
                    "debug": debug_info,
                    "visualizations": visualizations,
                    "angles": {"hand1": a1, "hand2": a2},
                    "reasoning": f"Physics: H={a1:.1f}°, M={a2:.1f}° → Time={h}:{m:02d}",
                    "error": "",
                    "ampm": None,
                    "drift": drift_str,
                    "ambiguity": ambiguity_warning,
                    "ambiguity_candidates": cands,
                    "c4_confidence": c4_conf_result.to_dict(),
                    "c1_conf": float(conf),
                    "c1_quality": quality,
                    "c1_clock_conf": float(c_conf) if c_conf != -1.0 else 0.0,
                    "c1_gauge_conf": float(g_conf) if g_conf != -1.0 else 0.0,
                    "c1_clock_quality": c_quality,
                    "c1_gauge_quality": g_quality,
                    "c2_research": None
                }
            
            # --- [C3] CLOCK EXPERT PATH ---
            else:
                if self.c3_model is None:
                    return {"time": f"{h}:{m:02d}", "method": "Fast Path (C3 Missing)", "visualizations": visualizations, "angles": {"hand1": a1, "hand2": a2}}

                # ── PHASE 1: Parallel I/O — shadow filter + C2 research + AM/PM ─────
                # All three touch Gemini (shadow: 2 calls, c2r: 1 call) or local models.
                # Running them concurrently cuts Phase 1 from ~4.5 s → ~1.5 s.
                _sf_candidates = [kpts[i] for i in range(1, min(3, len(kpts)))]

                def _shadow_work():
                    sr = self.c2_shadow_filter.filter_keypoints(target_crop, center, _sf_candidates)
                    sv = self.c2_shadow_filter.render_validation_image(target_crop, center, sr)
                    return sr, sv

                def _c2r_work():
                    # shadow_results=[]: patched below after shadow_work completes
                    return self.c2_research.analyze(target_crop, kpts, detected_type='clock',
                                                    shadow_results=[])

                with ThreadPoolExecutor(max_workers=3) as _p1:
                    _f_shadow = _p1.submit(_shadow_work)
                    _f_ampm   = _p1.submit(self._infer_ampm, img_array)
                    _f_c2r    = _p1.submit(_c2r_work)

                shadow_results = []
                try:
                    shadow_results, shadow_viz = _f_shadow.result()
                    visualizations['c2_shadow'] = shadow_viz
                    debug_info.append(f"Shadow Filter: {sum(1 for r in shadow_results if r.accepted)}/{len(shadow_results)} accepted")
                except Exception as e:
                    debug_info.append(f"Shadow Filter Error: {e}")

                ampm_status, ampm_conf = _f_ampm.result()
                debug_info.append(f"AM/PM: {ampm_status} (Conf: {ampm_conf:.2f})")

                c2_research_data = None
                try:
                    c2_research_data = _f_c2r.result()
                    if c2_research_data is not None and shadow_results:
                        c2_research_data['shadow_filter'] = self.c2_research._shadow_filter_data(shadow_results)
                except Exception as e:
                    debug_info.append(f"C2 Research Error: {e}")

                refined_angles = []
                heatmaps = []
                c3_crops = []
                xai_explanations = []   # structured XAI results per hand
                per_hand_xai = []       # per-hand uncertainty + alpha data
                afs_scores = []         # ROAR Attribution Fidelity Scores
                raw_cams = []           # stored for contrastive XAI reuse

                # Executor for Gemini router calls — dispatched async during the hand loop
                # so both hands' Gemini I/O overlaps with the other hand's CPU work.
                _router_pool = ThreadPoolExecutor(max_workers=2)
                _router_pending = []   # (hand_idx, future_or_None, source, reason, entropy, local_exp)

                for i, (tip, rough_angle) in enumerate(zip([tip1, tip2], [a1, a2])):
                    crop = self._get_crop(target_crop, center, rough_angle)
                    if crop is None or crop.size == 0:
                        refined_angles.append(rough_angle)
                        raw_cams.append(None)
                        continue
                    c3_crops.append(crop)

                    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    pil_resized = pil_crop.resize((64, 64))
                    t_input = self.c3_transform(pil_resized).unsqueeze(0).to(self.device)
                    norm_crop = np.array(pil_resized, dtype=np.float32) / 255.0

                    # L1 — GradCAM++ multi-layer fusion → (vis_overlay, raw_cam)
                    hand_heatmap_vis, raw_cam = self.xai.generate(t_input, norm_crop)
                    raw_cams.append(raw_cam)

                    # L1 annotation — draw quadrant box over peak activation
                    if raw_cam is not None and self.local_explainer:
                        hand_heatmap_vis = self.local_explainer.annotate_quadrant(
                            hand_heatmap_vis, raw_cam
                        )
                    heatmaps.append(hand_heatmap_vis)

                    # L1.5 — Integrated Gradients (theoretically rigorous regression XAI)
                    if self.ig_explainer and force_expert:
                        ig_vis, ig_raw = self.ig_explainer.explain(
                            self.c3_model, t_input, norm_crop
                        )
                        if ig_vis is not None:
                            visualizations[f'xai_ig_h{i+1}'] = ig_vis

                    # MC Dropout — 4 stochastic passes (halved from 8; negligible accuracy loss)
                    c3_angle, uncertainty_std_raw = self._predict_with_uncertainty(t_input, n_passes=4)
                    # Temperature-scaled uncertainty (calibrated proxy)
                    uncertainty_std = self._apply_temperature_scaling(uncertainty_std_raw)
                    delta = c3_angle - 360 if c3_angle > 180 else c3_angle

                    # ROAR — Attribution Fidelity Score (causal heatmap check)
                    afs_result = {"afs": 0.0, "delta_deg": 0.0}
                    if self.roar_scorer and raw_cam is not None and force_expert:
                        try:
                            afs_result = self.roar_scorer.score(
                                self.c3_model, t_input, raw_cam, c3_angle
                            )
                            debug_info.append(
                                f"ROAR AFS H{i+1}: {afs_result['afs']:.3f} "
                                f"(maskedΔ={afs_result['delta_deg']:.1f}°, "
                                f"top-{afs_result['mask_pct']}% blanked)"
                            )
                        except Exception as _e:
                            debug_info.append(f"ROAR H{i+1} error: {_e}")
                    afs_scores.append(afs_result)

                    # L4 — AdaptiveSemanticRouter: dispatch Gemini async so hand-2 CPU
                    # overlaps with hand-1's Gemini I/O instead of blocking serially.
                    entropy_val = compute_entropy(raw_cam) if raw_cam is not None else 0.0
                    if self.adaptive_router:
                        _thresh = AdaptiveSemanticRouter.ENTROPY_THRESHOLD
                        if entropy_val >= _thresh and self.adaptive_router.explainer.available:
                            _reason = f"entropy={entropy_val:.3f} ≥ {_thresh} → Gemini Vision"
                            _f = _router_pool.submit(
                                self.adaptive_router.explainer.explain,
                                crop, hand_heatmap_vis, c3_angle,
                                hand_type=f"Hand {i+1}", raw_heatmap=raw_cam, use_gemini=True
                            )
                            _router_pending.append((i, _f, "Gemini", _reason, entropy_val, None))
                        else:
                            _reason = f"entropy={entropy_val:.3f} < {_thresh} → LocalExplainer"
                            _local = self.adaptive_router.local.explain(raw_cam, c3_angle, f"Hand {i+1}")
                            _router_pending.append((i, None, "Local", _reason, entropy_val, _local))

                    debug_info.append(
                        f"Hand {i+1}: C3 angle={c3_angle:.1f}°, delta={delta:+.1f}°, "
                        f"uncertainty=±{uncertainty_std:.1f}°, entropy={entropy_val:.3f}"
                    )

                    # L6 — LIME superpixel overlay (50 samples — 4× faster than 200)
                    if self.lime_explainer and self.lime_explainer.available and force_expert:
                        try:
                            lime_overlay = self.lime_explainer.explain(
                                self.c3_model, t_input, norm_crop, n_samples=50
                            )
                            if lime_overlay is not None:
                                visualizations[f'xai_lime_h{i+1}'] = lime_overlay
                        except Exception as _e:
                            debug_info.append(f"LIME H{i+1} error: {_e}")

                    # L7 — SHAP DeepExplainer attribution
                    if self.shap_explainer and self.shap_explainer.available and force_expert:
                        try:
                            bg = torch.zeros((5, 3, 64, 64), device=self.device)
                            shap_overlay = self.shap_explainer.explain(
                                self.c3_model, t_input, bg
                            )
                            if shap_overlay is not None:
                                visualizations[f'xai_shap_h{i+1}'] = shap_overlay
                        except Exception as _e:
                            debug_info.append(f"SHAP H{i+1} error: {_e}")

                    # Adaptive blend α = clip(1 − σ/20, 0, 1)
                    alpha = float(np.clip(1.0 - (uncertainty_std / 20.0), 0.0, 1.0))

                    # Store per-hand XAI data for dedicated UI panels
                    per_hand_xai.append({
                        "hand": i + 1,
                        "c3_angle": round(c3_angle, 2),
                        "rough_angle": round(rough_angle, 2),
                        "delta": round(delta, 2),
                        "uncertainty_std": round(uncertainty_std, 2),
                        "uncertainty_raw": round(uncertainty_std_raw, 2),
                        "temperature": round(self._uncertainty_temperature, 3),
                        "alpha": round(alpha, 3),
                        "entropy": round(entropy_val, 3),
                    })

                    if abs(delta) > 20.0:
                        debug_info.append(
                            f"Hand {i+1}: C3 delta exceeds 20° cap — keeping C2 angle "
                            f"(α={alpha:.2f}, σ=±{uncertainty_std:.1f}°)"
                        )
                        refined_angles.append(rough_angle)
                    else:
                        blended = (alpha * (rough_angle + delta) + (1.0 - alpha) * rough_angle) % 360
                        # C3 Kalman smoother — applied on the C3-corrected output
                        smoothed = self._smooth_c3_angle(i, blended)
                        kalman_delta = abs(blended - smoothed)
                        if kalman_delta > 1.0:
                            debug_info.append(
                                f"Hand {i+1}: Kalman smoother: {blended:.1f}° → {smoothed:.1f}° "
                                f"(Δ={kalman_delta:.1f}°)"
                            )
                        debug_info.append(
                            f"Hand {i+1}: α={alpha:.2f} → blended {rough_angle:.1f}° + {delta:+.1f}° = {smoothed:.1f}°"
                        )
                        refined_angles.append(smoothed)

                # Collect async router results (Gemini calls ran during hand-2 CPU work)
                _router_pool.shutdown(wait=True)
                for (_ri, _rf, _rsrc, _rreason, _rentropy, _rlocal) in _router_pending:
                    if _rf is not None:
                        try:
                            _explanation = _rf.result(timeout=15)
                        except Exception as _re:
                            _explanation = f"[XAI unavailable: {_re}]"
                            _rsrc = "Local (timeout)"
                    else:
                        _explanation = _rlocal
                    xai_explanations.append({
                        "hand": _ri + 1,
                        "source": _rsrc,
                        "explanation": _explanation,
                        "routing_reason": _rreason,
                        "entropy": round(_rentropy, 3),
                    })
                    debug_info.append(f"XAI Hand {_ri+1} [{_rsrc}]: {_explanation}")
                    debug_info.append(f"XAI Routing (H{_ri+1}): {_rreason}")

                if len(heatmaps) == 2:
                    heatmap_img = np.hstack((heatmaps[0], heatmaps[1]))
                elif len(heatmaps) == 1:
                    heatmap_img = heatmaps[0]
                else:
                    heatmap_img = None

                visualizations['c3_crops'] = c3_crops
                visualizations['c3_angles'] = self._draw_angles_on_img(target_crop, center, tip1, tip2, refined_angles[0], refined_angles[1])

                # Build uncertainty summary for the result payload
                unc_vals = getattr(self, '_last_uncertainties', [])
                uncertainty_summary = ", ".join(
                    [f"H{j+1}=±{unc_vals[j]:.1f}°" for j in range(len(unc_vals))]
                ) if unc_vals else "N/A"
                self._last_uncertainties = []  # Reset for next call

                h_new, m_new, err_new, conf_new, cands_new, telemetry_log_new = self._solve_physics(
                    refined_angles[0], refined_angles[1], length1=len1, length2=len2
                )

                ambiguity_warning_expert = self._resolve_ambiguity(refined_angles[0], refined_angles[1], h_new, m_new)

                # L5 — Contrastive XAI ("Why not X:XX?") — reuse stored raw_cam (no recompute)
                contrastive_text = ""
                if self.contrastive_xai and len(heatmaps) > 0 and force_expert:
                    _raw_cam_h1 = raw_cams[0] if raw_cams else None
                    if _raw_cam_h1 is not None:
                        try:
                            contrastive_text = self.contrastive_xai.explain(
                                _raw_cam_h1, h_new, m_new, cands_new
                            )
                            debug_info.append(contrastive_text)
                        except Exception as _e:
                            debug_info.append(f"Contrastive XAI error: {_e}")

                debug_info.append(f"C4 Telemetry Trace: {telemetry_log_new}")
                cand_strings_new = [f"{c['hour']}:{c['minute']:02d} ({c['confidence']:.1f}%)" for c in cands_new]
                debug_info.append(f"Expert Top 3 Candidates: {', '.join(cand_strings_new)}")
                
                if ambiguity_warning_expert and not ambiguity_warning:
                    debug_info.append(ambiguity_warning_expert)
                drift_str_expert = self._calculate_drift(h_new, m_new, device_time_str)

                # [C4+] Physics Confidence Assessment (Expert Path)
                c4_conf_result_new = self.c4_confidence.analyze(refined_angles[0], refined_angles[1], err_new, h_new, m_new)
                debug_info.append(f"C4 Confidence (Expert): {c4_conf_result_new.reason}")
                
                return {
                    "time": f"{h_new}:{m_new:02d}",
                    "method": "Expert Path (C1+C2+C3+C4+MC-Dropout+XAI-7)",
                    "confidence": f"{conf_new:.1f}%",
                    "heatmap": heatmap_img,
                    "debug": debug_info,
                    "visualizations": visualizations,
                    "angles": {"hand1": refined_angles[0], "hand2": refined_angles[1]},
                    "reasoning": f"Refined: H={refined_angles[0]:.1f}°, M={refined_angles[1]:.1f}° → Time={h_new}:{m_new:02d}",
                    "error": "",
                    "ampm": ampm_status,
                    "drift": drift_str_expert,
                    "ambiguity": ambiguity_warning_expert or ambiguity_warning,
                    "ambiguity_candidates": cands_new,
                    "c4_confidence": c4_conf_result_new.to_dict(),
                    "c1_conf": float(conf),
                    "c1_quality": quality,
                    "c1_clock_conf": float(c_conf) if c_conf != -1.0 else 0.0,
                    "c1_gauge_conf": float(g_conf) if g_conf != -1.0 else 0.0,
                    "c1_clock_quality": c_quality,
                    "c1_gauge_quality": g_quality,
                    "c2_research": c2_research_data,
                    "uncertainty_deg": uncertainty_summary,
                    "xai_method": "GradCAM++ (L2+L3+L4 multi-layer) + IG + MC-Dropout",
                    "contrastive_xai": contrastive_text,
                    "xai_explanations": xai_explanations,   # L2/L3 per-hand text
                    "per_hand_xai": per_hand_xai,           # uncertainty + alpha per hand
                    "afs_scores": afs_scores,               # ROAR fidelity per hand
                }

# Alias mapping so you don't break existing `main.py` imports
ClockEngine = HARPEngine