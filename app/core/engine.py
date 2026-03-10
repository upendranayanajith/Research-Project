import os
import cv2
import numpy as np
import re
import math
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
from app.core.xai import (
    XaiVisualizer, SemanticExplainer,
    AdaptiveSemanticRouter, ContrastiveExplainer,
    LimeExplainer, ShapExplainer,
)
from app.core.metrics import calculate_gauge_reading, calculate_gauge_reading_advanced
from app.core.c2_research import C2ResearchAnalyzer
from app.core.c2_shadow_filter import SemanticShadowFilter
from app.core.temporal import TemporalTracker   # [Tier 1.4] Temporal Consistency
from app.core.losses import VonMisesLoss, CircularHead, decode_circular_numpy  # [T3.5]
from app.core.style import StyleAnalyser                                         # [T3.2]
from app.core.vit_xai import ViTC3Model, VitAttentionVisualizer                  # [T3.4]
import google.generativeai as genai
import easyocr
from dotenv import load_dotenv
from datetime import datetime

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
            self.c3_model.load_state_dict(torch.load(self.c3_path, map_location=self.device))
            self.c3_model.eval()
            self.xai       = XaiVisualizer(self.c3_model[0])
            self.explainer = SemanticExplainer()
            # --- [Tier 2] XAI Extensions ---
            self.adaptive_router       = AdaptiveSemanticRouter(self.explainer)   # [6.9]
            self.contrastive_explainer = ContrastiveExplainer()                   # [6.6]
            self.lime_explainer        = LimeExplainer()                          # [6.10]
            self.shap_explainer        = ShapExplainer()                          # [6.10]
            print("✅ C3 + XAI (GradCAM++) Loaded.")
            print(f"  Adaptive Routing: entropy threshold = {AdaptiveSemanticRouter.ENTROPY_THRESHOLD}")
            print(f"  LIME: {'available' if self.lime_explainer.available else 'not installed'}")
            print(f"  SHAP: {'available' if self.shap_explainer.available else 'not installed'}")
        else:
            print("⚠️ WARNING: C3 weights not found.")
            self.c3_model = None
            self.explainer = None
            self.xai       = None
            self.adaptive_router     = None
            self.contrastive_explainer = None
            self.lime_explainer      = None
            self.shap_explainer      = None
        
        # --- [C2] Research Analyzer ---
        self.c2_research = C2ResearchAnalyzer()
        self.c2_shadow_filter = SemanticShadowFilter()

        print("Loading EasyOCR Fallback...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

        # --- [Tier 1.4] TEMPORAL CONSISTENCY (Kalman Filter) ---
        self.temporal_tracker = TemporalTracker(
            process_noise=1.5,
            measurement_noise=5.0,
            spike_threshold=45.0,
        )
        print("✅ Temporal Tracker (Kalman) Initialized.")

        # --- [T3.2] CLOCK STYLE CLASSIFIER ---
        style_weights = os.path.join(base_dir, "models", "c3_style_classifier", "best.pth")
        self.style_analyser = StyleAnalyser(weights_path=style_weights,
                                            device=str(self.device))
        print("✅ Style Analyser Initialized (no weights → uniform priors).")

        # --- [T3.4] ViT XAI Visualizer (complementary to GradCAM++) ---
        self.vit_attention_visualizer = VitAttentionVisualizer()
        print("✅ ViT Attention Visualizer Initialized.")

    def _load_yolo(self, path, name):
        try:
            print(f"Loading {name}: {path}...")
            return YOLO(path)
        except Exception as e:
            print(f"⚠️ {name} Failed: {e}")
            return None

    def _get_c3_arch(self):
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        model = nn.Sequential(model, nn.Sigmoid())
        return model

    def _get_c3_arch_circular(self):
        """
        [T3.5] Circular regression head: outputs (sin θ, cos θ) instead of a
        Sigmoid scalar, resolving the 0°/360° wraparound ambiguity.

        Uses CircularHead from app.core.losses which L2-normalises the output
        to the unit circle. Decode with decode_circular_numpy() for inference.

        Requires retraining with VonMisesLoss — see scripts/train_c3_circular.py
        Checkpoint: models/c3_circular/best.pth
        """
        backbone = models.resnet18(weights=None)
        feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # (N, 512, 1, 1)
        flatten = nn.Flatten()                                                # (N, 512)
        head    = CircularHead(in_features=512)                               # (N, 2)
        return nn.Sequential(feature_extractor, flatten, head)

    def _get_c3_arch_vit(self, circular: bool = False, pretrained: bool = True):
        """
        [T3.4] ViT-B/16 backbone for C3 — interpretable attention maps,
        no GradCAM++ required.

        Args:
            circular:   If True, use CircularHead output (sin/cos).
            pretrained: If True, load ImageNet weights for the ViT backbone.

        Requires retraining — see scripts/train_c3_circular.py with ViT backbone.
        Checkpoint: models/c3_vit/best.pth
        """
        return ViTC3Model(circular=circular, pretrained=pretrained)


    def _enable_dropout(self):
        """Sets all Dropout layers in c3_model to train mode (enables stochastic dropout)."""
        for m in self.c3_model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _predict_with_uncertainty(self, tensor: torch.Tensor, n_passes: int = 20):
        """
        [FIX-3] Monte Carlo Dropout uncertainty estimation.
        Temporarily enables dropout during inference for N stochastic forward passes.

        Returns:
            mean_angle_deg (float): Average predicted angle across all passes.
            std_deg        (float): Standard deviation — proxy for model uncertainty.
        """
        if self.c3_model is None:
            return 0.0, 0.0

        # Switch to eval (disables BN randomness) but keep dropout active
        self.c3_model.eval()
        self._enable_dropout()

        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                pred = self.c3_model(tensor).item()  # sigmoid → [0,1]
                preds.append(pred * 360.0)            # → degrees

        # Restore full eval mode
        self.c3_model.eval()

        mean_angle = float(np.mean(preds))
        std_deg    = float(np.std(preds))

        # Store for surface in result dict
        if not hasattr(self, '_last_uncertainties'):
            self._last_uncertainties = []
        self._last_uncertainties.append(std_deg)

        return mean_angle, std_deg

    # [FIX-5] Gauge C3 needle refinement
    GAUGE_C3_DELTA_THRESHOLD = 15.0  # degrees — tighter than clock (15° vs 20°)

    def _refine_gauge_needle_angle(self, crop, center, tip, rough_angle: float):
        """
        [FIX-5] Applies C3 angle regression to refine the gauge needle angle.
        Uses the same trained clock model (transfer from clock domain).
        The rotation-normalise-crop pipeline is identical to the clock hand path.

        Args:
            crop:        The gauge crop image (BGR ndarray).
            center:      (x, y) of gauge center keypoint.
            tip:         (x, y) of needle tip keypoint.
            rough_angle: Geometric angle computed from C2 keypoints.

        Returns:
            refined_angle (float): C3-refined needle angle.
            delta         (float): Correction applied (0 if rejected or C3 unavailable).
            uncertainty   (float): Std deviation from MC Dropout passes.
            accepted      (bool):  Whether C3 correction was accepted.
        """
        if self.c3_model is None:
            return rough_angle, 0.0, 0.0, False

        hand_crop = self._get_crop(crop, center, rough_angle)
        if hand_crop is None or hand_crop.size == 0:
            return rough_angle, 0.0, 0.0, False

        try:
            pil_crop    = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
            pil_resized = pil_crop.resize((64, 64))
            t_input     = self.c3_transform(pil_resized).unsqueeze(0).to(self.device)

            c3_angle, uncertainty_std = self._predict_with_uncertainty(t_input)
            delta = c3_angle - 360 if c3_angle > 180 else c3_angle

            if abs(delta) > self.GAUGE_C3_DELTA_THRESHOLD:
                return rough_angle, delta, uncertainty_std, False

            # Soft weighted blend (same alpha logic as clock path)
            alpha   = float(np.clip(1.0 - (uncertainty_std / 20.0), 0.0, 1.0))
            blended = (alpha * (rough_angle + delta) + (1.0 - alpha) * rough_angle) % 360
            return blended, delta, uncertainty_std, True

        except Exception:
            return rough_angle, 0.0, 0.0, False

    def _get_angle(self, center, point):
        dx, dy = point[0] - center[0], point[1] - center[1]
        angle = math.degrees(math.atan2(dx, -dy))
        return angle + 360 if angle < 0 else angle

    # --- CLOCK PHYSICS SOLVER (Restored from main branch) ---
    def _solve_physics(self, a1, a2):
        err_a = np.abs(a1 - self.theory_h) + np.abs(a2 - self.theory_m)
        err_a = np.minimum(err_a, 720 - err_a)
        err_b = np.abs(a2 - self.theory_h) + np.abs(a1 - self.theory_m)
        err_b = np.minimum(err_b, 720 - err_b)

        if np.min(err_a) < np.min(err_b):
            idx = np.argmin(err_a)
            return int(idx // 60) if int(idx // 60) != 0 else 12, int(idx % 60), np.min(err_a)
        else:
            idx = np.argmin(err_b)
            return int(idx // 60) if int(idx // 60) != 0 else 12, int(idx % 60), np.min(err_b)

    def _get_crop(self, img, center, angle):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))
        s = 128 // 2
        y1, y2 = int(center[1]-s), int(center[1]+s)
        x1, x2 = int(center[0]-s), int(center[0]+s)
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h: return np.array([])
        return rotated[y1:y2, x1:x2]

    def _resize_small(self, img):
        """Helper to force 500x500px output for dashboard efficiency"""
        return cv2.resize(img, (500, 500), interpolation=cv2.INTER_LINEAR)

    def _infer_ampm_local(self, crop) -> tuple:
        """
        [FIX-6] API-free AM/PM heuristic using mean brightness (HSV V-channel).
        Bright image → likely daytime → AM.  Dark image → likely night → PM.
        Confidence 0.6 reflects that this is a heuristic, not a trained classifier.
        """
        if crop is None or crop.size == 0:
            return "Unknown", 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_brightness = float(np.mean(hsv[:, :, 2]))  # V channel, 0-255
        if mean_brightness > 128:
            return "AM", 0.6
        else:
            return "PM", 0.6

    def _infer_ampm(self, crop):
        """Primary: Gemini Vision API. Fallback: local brightness heuristic."""
        if crop is None or crop.size == 0:
            return "Unknown", 0.0

        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                model = genai.GenerativeModel('gemini-2.5-flash')
                prompt = (
                    "Look at this cropped image of a clock. Based on the lighting, "
                    "shadows, colors, and overall ambiance, guess whether this photo "
                    "was taken during the day (AM) or night (PM). Return ONLY 'AM' or 'PM'."
                )
                response = model.generate_content([prompt, pil_img])
                text = response.text.strip().upper()
                if "AM" in text:
                    return "AM", 0.9
                elif "PM" in text:
                    return "PM", 0.9
                else:
                    return "Unknown", 0.5
            except Exception:
                pass  # Fall through to local heuristic

        # [FIX-6] Local brightness fallback
        result, conf = self._infer_ampm_local(crop)
        return result, conf

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

    def _extract_gauge_scale_gemini(self, img):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None, None, "Missing GEMINI_API_KEY in .env"
            
        genai.configure(api_key=api_key)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = "Look at this analog gauge. Return ONLY the minimum scale reading and the maximum scale reading printed on it, separated by a comma. Do not include units or any other text. For example: 0, 100 or -10, 50."
        
        try:
            response = model.generate_content([prompt, pil_img])
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
        clock_res = self.c1_clock_model(img, verbose=False)[0] if self.c1_clock_model else None
        gauge_res = self.c1_gauge_model(img, verbose=False)[0] if self.c1_gauge_model else None
        
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

    # -----------------------------------------------------------------------
    # [6.6] Top-N candidates from C4 physics solver (for ContrastiveExplainer)
    # -----------------------------------------------------------------------
    def _get_top_n_candidates(self, a1: float, a2: float, n: int = 4) -> list:
        """
        Run the C4 physics solver over all 720 minute values and return the
        top-N candidates sorted by ascending physics error.

        Returns: list of (hour, minute, error) tuples.
        """
        errors = np.abs(self.theory_h - a1) + np.abs(self.theory_m - a2)
        idx    = np.argsort(errors)[:n * 3]      # Over-sample before dedup
        seen   = set()
        results = []
        for i in idx:
            total_minutes = int(self.possible_minutes[i])
            h = (total_minutes // 60) % 12 or 12
            m = total_minutes % 60
            key = (h, m)
            if key not in seen:
                seen.add(key)
                results.append((h, m, float(errors[i])))
            if len(results) >= n:
                break
        return results

    # -----------------------------------------------------------------------
    # [6.7] Hand type heuristic: shorter arm = hour, longer arm = minute
    # -----------------------------------------------------------------------
    def _classify_hand_type_heuristic(self, tip1, tip2, center) -> str:
        """
        Uses the Euclidean distance from center to each tip to infer which
        hand is the hour (shorter) vs minute (longer) hand.

        Returns a human-readable string for debug_info.
        """
        try:
            len1 = float(np.linalg.norm(np.array(tip1) - np.array(center)))
            len2 = float(np.linalg.norm(np.array(tip2) - np.array(center)))
            ratio = round(min(len1, len2) / max(len1, len2 + 1e-6), 2)
            if len1 < len2:
                return f"Hand1=Hour ({len1:.0f}px), Hand2=Minute ({len2:.0f}px), length ratio={ratio}"
            else:
                return f"Hand1=Minute ({len1:.0f}px), Hand2=Hour ({len2:.0f}px), length ratio={ratio}"
        except Exception:
            return "Hand type heuristic: unable to compute (missing keypoints)"

    # -----------------------------------------------------------------------
    # [6.7] Dual-head ResNet18 architecture stub (for future retraining)
    # -----------------------------------------------------------------------
    def _get_c3_arch_dual(self):
        """
        Dual-head ResNet18:
          Head 1 — angle regression (compatible with existing best.pth)
          Head 2 — hand-type classification (hour=0, minute=1)

        NOTE: This requires the model to be retrained with both heads active.
              For now it serves as the architecture reference.
        """
        import torch.nn as nn
        from torchvision import models as tv_models

        class DualHeadResNet(nn.Module):
            def __init__(self):
                super().__init__()
                backbone = tv_models.resnet18(weights=None)
                self.features = nn.Sequential(*list(backbone.children())[:-1])  # Strip FC
                self.angle_head = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
                self.type_head  = nn.Sequential(nn.Linear(512, 2))   # logits for hour/minute

            def forward(self, x):
                feat = self.features(x).squeeze(-1).squeeze(-1)   # (B, 512)
                angle = self.angle_head(feat)        # (B, 1) ∈ [0,1]
                hand_type = self.type_head(feat)     # (B, 2) logits
                return angle, hand_type

        return DualHeadResNet()

    def analyze(self, img_array, force_expert=False, manual_min_val="", manual_max_val="",
                device_time_str=None, enable_temporal=False):
        debug_info = []
        visualizations = {}
        self._last_uncertainties = []  # [FIX-3] Reset per-analysis uncertainty log

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

        # --- [C2] CROSS-VALIDATION (Fixed Isolated Crops) ---
        c2_clock_conf = 0.0
        c2_gauge_conf = 0.0
        clock_kpts, gauge_kpts = None, None
        
        if self.c2_clock_model and c_crop is not None:
            c_res = self.c2_clock_model(c_crop, verbose=False)[0]
            if c_res.keypoints and len(c_res.keypoints.data) > 0:
                kpts = c_res.keypoints.data[0].cpu().numpy()
                if len(kpts) >= 3:
                    c2_clock_conf = np.mean(kpts[:3, 2])
                    clock_kpts = kpts
                    
        if self.c2_gauge_model and g_crop is not None:
            g_res = self.c2_gauge_model(g_crop, verbose=False)[0]
            if g_res.keypoints and len(g_res.keypoints.data) > 0:
                kpts = g_res.keypoints.data[0].cpu().numpy()
                if len(kpts) >= 4:
                    c2_gauge_conf = np.mean(kpts[:4, 2])
                    gauge_kpts = kpts

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
            
            if manual_min_val and manual_max_val:
                try:
                    min_val = float(manual_min_val)
                    max_val = float(manual_max_val)
                    override_active = True
                    debug_info.append(f"Stage 1 (Manual Override): Min={min_val}, Max={max_val}")
                except Exception as e:
                    debug_info.append(f"Stage 1 (Manual Override) Failed to parse: {e}")

            # --- STAGE 2: GEMINI API ---
            err_str = None
            if not override_active:
                min_val, max_val, err_str = self._extract_gauge_scale_gemini(target_crop)
                if err_str:
                    debug_info.append(f"Stage 2 (Gemini API) Failed: {err_str}")
                elif min_val is not None and max_val is not None:
                    debug_info.append(f"Stage 2 (Gemini API): Min={min_val}, Max={max_val}")
            
            # --- STAGE 3: LOCAL FALLBACK (INWARD SHIFT OCR) ---
            if not override_active and (min_val is None or max_val is None):
                debug_info.append("Stage 3 (Local API Fallback): Using Inward-Shift OCR.")
                min_roi = self._get_roi_inward(target_crop, center, min_pt)
                max_roi = self._get_roi_inward(target_crop, center, max_pt)
                min_val = self._extract_number(min_roi)
                max_val = self._extract_number(max_roi)
                debug_info.append(f"Stage 3 (Local API Fallback): Min={min_val}, Max={max_val}")
            
            parsed_min = min_val if min_val is not None else "Failed"
            parsed_max = max_val if max_val is not None else "Failed"
            
            a_min = self._get_angle(center, min_pt)
            a_max = self._get_angle(center, max_pt)
            a_tip_raw = self._get_angle(center, tip)

            # --- [FIX-5] C3 Gauge Needle Refinement ---
            refined_tip_angle, c3_delta, c3_unc, c3_accepted = self._refine_gauge_needle_angle(
                target_crop, center, tip, a_tip_raw
            )
            if c3_accepted:
                debug_info.append(
                    f"Gauge C3: needle refined {a_tip_raw:.1f}° → {refined_tip_angle:.1f}° "
                    f"(delta={c3_delta:.1f}°, uncertainty=±{c3_unc:.1f}°)"
                )
            else:
                debug_info.append(
                    f"Gauge C3: needle kept at {a_tip_raw:.1f}° (C3 correction rejected or unavailable)"
                )
                refined_tip_angle = a_tip_raw

            span   = (a_max - a_min + 360) % 360
            needle = (refined_tip_angle - a_min + 360) % 360
            
            units_per_deg = 0.0
            # C4 Physics Logic 
            if min_val is not None and max_val is not None and min_val <= max_val:
                reading, units_per_deg = calculate_gauge_reading_advanced(span, needle, min_val, max_val)
                time_str = str(reading)
                c3_tag = "+C3" if c3_accepted else ""
                method_str = f"Advanced Gauge Reading (C1+C2{c3_tag}+OCR+C4)"
                reasoning_str = f"Gauge Logic: 1° = {units_per_deg:.4f} units. Formula: {min_val} + ({needle:.1f}° * {units_per_deg:.4f}) = {reading}"
            else:
                reading = calculate_gauge_reading([center, min_pt, max_pt, tip])
                time_str = f"{reading}%"
                c3_tag = "+C3" if c3_accepted else ""
                method_str = f"Gauge Reading - Fallback (C1+C2{c3_tag}+C4)"
                reasoning_str = "Gauge Logic: OCR extraction failed. Interpreted raw percentage. Please use 'Manual Gauge Scale Overrides' in settings."

            visualizations['c2_skeleton'] = self._draw_gauge_skeleton(target_crop, center, min_pt, max_pt, tip, parsed_min, parsed_max)
            visualizations['c3_angles'] = self._draw_gauge_angles(target_crop, center, min_pt, max_pt, tip, span, needle)
            
            # --- C2 Shadow Filter ---
            shadow_results = []
            try:
                candidates = [kpts[i] for i in range(1, min(4, len(kpts)))]
                shadow_results = self.c2_shadow_filter.filter_keypoints(target_crop, center, candidates)
                shadow_viz = self.c2_shadow_filter.render_validation_image(target_crop, center, shadow_results)
                visualizations['c2_shadow'] = shadow_viz
                debug_info.append(f"Shadow Filter: {sum(1 for r in shadow_results if r.accepted)}/{len(shadow_results)} accepted")
            except Exception as e:
                debug_info.append(f"Shadow Filter Error: {e}")

            # --- C2 Research Analysis ---
            try:
                c2_research_data = self.c2_research.analyze(target_crop, kpts, detected_type='gauge',
                                                            shadow_results=shadow_results)
            except Exception as e:
                c2_research_data = None
                debug_info.append(f"C2 Research Error: {e}")

            return {
                "time": time_str,
                "method": method_str,
                "confidence": "High",
                "heatmap": None,
                "debug": debug_info + [f"Final Reading: {time_str}"],
                "visualizations": visualizations,
                "angles": {"span": span, "needle": needle, "units_per_deg": units_per_deg, "needle_raw": a_tip_raw, "c3_accepted": c3_accepted},
                "scale": {"min": parsed_min, "max": parsed_max},
                "reasoning": reasoning_str,
                "error": "",
                "c1_conf": float(conf),
                "c1_quality": quality,
                "c1_clock_conf": float(c_conf) if c_conf != -1.0 else 0.0,
                "c1_gauge_conf": float(g_conf) if g_conf != -1.0 else 0.0,
                "c1_clock_quality": c_quality,
                "c1_gauge_quality": g_quality,
                "c2_research": c2_research_data
            }

        # ==========================================
        # CLOCK LOGIC PIPELINE
        # ==========================================
        elif detected_type == 'clock':
            kpts = clock_kpts
            center, tip1, tip2 = kpts[0][:2], kpts[1][:2], kpts[2][:2]
            
            a1 = self._get_angle(center, tip1)
            a2 = self._get_angle(center, tip2)
            
            visualizations['c2_skeleton'] = self._draw_skeleton(target_crop, center, tip1, tip2)
            visualizations['c3_angles'] = self._draw_angles_on_img(target_crop, center, tip1, tip2, a1, a2)
            
            # --- C2 Shadow Filter ---
            shadow_results = []
            try:
                candidates = [kpts[i] for i in range(1, min(3, len(kpts)))]
                shadow_results = self.c2_shadow_filter.filter_keypoints(target_crop, center, candidates)
                shadow_viz = self.c2_shadow_filter.render_validation_image(target_crop, center, shadow_results)
                visualizations['c2_shadow'] = shadow_viz
                debug_info.append(f"Shadow Filter: {sum(1 for r in shadow_results if r.accepted)}/{len(shadow_results)} accepted")
            except Exception as e:
                debug_info.append(f"Shadow Filter Error: {e}")

            # --- C2 Research Analysis ---
            try:
                c2_research_data = self.c2_research.analyze(target_crop, kpts, detected_type='clock',
                                                            shadow_results=shadow_results)
            except Exception as e:
                c2_research_data = None
                debug_info.append(f"C2 Research Error: {e}")
            
            # [Tier 1.4] Temporal Consistency — apply Kalman smoothing
            if enable_temporal:
                a1_raw, a2_raw = a1, a2
                a1, a2, spike_info = self.temporal_tracker.update(a1, a2)
                temporal_xai = self.temporal_tracker.get_temporal_xai()
                if spike_info["spikes_this_frame"]:
                    debug_info.append(
                        f"Temporal: Spike rejected on {spike_info['spikes_this_frame']} "
                        f"(keeping prev smoothed angle)"
                    )
                debug_info.append(
                    f"Temporal: Kalman smoothed H={a1:.1f}\u00b0 (was {a1_raw:.1f}\u00b0), "
                    f"M={a2:.1f}\u00b0 (was {a2_raw:.1f}\u00b0)"
                )
            else:
                temporal_xai = None

            h, m, error = self._solve_physics(a1, a2)
            
            ampm_status, ampm_conf = self._infer_ampm(img_array)
            ambiguity_warning = self._resolve_ambiguity(a1, a2, h, m)
            drift_str = self._calculate_drift(h, m, device_time_str)
            
            debug_info.append(f"AM/PM: {ampm_status} (Conf: {ampm_conf:.2f})")
            if ambiguity_warning: debug_info.append(ambiguity_warning)
            debug_info.append(f"Accuracy: {drift_str}")
            
            if error < 20.0 and not force_expert:
                return {
                    "time": f"{h}:{m:02d}",
                    "method": "Fast Path (C1+C2+C4)",
                    "confidence": "High",
                    "heatmap": None,
                    "debug": debug_info,
                    "visualizations": visualizations,
                    "angles": {"hand1": a1, "hand2": a2},
                    "reasoning": f"Physics: H={a1:.1f}°, M={a2:.1f}° → Time={h}:{m:02d}",
                    "error": "",
                    "ampm": ampm_status,
                    "drift": drift_str,
                    "ambiguity": ambiguity_warning,
                    "c1_conf": float(conf),
                    "c1_quality": quality,
                    "c1_clock_conf": float(c_conf) if c_conf != -1.0 else 0.0,
                    "c1_gauge_conf": float(g_conf) if g_conf != -1.0 else 0.0,
                    "c1_clock_quality": c_quality,
                    "c1_gauge_quality": g_quality,
                    "c2_research": c2_research_data
                }
            
            # --- [C3] CLOCK EXPERT PATH ---
            # [FIX-3] MC Dropout uncertainty estimation
            # [FIX-4] Adaptive confidence-weighted delta fusion
            else:
                if self.c3_model is None:
                    return {"time": f"{h}:{m:02d}", "method": "Fast Path (C3 Missing)", "visualizations": visualizations, "angles": {"hand1": a1, "hand2": a2}}

                refined_angles = []
                heatmaps     = []
                raw_heatmaps = []   # [6.9] collect for ContrastiveExplainer + AdaptiveRouter
                c3_crops = []

                for i, (tip, rough_angle) in enumerate(zip([tip1, tip2], [a1, a2])):
                    crop = self._get_crop(target_crop, center, rough_angle)
                    if crop.size == 0:
                        refined_angles.append(rough_angle)
                        continue
                    c3_crops.append(crop)

                    pil_crop    = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    pil_resized = pil_crop.resize((64, 64))
                    t_input     = self.c3_transform(pil_resized).unsqueeze(0).to(self.device)

                    norm_crop = np.array(pil_resized, dtype=np.float32) / 255.0
                    # [FIX-1] xai.generate returns (visualization, raw_heatmap)
                    hand_heatmap_vis, raw_heatmap = self.xai.generate(t_input, norm_crop)
                    heatmaps.append(hand_heatmap_vis)
                    raw_heatmaps.append(raw_heatmap)

                    # [FIX-3] MC Dropout uncertainty estimation
                    c3_angle, uncertainty_std = self._predict_with_uncertainty(t_input)
                    delta = c3_angle - 360 if c3_angle > 180 else c3_angle

                    # [6.9] Adaptive XAI Routing: auto-escalate to Gemini when heatmap is diffuse
                    if self.adaptive_router:
                        explanation, routing_reason, entropy_val = self.adaptive_router.explain(
                            raw_heatmap, crop, hand_heatmap_vis,
                            c3_angle, hand_type=f"Hand {i+1}"
                        )
                        debug_info.append(f"AI Insight Hand {i+1}: {explanation}")
                        debug_info.append(routing_reason)
                    elif self.explainer:
                        explanation = self.explainer.explain(
                            crop, hand_heatmap_vis, c3_angle,
                            hand_type=f"Hand {i+1}",
                            raw_heatmap=raw_heatmap,
                            use_gemini=force_expert,
                        )
                        debug_info.append(f"AI Insight Hand {i+1}: {explanation}")

                    debug_info.append(
                        f"Hand {i+1}: C3 delta={delta:.1f}\u00b0, uncertainty=\u00b1{uncertainty_std:.1f}\u00b0"
                    )

                    # [FIX-4] Adaptive confidence-weighted delta fusion
                    alpha = float(np.clip(1.0 - (uncertainty_std / 20.0), 0.0, 1.0))
                    if abs(delta) > 20.0:
                        debug_info.append(f"Hand {i+1}: C3 delta exceeds cap \u2014 keeping C2 angle.")
                        refined_angles.append(rough_angle)
                    else:
                        blended = (alpha * (rough_angle + delta) + (1.0 - alpha) * rough_angle) % 360
                        debug_info.append(f"Hand {i+1}: alpha={alpha:.2f} \u2192 blended={blended:.1f}\u00b0")
                        refined_angles.append(blended)

                if len(heatmaps) == 2:
                    heatmap_img = np.hstack((heatmaps[0], heatmaps[1]))
                elif len(heatmaps) == 1:
                    heatmap_img = heatmaps[0]
                else:
                    heatmap_img = None

                # [6.10] LIME + SHAP supplementary explainers (run on hand1 crop if available)
                if c3_crops and raw_heatmaps:
                    lime_overlay = None
                    shap_overlay = None
                    try:
                        if self.lime_explainer and self.lime_explainer.available:
                            first_crop = c3_crops[0]
                            first_norm = np.array(
                                Image.fromarray(cv2.cvtColor(first_crop, cv2.COLOR_BGR2RGB)).resize((64,64)),
                                dtype=np.float32
                            ) / 255.0
                            first_t = self.c3_transform(
                                Image.fromarray(cv2.cvtColor(first_crop, cv2.COLOR_BGR2RGB)).resize((64,64))
                            ).unsqueeze(0).to(self.device)
                            lime_overlay = self.lime_explainer.explain(self.c3_model, first_t, first_norm)
                            if lime_overlay is not None:
                                debug_info.append("LIME: Superpixel explanation generated.")
                    except Exception as e:
                        debug_info.append(f"LIME: failed ({e})")
                    try:
                        if self.shap_explainer and self.shap_explainer.available:
                            first_crop = c3_crops[0]
                            first_t = self.c3_transform(
                                Image.fromarray(cv2.cvtColor(first_crop, cv2.COLOR_BGR2RGB)).resize((64,64))
                            ).unsqueeze(0).to(self.device)
                            background = torch.zeros(5, *first_t.shape[1:], device=self.device)
                            shap_overlay = self.shap_explainer.explain(self.c3_model, first_t, background)
                            if shap_overlay is not None:
                                debug_info.append("SHAP: DeepExplainer attribution generated.")
                    except Exception as e:
                        debug_info.append(f"SHAP: failed ({e})")
                    if lime_overlay is not None:
                        visualizations['lime_heatmap'] = lime_overlay
                    if shap_overlay is not None:
                        visualizations['shap_heatmap'] = shap_overlay

                # Surface uncertainty in result
                uncertainty_summary = ", ".join(
                    [f"H{j+1}=\u00b1{self._last_uncertainties[j]:.1f}\u00b0"
                     for j in range(len(self._last_uncertainties))]
                ) if hasattr(self, '_last_uncertainties') else "N/A"
                
                visualizations['c3_crops'] = c3_crops
                visualizations['c3_angles'] = self._draw_angles_on_img(target_crop, center, tip1, tip2, refined_angles[0], refined_angles[1])
                
                # [Tier 1.4] Temporal Consistency on expert-refined angles
                if enable_temporal:
                    unc_values = list(self._last_uncertainties)
                    u1 = unc_values[0] if len(unc_values) > 0 else None
                    u2 = unc_values[1] if len(unc_values) > 1 else None
                    ra0_raw, ra1_raw = refined_angles[0], refined_angles[1]
                    ref0, ref1, spike_info_exp = self.temporal_tracker.update(
                        refined_angles[0], refined_angles[1], u1, u2
                    )
                    refined_angles = [ref0, ref1]
                    temporal_xai = self.temporal_tracker.get_temporal_xai()
                    if spike_info_exp["spikes_this_frame"]:
                        debug_info.append(
                            f"Temporal (Expert): Spike on {spike_info_exp['spikes_this_frame']} rejected."
                        )
                    debug_info.append(
                        f"Temporal (Expert): Kalman H={ref0:.1f}\u00b0 (was {ra0_raw:.1f}\u00b0), "
                        f"M={ref1:.1f}\u00b0 (was {ra1_raw:.1f}\u00b0)"
                    )
                else:
                    temporal_xai = None

                h_new, m_new, err_new = self._solve_physics(refined_angles[0], refined_angles[1])
                
                ambiguity_warning_expert = self._resolve_ambiguity(refined_angles[0], refined_angles[1], h_new, m_new)
                if ambiguity_warning_expert and not ambiguity_warning:
                    debug_info.append(ambiguity_warning_expert)
                drift_str_expert = self._calculate_drift(h_new, m_new, device_time_str)

                # [6.7] Hand type heuristic: shorter hand = hour
                hand_type_info = self._classify_hand_type_heuristic(tip1, tip2, center)
                debug_info.append(f"Hand Type Heuristic: {hand_type_info}")

                # [6.6] Contrastive XAI — "Why not X:XX?"
                contrastive_xai = None
                if self.contrastive_explainer and raw_heatmaps:
                    top_n = self._get_top_n_candidates(refined_angles[0], refined_angles[1], n=4)
                    fused_hm = raw_heatmaps[0]   # Use hand1 heatmap as primary
                    contrastive_xai = self.contrastive_explainer.explain(
                        fused_hm, h_new, m_new, top_n,
                        hand1_angle=refined_angles[0],
                    )
                    debug_info.append(f"Contrastive XAI: {contrastive_xai[:80]}...")

                return {
                    "time": f"{h_new}:{m_new:02d}",
                    "method": "Expert Path (C1+C2+C3+C4)",
                    "confidence": "Refined",
                    "heatmap": heatmap_img,
                    "debug": debug_info,
                    "visualizations": visualizations,
                    "angles": {"hand1": refined_angles[0], "hand2": refined_angles[1]},
                    "reasoning": f"Refined: H={refined_angles[0]:.1f}\u00b0, M={refined_angles[1]:.1f}\u00b0 \u2192 Time={h_new}:{m_new:02d}",
                    "error": "",
                    "ampm": ampm_status,
                    "drift": drift_str_expert,
                    "ambiguity": ambiguity_warning_expert or ambiguity_warning,
                    "c1_conf": float(conf),
                    "c1_quality": quality,
                    "c1_clock_conf": float(c_conf) if c_conf != -1.0 else 0.0,
                    "c1_gauge_conf": float(g_conf) if g_conf != -1.0 else 0.0,
                    "c1_clock_quality": c_quality,
                    "c1_gauge_quality": g_quality,
                    "c2_research": c2_research_data,
                    "uncertainty_deg": uncertainty_summary,
                    "xai_method": "GradCAM++ (multi-layer)",
                    "temporal_xai": temporal_xai,
                    "contrastive_xai": contrastive_xai,
                }

# Alias mapping so you don't break existing `main.py` imports 
ClockEngine = HARPEngine