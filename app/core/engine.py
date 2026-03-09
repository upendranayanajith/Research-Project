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
from app.core.xai import XaiVisualizer, SemanticExplainer
from app.core.metrics import calculate_gauge_reading, calculate_gauge_reading_advanced
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
            self.xai = XaiVisualizer(self.c3_model[0]) 
            self.explainer = SemanticExplainer()
        else:
            print("⚠️ WARNING: C3 weights not found.")
            self.c3_model = None
            self.explainer = None
        
        print("Loading EasyOCR Fallback...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

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

    def _infer_ampm(self, crop):
        if crop is None or crop.size == 0: return "Unknown", 0.0
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Unknown (Missing API Key)", 0.0
            
        try:
            genai.configure(api_key=api_key)
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = "Look at this cropped image of a clock. Based on the lighting, shadows, colors, and overall ambiance, guess whether this photo was taken during the day (AM) or night (PM). Return ONLY 'AM' or 'PM'."
            
            response = model.generate_content([prompt, pil_img])
            text = response.text.strip().upper()
            
            if "AM" in text:
                return "AM", 0.9
            elif "PM" in text:
                return "PM", 0.9
            else:
                return "Unknown", 0.5
        except Exception as e:
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
                "method": f"No Object Detected",
                "confidence": "0.0",
                "heatmap": None,
                "debug": debug_info,
                "visualizations": visualizations,
                "angles": {"hand1": 0.0, "hand2": 0.0},
                "reasoning": f"No valid clock or gauge detected.",
                "error": f"No Object Detected"
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
             return {"error": "C2 Failed: Neither clock hands nor gauge skeleton found in the crop."}
             
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
                reading = calculate_gauge_reading([center, min_pt, max_pt, tip])
                time_str = f"{reading}%"
                method_str = "Gauge Reading - Fallback (C1+C2+C4)"
                reasoning_str = "Gauge Logic: OCR extraction failed. Interpreted raw percentage. Please use 'Manual Gauge Scale Overrides' in settings."

            visualizations['c2_skeleton'] = self._draw_gauge_skeleton(target_crop, center, min_pt, max_pt, tip, parsed_min, parsed_max)
            visualizations['c3_angles'] = self._draw_gauge_angles(target_crop, center, min_pt, max_pt, tip, span, needle)
            
            return {
                "time": time_str,
                "method": method_str,
                "confidence": "High",
                "heatmap": None,
                "debug": debug_info + [f"Final Reading: {time_str}"],
                "visualizations": visualizations,
                "angles": {"span": span, "needle": needle, "units_per_deg": units_per_deg},
                "scale": {"min": parsed_min, "max": parsed_max},
                "reasoning": reasoning_str,
                "error": "",
                "c1_conf": float(conf),
                "c1_quality": quality,
                "c1_clock_conf": float(c_conf) if c_conf != -1.0 else 0.0,
                "c1_gauge_conf": float(g_conf) if g_conf != -1.0 else 0.0,
                "c1_clock_quality": c_quality,
                "c1_gauge_quality": g_quality
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
                    "c1_gauge_quality": g_quality
                }
            
            # --- [C3] CLOCK EXPERT PATH ---
            else:
                if self.c3_model is None:
                    return {"time": f"{h}:{m:02d}", "method": "Fast Path (C3 Missing)", "visualizations": visualizations, "angles": {"hand1": a1, "hand2": a2}}

                refined_angles = []
                heatmaps = []
                c3_crops = []
                
                for i, (tip, rough_angle) in enumerate(zip([tip1, tip2], [a1, a2])):
                    crop = self._get_crop(target_crop, center, rough_angle)
                    if crop.size == 0:
                        refined_angles.append(rough_angle)
                        continue
                    c3_crops.append(crop)
                    
                    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    pil_resized = pil_crop.resize((64, 64))
                    t_input = self.c3_transform(pil_resized).unsqueeze(0).to(self.device)
                    
                    norm_crop = np.array(pil_resized, dtype=np.float32) / 255.0
                    hand_heatmap = self.xai.generate(t_input, norm_crop)
                    heatmaps.append(hand_heatmap)

                    with torch.no_grad():
                        pred = self.c3_model(t_input).item()
                    
                    c3_angle = pred * 360.0
                    delta = c3_angle - 360 if c3_angle > 180 else c3_angle
                    
                    if self.explainer and force_expert:
                        explanation = self.explainer.explain(
                            crop, hand_heatmap, c3_angle, hand_type=f"Hand {i+1}"
                        )
                        debug_info.append(f"AI Insight Hand {i+1}: {explanation}")

                    if abs(delta) > 20.0:
                        debug_info.append(f"Hand {i}: Rejected C3 delta {delta:.1f}°")
                        refined_angles.append(rough_angle)
                    else:
                        debug_info.append(f"Hand {i}: Accepted C3 delta {delta:.1f}°")
                        refined_angles.append((rough_angle + delta) % 360)

                if len(heatmaps) == 2:
                    heatmap_img = np.hstack((heatmaps[0], heatmaps[1]))
                elif len(heatmaps) == 1:
                    heatmap_img = heatmaps[0]
                else:
                    heatmap_img = None
                
                visualizations['c3_crops'] = c3_crops
                visualizations['c3_angles'] = self._draw_angles_on_img(target_crop, center, tip1, tip2, refined_angles[0], refined_angles[1])
                
                h_new, m_new, err_new = self._solve_physics(refined_angles[0], refined_angles[1])
                
                ambiguity_warning_expert = self._resolve_ambiguity(refined_angles[0], refined_angles[1], h_new, m_new)
                if ambiguity_warning_expert and not ambiguity_warning:
                    debug_info.append(ambiguity_warning_expert)
                drift_str_expert = self._calculate_drift(h_new, m_new, device_time_str)
                
                return {
                    "time": f"{h_new}:{m_new:02d}",
                    "method": "Expert Path (C1+C2+C3+C4)",
                    "confidence": "Refined",
                    "heatmap": heatmap_img,
                    "debug": debug_info,
                    "visualizations": visualizations,
                    "angles": {"hand1": refined_angles[0], "hand2": refined_angles[1]},
                    "reasoning": f"Refined: H={refined_angles[0]:.1f}°, M={refined_angles[1]:.1f}° → Time={h_new}:{m_new:02d}",
                    "error": "",
                    "ampm": ampm_status,
                    "drift": drift_str_expert,
                    "ambiguity": ambiguity_warning_expert or ambiguity_warning,
                    "c1_conf": float(conf),
                    "c1_quality": quality,
                    "c1_clock_conf": float(c_conf) if c_conf != -1.0 else 0.0,
                    "c1_gauge_conf": float(g_conf) if g_conf != -1.0 else 0.0,
                    "c1_clock_quality": c_quality,
                    "c1_gauge_quality": g_quality
                }

# Alias mapping so you don't break existing `main.py` imports 
ClockEngine = HARPEngine