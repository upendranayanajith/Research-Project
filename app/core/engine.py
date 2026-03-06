import os
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
from app.core.xai import XaiVisualizer, SemanticExplainer
from app.core.metrics import calculate_gauge_reading, calculate_gauge_reading_advanced
import easyocr
import re
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
        print("Loading EasyOCR...")
        # Use simple English model, disabled GPU if needed
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

    # --- CLOCK PHYSICS SOLVER ---
    def _solve_physics(self, a1, a2):
        raw_minutes = a2 / 6.0
        minutes = int(round(raw_minutes)) % 60
        raw_hour_pos = a1 / 30.0
        
        if raw_hour_pos > 11.5 and minutes < 30:
            raw_hour_pos = 0.0 
            
        if minutes < 45:
            hour = int(math.floor(raw_hour_pos + 0.2))
        else:
            hour = int(round(raw_hour_pos)) - 1
            
        if hour <= 0: hour = 12
        if hour > 12: hour = hour - 12
        
        theory_h_angle = (hour % 12 * 30) + (minutes * 0.5)
        diff = abs(a1 - theory_h_angle)
        error = min(diff, 360 - diff)

        return hour, minutes, error

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
        return cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)

    def _get_roi(self, img, center_pt, patch_size=60):
        h, w = img.shape[:2]
        x, y = int(center_pt[0]), int(center_pt[1])
        half = patch_size // 2
        
        y1, y2 = max(0, y - half), min(h, y + half)
        x1, x2 = max(0, x - half), min(w, x + half)
        
        if x2 - x1 == 0 or y2 - y1 == 0: return None
        return img[y1:y2, x1:x2]
        
    def _extract_number(self, roi_img):
        if roi_img is None or roi_img.size == 0: return None
        
        # Preprocessing for OCR
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # Apply slight blur to remove noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Adaptive thresholding to handle lighting
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        results = self.reader.readtext(thresh)
        if not results: return None
        
        # Join pieces of text just in case, removing spaces
        full_text = "".join([res[1] for res in results])
        
        # Regex to find numbers (integers or decimals)
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", full_text)
        if matches:
            try: return float(matches[0])
            except: return None
            
        return None

    def _localize_object(self, img):
        # Run both Gatekeeper models
        clock_res = self.c1_clock_model(img, verbose=False)[0] if self.c1_clock_model else None
        gauge_res = self.c1_gauge_model(img, verbose=False)[0] if self.c1_gauge_model else None
        
        clock_conf, gauge_conf = -1, -1
        clock_box, gauge_box = None, None
        
        if clock_res and len(clock_res.boxes) > 0:
            clock_box = clock_res.boxes[0]
            clock_conf = clock_box.conf.item()
            
        if gauge_res and len(gauge_res.boxes) > 0:
            gauge_box = gauge_res.boxes[0]
            gauge_conf = gauge_box.conf.item()
            
        if clock_conf == -1 and gauge_conf == -1: return img, False, None, None
        
        detected_type = 'clock' if clock_conf > gauge_conf else 'gauge'
        best_box = clock_box if detected_type == 'clock' else gauge_box
        
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        h, w = img.shape[:2]
        pad = 30
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        return img[y1:y2, x1:x2], True, (x1, y1, x2, y2), detected_type

    def _draw_bbox(self, img, bbox, detected_type):
        img_copy = img.copy()
        x1, y1, x2, y2 = bbox
        label = f"{detected_type.capitalize()} Detected"
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(img_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return self._resize_small(img_copy)

    def _draw_clock_angles(self, img, center, tip1, tip2, a1, a2):
        img_copy = img.copy()
        center_pt = (int(center[0]), int(center[1]))
        tip1_pt = (int(tip1[0]), int(tip1[1]))
        tip2_pt = (int(tip2[0]), int(tip2[1]))
        
        cv2.line(img_copy, center_pt, tip1_pt, (0, 255, 0), 4)
        cv2.line(img_copy, center_pt, tip2_pt, (0, 0, 255), 4)
        cv2.circle(img_copy, center_pt, 8, (255, 255, 0), -1)
        cv2.circle(img_copy, tip1_pt, 10, (0, 255, 0), -1)
        cv2.circle(img_copy, tip2_pt, 10, (0, 0, 255), -1)
        
        cv2.putText(img_copy, "H", (tip1_pt[0]+10, tip1_pt[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        cv2.putText(img_copy, "M", (tip2_pt[0]+10, tip2_pt[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.putText(img_copy, f"H: {a1:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img_copy, f"M: {a2:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
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

    def analyze(self, img_array, force_expert=False):
        debug_info = []
        visualizations = {}

        # --- [C1] LOCALIZATION (Runs dynamically for both modes now) ---
        target_crop, found_object, bbox, c1_detected_type = self._localize_object(img_array)
        
        if not found_object:
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

        debug_info.append(f"C1: Initial guess is {c1_detected_type.capitalize()}")

        # --- [C2] CROSS-VALIDATION ---
        # Instead of trusting C1 natively, we use C2 Specialists to validate presence of keypoints
        c2_clock_conf = 0.0
        c2_gauge_conf = 0.0
        clock_kpts, gauge_kpts = None, None
        clock_kpts_data, gauge_kpts_data = None, None
        
        if self.c2_clock_model:
            c_res = self.c2_clock_model(target_crop, verbose=False)[0]
            if c_res.keypoints and len(c_res.keypoints.data) > 0:
                kpts = c_res.keypoints.data[0].cpu().numpy()
                if len(kpts) >= 3:
                    c2_clock_conf = np.mean(kpts[:3, 2]) # Average confidence of keypoints
                    clock_kpts = kpts
                    clock_kpts_data = c_res.keypoints
                    
        if self.c2_gauge_model:
            g_res = self.c2_gauge_model(target_crop, verbose=False)[0]
            if g_res.keypoints and len(g_res.keypoints.data) > 0:
                kpts = g_res.keypoints.data[0].cpu().numpy()
                if len(kpts) >= 4:
                    c2_gauge_conf = np.mean(kpts[:4, 2]) # Average confidence of keypoints
                    gauge_kpts = kpts
                    gauge_kpts_data = g_res.keypoints

        if c2_clock_conf == 0.0 and c2_gauge_conf == 0.0:
             return {"error": "C2 Failed: Neither clock hands nor gauge skeleton found in the crop."}
             
        # The true type is whichever C2 model is more confident about its keypoints
        detected_type = 'clock' if c2_clock_conf > c2_gauge_conf else 'gauge'
        debug_info.append(f"C2 Validation: Chose {detected_type.capitalize()} (Clock KP Conf: {c2_clock_conf:.2f}, Gauge KP Conf: {c2_gauge_conf:.2f})")

        visualizations['c1_detection'] = self._draw_bbox(img_array, bbox, detected_type)

        # ==========================================
        # GAUGE LOGIC PIPELINE
        # ==========================================
        if detected_type == 'gauge':
            
            kpts = gauge_kpts
            
            center, min_pt, max_pt, tip = kpts[0][:2], kpts[1][:2], kpts[2][:2], kpts[3][:2]
            
            # ROI Extraction & OCR
            min_roi = self._get_roi(target_crop, min_pt)
            max_roi = self._get_roi(target_crop, max_pt)
            
            min_val = self._extract_number(min_roi)
            max_val = self._extract_number(max_roi)
            
            parsed_min = min_val if min_val is not None else "Failed"
            parsed_max = max_val if max_val is not None else "Failed"
            
            debug_info.append(f"OCR: Extracted Min={parsed_min}, Max={parsed_max}")
            
            # C4 Physics Logic 
            if min_val is not None and max_val is not None and min_val <= max_val:
                reading = calculate_gauge_reading_advanced([center, min_pt, max_pt, tip], min_val, max_val)
                time_str = str(reading)
                method_str = "Advanced Gauge Reading (C1+C2+OCR+C4)"
            else:
                reading = calculate_gauge_reading([center, min_pt, max_pt, tip])
                time_str = f"{reading}%"
                method_str = "Gauge Reading - Fallback (C1+C2+C4)"
            
            a_min = self._get_angle(center, min_pt)
            a_max = self._get_angle(center, max_pt)
            a_tip = self._get_angle(center, tip)
            span = (a_max - a_min + 360) % 360
            needle = (a_tip - a_min + 360) % 360

            visualizations['c2_skeleton'] = self._draw_gauge_skeleton(target_crop, center, min_pt, max_pt, tip, parsed_min, parsed_max)
            visualizations['c3_angles'] = self._draw_gauge_angles(target_crop, center, min_pt, max_pt, tip, span, needle)
            
            return {
                "time": time_str,
                "method": method_str,
                "confidence": "High",
                "heatmap": None,
                "debug": debug_info + [f"Final Reading: {time_str}"],
                "visualizations": visualizations,
                "angles": {"span": span, "needle": needle},
                "scale": {"min": parsed_min, "max": parsed_max},
                "reasoning": f"Gauge Logic: Interpreted scale and position.",
                "error": ""
            }

        # ==========================================
        # CLOCK LOGIC PIPELINE
        # ==========================================
        elif detected_type == 'clock':
            kpts = clock_kpts
            center, tip1, tip2 = kpts[0][:2], kpts[1][:2], kpts[2][:2]
            
            a1 = self._get_angle(center, tip1)
            a2 = self._get_angle(center, tip2)
            
            visualizations['c2_skeleton'] = self._draw_clock_angles(target_crop, center, tip1, tip2, a1, a2)
            visualizations['c3_angles'] = visualizations['c2_skeleton']
            
            h, m, error = self._solve_physics(a1, a2)
            
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
                    "error": ""
                }
            
            # --- [C3] CLOCK EXPERT PATH ---
            else:
                if self.c3_model is None:
                    return {"time": f"{h}:{m:02d}", "method": "Fast Path (C3 Missing)", "visualizations": visualizations, "angles": {"hand1": a1, "hand2": a2}}

                refined_angles = []
                heatmap_img = None
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
                    
                    if heatmap_img is None:
                        norm_crop = np.array(pil_resized, dtype=np.float32) / 255.0
                        heatmap_img = self.xai.generate(t_input, norm_crop)

                    with torch.no_grad():
                        pred = self.c3_model(t_input).item()
                    
                    c3_angle = pred * 360.0
                    delta = c3_angle - 360 if c3_angle > 180 else c3_angle
                    
                    if self.explainer and force_expert:
                        explanation = self.explainer.explain(
                            crop, heatmap_img, c3_angle, hand_type=f"Hand {i+1}"
                        )
                        debug_info.append(f"AI Insight Hand {i+1}: {explanation}")

                    if abs(delta) > 20.0:
                        debug_info.append(f"Hand {i}: Rejected C3 delta {delta:.1f}°")
                        refined_angles.append(rough_angle)
                    else:
                        debug_info.append(f"Hand {i}: Accepted C3 delta {delta:.1f}°")
                        refined_angles.append((rough_angle + delta) % 360)

                visualizations['c3_crops'] = c3_crops
                visualizations['c3_angles'] = self._draw_clock_angles(target_crop, center, tip1, tip2, refined_angles[0], refined_angles[1])
                
                h_new, m_new, err_new = self._solve_physics(refined_angles[0], refined_angles[1])
                
                return {
                    "time": f"{h_new}:{m_new:02d}",
                    "method": "Expert Path (C1+C2+C3+C4)",
                    "confidence": "Refined",
                    "heatmap": heatmap_img,
                    "debug": debug_info,
                    "visualizations": visualizations,
                    "angles": {"hand1": refined_angles[0], "hand2": refined_angles[1]},
                    "reasoning": f"Refined: H={refined_angles[0]:.1f}°, M={refined_angles[1]:.1f}° → Time={h_new}:{m_new:02d}",
                    "error": ""
                }

# Alias mapping so you don't break existing `main.py` imports 
ClockEngine = HARPEngine