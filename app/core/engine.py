import os
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
from app.core.xai import XaiVisualizer

class ClockEngine:
    def __init__(self, base_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- [C1] PATHS & MODEL LOADING (Member 1) ---
        self.c1_path = os.path.join(base_dir, "models", "c1_localization", "best.pt")
        self.c1_model = self._load_yolo(self.c1_path, "C1")

        # --- [C2] PATHS & MODEL LOADING (Member 2) ---
        self.c2_path = os.path.join(base_dir, "models", "c2_hands_skeleton", "best.pt")
        self.c2_model = self._load_yolo(self.c2_path, "C2")
        
        # --- [C3] ANGLE PREDICTION MODELS (Member 3) ---
        self.c3_path = os.path.join(base_dir, "models", "c3_angle_regression", "best.pth")
        print(f"Loading C3: {self.c3_path}...")
        self.c3_model = self._get_c3_arch().to(self.device)
        
        if os.path.exists(self.c3_path):
            self.c3_model.load_state_dict(torch.load(self.c3_path, map_location=self.device))
            self.c3_model.eval()
            self.xai = XaiVisualizer(self.c3_model[0]) 
        else:
            print("⚠️ WARNING: C3 weights not found.")
            self.c3_model = None

        # --- [C4] PHYSICS LOGIC CONSTANTS (Member 4) ---
        self.possible_minutes = np.arange(0, 720)
        self.theory_h = (self.possible_minutes * 0.5) % 360
        self.theory_m = (self.possible_minutes * 6.0) % 360

        # --- [C3] PREPROCESSING (Member 3) ---
        self.c3_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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

    def _localize_clock(self, img):
        if self.c1_model is None: return img, False, None
        results = self.c1_model(img, verbose=False)[0]
        if len(results.boxes) == 0: return img, False, None
        best_box = results.boxes[0]
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        h, w = img.shape[:2]
        pad = 30
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        return img[y1:y2, x1:x2], True, (x1, y1, x2, y2)

    def _draw_bbox(self, img, bbox):
        img_copy = img.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(img_copy, "Clock Detected", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
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

    def analyze(self, img_array, force_expert=False):
        debug_info = []
        visualizations = {}

        # --- [C1] LOCALIZATION ---
        clock_crop, found_clock, bbox = self._localize_clock(img_array)
        debug_info.append("C1: Clock Found" if found_clock else "C1: Full Scan")
        
        # [Updated] Resize C1 output too
        if found_clock and bbox:
            visualizations['c1_detection'] = self._draw_bbox(img_array, bbox)
        else:
            visualizations['c1_detection'] = self._resize_small(img_array.copy())

        # --- [C2] HAND POSE ---
        if self.c2_model is None: return {"error": "Model files missing."}
        results = self.c2_model(clock_crop, verbose=False)[0]
        if not results.keypoints or len(results.keypoints.data) == 0:
            return {"error": "C2 Failed: No hands found"}
        
        kpts = results.keypoints.data[0].cpu().numpy()
        center, tip1, tip2 = kpts[0][:2], kpts[1][:2], kpts[2][:2]
        
        # [C2 Viz] Skeleton Only (Small)
        visualizations['c2_skeleton'] = self._draw_skeleton(clock_crop, center, tip1, tip2)

        # --- [C3] ANGLE PREDICTION (Fast Path) ---
        a1 = self._get_angle(center, tip1)
        a2 = self._get_angle(center, tip2)
        
        # [C3 Viz] Skeleton + Angle Text (Small)
        visualizations['c3_angles'] = self._draw_angles_on_img(clock_crop, center, tip1, tip2, a1, a2)
        
        # --- [C4] PHYSICS VALIDATION ---
        h, m, error = self._solve_physics(a1, a2)
        
        if error < 8.0 and not force_expert:
            return {
                "time": f"{h}:{m:02d}",
                "method": "Fast Path (C1+C2+C4)",
                "confidence": "High",
                "heatmap": None,
                "debug": debug_info,
                "visualizations": visualizations,
                "angles": {"hand1": a1, "hand2": a2},
                "reasoning": f"Physics: H={a1:.1f}°, M={a2:.1f}° → Time={h}:{m:02d}"
            }
        
        # --- [C3] EXPERT PATH REFINEMENT ---
        else:
            if self.c3_model is None:
                return {"time": f"{h}:{m:02d}", "method": "Fast Path (C3 Missing)", "visualizations": visualizations, "angles": {"hand1": a1, "hand2": a2}}

            refined_angles = []
            heatmap_img = None
            c3_crops = []
            
            for i, (tip, rough_angle) in enumerate(zip([tip1, tip2], [a1, a2])):
                crop = self._get_crop(clock_crop, center, rough_angle)
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
                
                if abs(delta) > 20.0:
                    debug_info.append(f"Hand {i}: Rejected C3 delta {delta:.1f}°")
                    refined_angles.append(rough_angle)
                else:
                    debug_info.append(f"Hand {i}: Accepted C3 delta {delta:.1f}°")
                    refined_angles.append((rough_angle + delta) % 360)

            visualizations['c3_crops'] = c3_crops
            
            # [C3 Viz Update] Update the angle image with refined angles if expert path ran
            visualizations['c3_angles'] = self._draw_angles_on_img(clock_crop, center, tip1, tip2, refined_angles[0], refined_angles[1])
            
            h_new, m_new, err_new = self._solve_physics(refined_angles[0], refined_angles[1])
            
            return {
                "time": f"{h_new}:{m_new:02d}",
                "method": "Expert Path (C1+C2+C3+C4)",
                "confidence": "Refined",
                "heatmap": heatmap_img,
                "debug": debug_info,
                "visualizations": visualizations,
                "angles": {"hand1": refined_angles[0], "hand2": refined_angles[1]},
                "reasoning": f"Refined: H={refined_angles[0]:.1f}°, M={refined_angles[1]:.1f}° → Time={h_new}:{m:02d}"
            }