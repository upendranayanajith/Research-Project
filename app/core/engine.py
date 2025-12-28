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
        
        # --- PATHS ---
        self.c1_path = os.path.join(base_dir, "models", "c1_localization", "best.pt")
        self.c2_path = os.path.join(base_dir, "models", "c2_hands_skeleton", "best.pt")
        self.c3_path = os.path.join(base_dir, "models", "c3_angle_regression", "best.pth")
        
        # --- LOAD C1 (LOCALIZATION) ---
        print(f"Loading C1: {self.c1_path}...")
        try:
            self.c1_model = YOLO(self.c1_path)
        except Exception as e:
            print(f"⚠️ C1 Failed to load: {e}")
            self.c1_model = None

        # --- LOAD C2 (POSE) ---
        print(f"Loading C2: {self.c2_path}...")
        self.c2_model = YOLO(self.c2_path)
        
        # --- LOAD C3 (PRECISION) ---
        print(f"Loading C3: {self.c3_path}...")
        self.c3_model = self._get_c3_arch().to(self.device)
        
        if os.path.exists(self.c3_path):
            self.c3_model.load_state_dict(torch.load(self.c3_path, map_location=self.device))
            self.c3_model.eval()
            self.xai = XaiVisualizer(self.c3_model[0]) 
        else:
            print("⚠️ WARNING: C3 weights not found.")
            self.c3_model = None

        # --- PHYSICS DATA (C4) ---
        self.possible_minutes = np.arange(0, 720)
        self.theory_h = (self.possible_minutes * 0.5) % 360
        self.theory_m = (self.possible_minutes * 6.0) % 360

        # --- TRANSFORMS ---
        self.c3_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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
        """C4 Logic"""
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
        return rotated[int(center[1]-s):int(center[1]+s), int(center[0]-s):int(center[0]+s)]

    def _localize_clock(self, img):
        """
        C1: Finds the clock in a wild image and crops it.
        """
        if self.c1_model is None:
            return img, False  # Fallback: Use full image
            
        # Run C1 Detection
        results = self.c1_model(img, verbose=False)[0]
        
        if len(results.boxes) == 0:
            return img, False # No clock found
            
        # Get the box with highest confidence
        best_box = results.boxes[0]
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        conf = float(best_box.conf[0])
        
        # Crop with some padding
        h, w = img.shape[:2]
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        crop = img[y1:y2, x1:x2]
        return crop, True

    def analyze(self, img_array, force_expert=False):
        debug_info = []

        # [STEP 1] C1: LOCALIZATION
        # Scan the wild image to find the clock
        clock_crop, found_clock = self._localize_clock(img_array)
        
        if found_clock:
            debug_info.append("C1: Clock Detected & Cropped")
        else:
            debug_info.append("C1: No clock detected (Scanning full image)")

        # [STEP 2] C2: POSE ESTIMATION
        # Run C2 on the CROP, not the full room
        results = self.c2_model(clock_crop, verbose=False)[0]
        
        if not results.keypoints or len(results.keypoints.data) == 0:
            return {"error": "C2 Failed: No hands found on clock surface"}
        
        kpts = results.keypoints.data[0].cpu().numpy()
        center, tip1, tip2 = kpts[0][:2], kpts[1][:2], kpts[2][:2]
        
        a1 = self._get_angle(center, tip1)
        a2 = self._get_angle(center, tip2)
        
        # [STEP 3] C4: PHYSICS REASONING (Fast Path)
        h, m, error = self._solve_physics(a1, a2)
        
        # [STEP 4] HYBRID DECISION GATE
        if error < 8.0 and not force_expert:
            return {
                "time": f"{h}:{m:02d}",
                "method": "Fast Path (C1 -> C2 -> C4)",
                "confidence": "High",
                "heatmap": None,
                "debug": debug_info
            }
        
        else:
            # [STEP 5] EXPERT PATH (C3 Refinement)
            if self.c3_model is None:
                return {"time": f"{h}:{m:02d}", "method": "Fast Path (C3 Missing)", "heatmap": None}

            refined_angles = []
            heatmap_img = None
            
            for tip, rough_angle in zip([tip1, tip2], [a1, a2]):
                crop = self._get_crop(clock_crop, center, rough_angle)
                if crop.size == 0: 
                    refined_angles.append(rough_angle)
                    continue
                
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                t_input = self.c3_transform(pil_crop).unsqueeze(0).to(self.device)
                
                if heatmap_img is None:
                    norm_crop = np.array(pil_crop, dtype=np.float32) / 255.0
                    heatmap_img = self.xai.generate(t_input, norm_crop)

                with torch.no_grad():
                    pred = self.c3_model(t_input).item()
                
                c3_angle = pred * 360.0
                delta = c3_angle - 360 if c3_angle > 180 else c3_angle
                refined_angles.append((rough_angle + delta) % 360)

            h_new, m_new, err_new = self._solve_physics(refined_angles[0], refined_angles[1])
            
            return {
                "time": f"{h_new}:{m_new:02d}",
                "method": "Expert Path (C1 -> C2 -> C3 -> C4)",
                "confidence": "Refined",
                "heatmap": heatmap_img,
                "debug": debug_info
            }