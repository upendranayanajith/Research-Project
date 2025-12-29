from ultralytics import YOLO
import cv2
import numpy as np
import os
import math
from c4_reasoning_engine import ClockPhysicsEngine 

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
C2_PATH = os.path.join(PROJECT_ROOT, "models", "c2_hands_skeleton", "best.pt")
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "straight_clocks_dataset")
# --------------

def calculate_angle(center, point):
    """Geometry: Angle from 12 o'clock (Up), Clockwise positive"""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    angle = math.degrees(math.atan2(dx, -dy))
    if angle < 0: angle += 360
    return angle

def main():
    print(f"Loading C2 Model: {C2_PATH}...")
    c2_model = YOLO(C2_PATH)
    physics_engine = ClockPhysicsEngine()
    
    images = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.png'))]
    print(f"Loaded {len(images)} images. Controls: [SPACE] Next, [Q] Quit")
    
    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # --- C2: STRUCTURAL DISAGGREGATION ---
        results = c2_model(img, verbose=False)[0]
        if results.keypoints is None or len(results.keypoints.data) == 0: continue
            
        kpts = results.keypoints.data[0].cpu().numpy()
        p0 = (kpts[0][0], kpts[0][1]) # Center
        p1 = (kpts[1][0], kpts[1][1]) # Tip A
        p2 = (kpts[2][0], kpts[2][1]) # Tip B
        
        # --- GEOMETRY EXTRACTION ---
        # We trust the model's Point 0 is Center
        center = p0
        angle_A = calculate_angle(center, p1)
        angle_B = calculate_angle(center, p2)
        
        # --- C4: COGNITIVE REASONING ---
        # "Here are two lines. Which time physically matches them?"
        h, m, score = physics_engine.solve_time(angle_A, angle_B)
        time_str = f"{h}:{m:02d}"
        
        # --- VISUALIZATION ---
        display_img = img.copy()
        cv2.putText(display_img, f"{time_str}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.putText(display_img, f"Conf: {score:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw Skeleton
        cv2.circle(display_img, (int(center[0]), int(center[1])), 6, (0, 0, 255), -1)
        cv2.line(display_img, (int(center[0]), int(center[1])), (int(p1[0]), int(p1[1])), (0, 255, 255), 2)
        cv2.line(display_img, (int(center[0]), int(center[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 2)
        
        cv2.imshow("Main Pipeline (C2 + C4)", display_img)
        if cv2.waitKey(0) == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()