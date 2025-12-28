from ultralytics import YOLO
import cv2
import numpy as np
import os
import math
from scripts.c4_reasoning_engine import ClockPhysicsEngine # Import our new solver

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
C2_PATH = os.path.join(PROJECT_ROOT, "models", "c2_hands_skeleton", "best.pt")
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "straight_clocks_dataset")
# --------------

def calculate_angle(center, point):
    """Returns angle from 12 o'clock (Up), Clockwise positive"""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    # atan2(y, x) -> atan2(dx, -dy) swaps to 0=Up, CW rotation
    angle = math.degrees(math.atan2(dx, -dy))
    if angle < 0: angle += 360
    return angle

def main():
    print(f"Loading C2 Model: {C2_PATH}...")
    c2_model = YOLO(C2_PATH)
    physics_engine = ClockPhysicsEngine()
    
    images = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.png'))]
    print("Controls:\n  [SPACE] Next Image\n  [Q] Quit")
    
    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 1. Get Skeleton from C2
        results = c2_model(img, verbose=False)[0]
        if results.keypoints is None or len(results.keypoints.data) == 0:
            continue
            
        kpts = results.keypoints.data[0].cpu().numpy()
        
        # Extract points (We DON'T care which is H or M yet)
        p0 = (kpts[0][0], kpts[0][1]) # Center
        p1 = (kpts[1][0], kpts[1][1]) # Tip A
        p2 = (kpts[2][0], kpts[2][1]) # Tip B
        
        # 2. Identify Center (Geometric Median)
        # In a triangle of points, center is usually the one with acute angles to others
        # Simplified: We trust the model's "Point 0" is Center for now.
        center = p0
        
        # 3. Calculate Raw Angles
        angle_A = calculate_angle(center, p1)
        angle_B = calculate_angle(center, p2)
        
        # 4. SOLVE PHYSICS
        # We pass both angles. The solver decides which is H and which is M.
        best_h, best_m, error = physics_engine.solve_time(angle_A, angle_B)
        
        time_str = f"{best_h}:{best_m:02d}"
        
        # --- VISUALIZATION ---
        display_img = img.copy()
        
        # Draw Center
        cv2.circle(display_img, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
        
        # Draw Lines (Yellow)
        cv2.line(display_img, (int(center[0]), int(center[1])), (int(p1[0]), int(p1[1])), (0, 255, 255), 2)
        cv2.line(display_img, (int(center[0]), int(center[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 2)
        
        # Overlay Text
        text = f"{time_str} (Err: {error:.1f})"
        color = (0, 255, 0) if error < 15 else (0, 0, 255) # Green if confident, Red if unsure
        
        cv2.putText(display_img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display_img, f"A:{angle_A:.0f} B:{angle_B:.0f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        print(f"🕒 {img_name} -> {time_str}")
        
        cv2.imshow("Physics Solver Result", display_img)
        if cv2.waitKey(0) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()