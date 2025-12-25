import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
import cv2
import numpy as np
import os
import math
from PIL import Image

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

C2_PATH = os.path.join(PROJECT_ROOT, "models", "c2_hands_skeleton", "best.pt")
C3_PATH = os.path.join(PROJECT_ROOT, "models", "c3_angle_regression", "best.pth")
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "straight_clocks_dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------

# --- 1. DEFINE C3 MODEL ARCHITECTURE ---
# (Must match the training script exactly)
def get_c3_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = nn.Sequential(model, nn.Sigmoid())
    return model

# --- 2. UTILITY FUNCTIONS ---
def calculate_initial_angle(center, tip):
    """ Returns angle of the hand from 12 o'clock (CW positive) """
    cx, cy = center
    tx, ty = tip
    dx = tx - cx
    dy = ty - cy 
    # math.atan2(y, x). Adjusted for 12 o'clock origin
    angle_rad = math.atan2(dx, -dy)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0: angle_deg += 360
    return angle_deg

def get_aligned_crop(img, center, rough_angle, box_size=128):
    """
    Rotates the whole image by -rough_angle so the hand becomes vertical (0 deg).
    Then crops the center.
    """
    h, w = img.shape[:2]
    cx, cy = center
    
    # Rotate CCW by rough_angle to bring hand to 0
    M = cv2.getRotationMatrix2D((cx, cy), rough_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))
    
    # Crop
    half = box_size // 2
    x1, y1 = int(cx - half), int(cy - half)
    
    # Padding
    pad_w, pad_h = 0, 0
    if x1 < 0: pad_w = -x1
    if y1 < 0: pad_h = -y1
    if x1+box_size > w: pad_w = max(pad_w, (x1+box_size)-w)
    if y1+box_size > h: pad_h = max(pad_h, (y1+box_size)-h)
    
    if pad_w > 0 or pad_h > 0:
        rotated = cv2.copyMakeBorder(rotated, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255,255,255))
        x1 += pad_w
        y1 += pad_h
        
    return rotated[y1:y1+box_size, x1:x1+box_size]

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# --- 3. MAIN PIPELINE ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load C2 (YOLO)
    print(f"Loading C2: {C2_PATH}...")
    c2_model = YOLO(C2_PATH)
    
    # Load C3 (ResNet)
    print(f"Loading C3: {C3_PATH}...")
    c3_model = get_c3_model().to(DEVICE)
    c3_model.load_state_dict(torch.load(C3_PATH, map_location=DEVICE))
    c3_model.eval()
    
    # Transforms for C3
    c3_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("\nControls:\n  [SPACE] Next Clock\n  [Q] Quit")
    
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.png'))]
    
    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        original_img = cv2.imread(img_path)
        if original_img is None: continue
        
        display_img = original_img.copy()
        h_img, w_img = original_img.shape[:2]
        img_center = (w_img/2, h_img/2)
        
        # --- A. RUN C2 (Pose) ---
        results = c2_model(original_img, verbose=False)[0]
        
        if results.keypoints is None or len(results.keypoints.data) == 0:
            print(f"Skipping {img_name}: No clock hands found.")
            continue
            
        kpts = results.keypoints.data[0].cpu().numpy()
        
        # --- B. PARSE HANDS ---
        # (Same logic as generator: identify center, hour, minute)
        points = [(kpts[i][0], kpts[i][1]) for i in range(3)]
        dists_to_center = [dist(p, img_center) for p in points]
        center_idx = np.argmin(dists_to_center)
        center_pt = points[center_idx]
        
        other_indices = [i for i in range(3) if i != center_idx]
        tip1, tip2 = points[other_indices[0]], points[other_indices[1]]
        
        if dist(center_pt, tip1) > dist(center_pt, tip2):
            minute_pt, hour_pt = tip1, tip2
        else:
            minute_pt, hour_pt = tip2, tip1
            
        # --- C. PROCESS EACH HAND ---
        times = {}
        for hand_name, tip_pt in [("Hour", hour_pt), ("Minute", minute_pt)]:
            # 1. Rough Angle from C2
            rough_angle = calculate_initial_angle(center_pt, tip_pt)
            
            # 2. Crop Aligned (Hand is now roughly vertical at 0 deg)
            crop = get_aligned_crop(original_img, center_pt, rough_angle)
            
            # 3. Refine with C3
            # Convert to PIL for PyTorch
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = c3_transform(crop_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                prediction = c3_model(input_tensor).item() # 0.0 to 1.0
                
            # Convert prediction to degrees (0-360)
            c3_angle = prediction * 360
            
            # 4. Combine
            # We rotated world by -rough_angle. 
            # The hand is now at `c3_angle`.
            # So Real Angle = rough_angle + c3_angle
            # (But handle the 360 wrap around)
            final_angle = (rough_angle + c3_angle) % 360
            
            times[hand_name] = final_angle
            
            # VISUALIZATION
            cv2.line(display_img, (int(center_pt[0]), int(center_pt[1])), 
                     (int(tip_pt[0]), int(tip_pt[1])), (0,255,0), 2)
            
        # --- D. CALCULATE TIME ---
        h_angle = times["Hour"]
        m_angle = times["Minute"]
        
        # Minutes: Simple (Angle / 6)
        minutes = int(round(m_angle / 6))
        if minutes == 60: minutes = 0
        
        # Hours: (Angle / 30). Note: 0 deg = 12.
        hours = int(h_angle / 30)
        if hours == 0: hours = 12
        
        time_str = f"{hours}:{minutes:02d}"
        
        print(f"🕒 {img_name} -> Predicted: {time_str} (H:{h_angle:.1f}°, M:{m_angle:.1f}°)")
        
        # Draw Text
        cv2.putText(display_img, time_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (0, 0, 255), 3)
        
        cv2.imshow("Final Result", display_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()