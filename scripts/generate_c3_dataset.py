import cv2
import numpy as np
import os
from ultralytics import YOLO
import math

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "c2_hands_skeleton", "best.pt")
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "straight_clocks_dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "c3_hand_crops")
DEBUG_DIR = os.path.join(PROJECT_ROOT, "data", "c3_debug") # New folder for checking errors
# --------------

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_clock_angle(center, tip):
    cx, cy = center
    tx, ty = tip
    dx = tx - cx
    dy = ty - cy 
    # math.atan2(y, x). We swap x/y to make 0 degrees = UP (North)
    # This gives us degrees CLOCKWISE from 12 o'clock
    angle_rad = math.atan2(dx, -dy) 
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg

def rotate_and_crop(img, center, angle_to_vertical, box_size=128):
    h, w = img.shape[:2]
    cx, cy = center
    
    # FIX: cv2 rotates Counter-Clockwise (CCW).
    # Our angle is Clockwise (CW) from 12.
    # To bring a hand at 3 o'clock (90 deg CW) back to 12, we must rotate 90 deg CCW.
    # So we use Positive Angle.
    M = cv2.getRotationMatrix2D((cx, cy), angle_to_vertical, 1.0)
    
    rotated_img = cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))
    
    # Crop around center
    half = box_size // 2
    x1 = int(cx - half)
    y1 = int(cy - half)
    
    # Safe Padding (Crucial for hands near the edge)
    pad_w, pad_h = 0, 0
    if x1 < 0: pad_w = -x1
    if y1 < 0: pad_h = -y1
    if x1 + box_size > w: pad_w = max(pad_w, (x1 + box_size) - w)
    if y1 + box_size > h: pad_h = max(pad_h, (y1 + box_size) - h)
    
    if pad_w > 0 or pad_h > 0:
        rotated_img = cv2.copyMakeBorder(rotated_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255,255,255))
        # Adjust coordinates after padding
        x1 += pad_w
        y1 += pad_h
        
    crop = rotated_img[y1:y1+box_size, x1:x1+box_size]
    return crop

def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found.")
        return
    
    # Clean folders
    import shutil
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    print("✅ Model found. Loading...")
    model = YOLO(MODEL_PATH)
    
    images = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.png'))]
    print(f"🚀 Processing {len(images)} images...")
    
    count = 0
    
    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        h_img, w_img = img.shape[:2]
        img_center = (w_img / 2, h_img / 2)
        
        # Inference
        results = model(img, verbose=False)[0]
        
        if results.keypoints is not None and len(results.keypoints.data) > 0:
            kpts = results.keypoints.data[0].cpu().numpy() # [3, 3]
            
            # Confidence Filter (Skip detected garbage)
            if kpts[0][2] < 0.6: continue 

            points = [(kpts[i][0], kpts[i][1]) for i in range(3)]
            
            # 1. FIND CENTER (Closest to image center)
            dists_to_img_center = [dist(p, img_center) for p in points]
            center_idx = np.argmin(dists_to_img_center)
            center_pt = points[center_idx]
            
            # 2. FIND TIPS
            other_indices = [i for i in range(3) if i != center_idx]
            tip1 = points[other_indices[0]]
            tip2 = points[other_indices[1]]
            
            # 3. IDENTIFY HANDS (Longer = Minute)
            len1 = dist(center_pt, tip1)
            len2 = dist(center_pt, tip2)
            
            if len1 > len2:
                minute_pt, hour_pt = tip1, tip2
            else:
                minute_pt, hour_pt = tip2, tip1
            
            # --- DEBUG VISUALIZATION (Save first 20 images) ---
            if count < 20:
                debug_img = img.copy()
                cv2.line(debug_img, (int(center_pt[0]), int(center_pt[1])), (int(hour_pt[0]), int(hour_pt[1])), (0, 255, 0), 2) # Green Hour
                cv2.line(debug_img, (int(center_pt[0]), int(center_pt[1])), (int(minute_pt[0]), int(minute_pt[1])), (0, 0, 255), 2) # Red Minute
                cv2.imwrite(os.path.join(DEBUG_DIR, f"debug_{count}.jpg"), debug_img)

            # --- PROCESS CROPS ---
            h_angle = calculate_clock_angle(center_pt, hour_pt)
            h_crop = rotate_and_crop(img, center_pt, h_angle)
            if h_crop is not None:
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"hour_{count}.jpg"), h_crop)
            
            m_angle = calculate_clock_angle(center_pt, minute_pt)
            m_crop = rotate_and_crop(img, center_pt, m_angle)
            if m_crop is not None:
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"minute_{count}.jpg"), m_crop)
                
            count += 1
            if count % 100 == 0:
                print(f"   Processed {count} clocks...")

    print(f"\n✅ DONE! Check 'data/c3_debug' to see if the AI is detecting correctly.")
    print(f"✅ Check 'data/c3_hand_crops' for the final vertical hands.")

if __name__ == "__main__":
    main()