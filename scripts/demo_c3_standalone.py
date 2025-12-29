import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image
import math

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
C2_PATH = os.path.join(PROJECT_ROOT, "models", "c2_hands_skeleton", "best.pt")
C3_PATH = os.path.join(PROJECT_ROOT, "models", "c3_angle_regression", "best.pth")
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "straight_clocks_dataset")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. DEFINE C3 ARCHITECTURE ---
# (Must exactly match the training script)
def get_c3_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = nn.Sequential(model, nn.Sigmoid())
    return model

# --- 2. UTILS ---
def get_aligned_crop(img, center, rough_angle, box_size=128):
    """
    C3 requires a standard input: A hand pointing UP.
    We rotate the image by -rough_angle to normalize it.
    """
    h, w = img.shape[:2]
    cx, cy = center
    M = cv2.getRotationMatrix2D((cx, cy), rough_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))
    
    half = box_size // 2
    x1, y1 = int(cx - half), int(cy - half)
    
    # Simple bounds check
    x1 = max(0, min(x1, w - box_size))
    y1 = max(0, min(y1, h - box_size))
    
    return rotated[y1:y1+box_size, x1:x1+box_size]

def calculate_geometry_angle(center, point):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    angle = math.degrees(math.atan2(dx, -dy))
    if angle < 0: angle += 360
    return angle

# --- 3. MAIN DEMO ---
def main():
    print("--- COMPONENT 3: STANDALONE DEMONSTRATION ---")
    
    # A. Load Models
    print(f"1. Loading C2 (Helper to find hands)...")
    c2_model = YOLO(C2_PATH)
    
    print(f"2. Loading C3 (The Research Model)...")
    c3_model = get_c3_model().to(DEVICE)
    if os.path.exists(C3_PATH):
        c3_model.load_state_dict(torch.load(C3_PATH, map_location=DEVICE))
        c3_model.eval()
        print("   ✅ C3 Model Loaded Successfully.")
    else:
        print("   ❌ Error: C3 Model not found.")
        return

    # B. Transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    images = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.png'))]
    print(f"\nPress [SPACE] to analyze next hand, [Q] to quit.\n")

    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        original_img = cv2.imread(img_path)
        if original_img is None: continue

        # 1. Use C2 to find a hand to test
        results = c2_model(original_img, verbose=False)[0]
        if results.keypoints is None or len(results.keypoints.data) == 0: continue
        
        kpts = results.keypoints.data[0].cpu().numpy()
        center = (kpts[0][0], kpts[0][1])
        tip = (kpts[1][0], kpts[1][1]) # Take the first hand found

        # 2. Prepare Input for C3
        # Geometry Angle (The "Rough" guess)
        geo_angle = calculate_geometry_angle(center, tip)
        
        # Crop the hand (Normalized to point Up)
        crop = get_aligned_crop(original_img, center, geo_angle)
        
        # 3. Run C3 Inference
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_crop).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            prediction = c3_model(input_tensor).item() # 0.0 to 1.0
        
        # 4. Interpret Result
        # Model predicts deviation from "Perfect Upright"
        # 0.0 = 0 deg (or 360), 0.5 = 180 deg
        c3_raw_angle = prediction * 360.0
        
        # Calculate Delta (How much C3 disagrees with C2)
        if c3_raw_angle > 180:
            delta = c3_raw_angle - 360
        else:
            delta = c3_raw_angle
            
        final_angle = (geo_angle + delta) % 360

        # --- VISUALIZATION ---
        # Show the "Crop" C3 sees
        crop_display = cv2.resize(crop, (256, 256)) # Blow it up for display
        cv2.putText(crop_display, "C3 Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        # Show the numeric result
        print(f"Image: {img_name}")
        print(f"   > Geometric Angle: {geo_angle:.2f}°")
        print(f"   > C3 Perception:   {c3_raw_angle:.2f}° (Delta: {delta:.2f}°)")
        print(f"   > Final Precision: {final_angle:.2f}°")
        print("-" * 30)
        
        cv2.imshow("Component 3 Input (The Neural Crop)", crop_display)
        
        # Show context on main image
        display_main = original_img.copy()
        cv2.line(display_main, (int(center[0]), int(center[1])), (int(tip[0]), int(tip[1])), (0,255,255), 2)
        cv2.putText(display_main, f"C3: {final_angle:.1f}", (int(tip[0]), int(tip[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        cv2.imshow("Full Context", display_main)

        key = cv2.waitKey(0)
        if key == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()