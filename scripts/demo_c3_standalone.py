"""
demo_c3_standalone.py — Standalone demo for the C3 angle-regression model.

Uses C2 (YOLO-pose) to find a hand, feeds the aligned crop to C3, and shows
the raw vs. refined angle side-by-side.

Architecture must stay in sync with HARPEngine._get_c3_arch() in app/core/engine.py.
"""
import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

# --- CONFIGURATION ---
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
C2_PATH      = os.path.join(PROJECT_ROOT, "models", "c2_hands_skeleton",   "best.pt")
C3_PATH      = os.path.join(PROJECT_ROOT, "models", "c3_angle_regression", "best.pth")
INPUT_DIR    = os.path.join(PROJECT_ROOT, "data", "straight_clocks_dataset")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. DEFINE C3 ARCHITECTURE ---
# Must exactly match HARPEngine._get_c3_arch() in app/core/engine.py.
def get_c3_model():
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(backbone.fc.in_features, 2),   # (sin θ, cos θ)
    )
    return nn.Sequential(backbone)   # model[0] = backbone for XAI hooks


# --- 2. UTILS ---
def get_aligned_crop(img, center, rough_angle, box_size=128):
    """
    Rotate the image CCW by rough_angle so the hand is approximately vertical,
    then crop box_size×box_size around the centre point.
    Pads with white if the crop window exceeds the image boundary.
    """
    h, w = img.shape[:2]
    cx, cy = float(center[0]), float(center[1])

    M = cv2.getRotationMatrix2D((cx, cy), rough_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    half = box_size // 2
    x1 = int(cx - half)
    y1 = int(cy - half)

    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, (x1 + box_size) - w)
    pad_bottom = max(0, (y1 + box_size) - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        rotated = cv2.copyMakeBorder(
            rotated, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        x1 += pad_left
        y1 += pad_top

    return rotated[y1 : y1 + box_size, x1 : x1 + box_size]


def calculate_geometry_angle(center, point):
    """Clockwise angle from 12 o'clock, range [0, 360)."""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return math.degrees(math.atan2(dx, -dy)) % 360


# --- 3. MAIN DEMO ---
def main():
    print("--- COMPONENT 3: STANDALONE DEMONSTRATION ---")

    print("1. Loading C2 (helper to find hands)...")
    c2_model = YOLO(C2_PATH)

    print("2. Loading C3 (angle regression model)...")
    c3_model = get_c3_model().to(DEVICE)
    if os.path.exists(C3_PATH):
        c3_model.load_state_dict(torch.load(C3_PATH, map_location=DEVICE))
        c3_model.eval()
        print("   ✅ C3 loaded successfully.")
    else:
        print("   ❌ Error: C3 model not found.")
        return

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".png"))]
    print(f"\nPress [SPACE] to analyze next hand, [Q] to quit.\n")

    for img_name in images:
        original_img = cv2.imread(os.path.join(INPUT_DIR, img_name))
        if original_img is None:
            continue

        results = c2_model(original_img, verbose=False)[0]
        if results.keypoints is None or len(results.keypoints.data) == 0:
            continue

        kpts   = results.keypoints.data[0].cpu().numpy()
        center = (float(kpts[0][0]), float(kpts[0][1]))
        tip    = (float(kpts[1][0]), float(kpts[1][1]))

        # Rough angle from C2 keypoints
        geo_angle = calculate_geometry_angle(center, tip)

        # Aligned crop (hand ≈ vertical)
        crop = get_aligned_crop(original_img, center, geo_angle)

        # C3 inference: predict (sin θ, cos θ) of the residual
        pil_crop     = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            raw = c3_model(input_tensor)[0]   # shape [2]

        sin_p = raw[0].item()
        cos_p = raw[1].item()

        # Residual = how far off from vertical the hand is in the crop
        c3_residual = math.degrees(math.atan2(sin_p, cos_p))
        # Delta stays in [-180, 180] naturally from atan2
        delta = c3_residual

        final_angle = (geo_angle + delta) % 360

        # --- VISUALIZATION ---
        crop_display = cv2.resize(crop, (256, 256))
        cv2.putText(crop_display, "C3 Input", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        print(f"Image: {img_name}")
        print(f"   > Geometric (C2) angle : {geo_angle:.2f}°")
        print(f"   > C3 residual          : {delta:+.2f}°  (sin={sin_p:.3f}, cos={cos_p:.3f})")
        print(f"   > Final refined angle  : {final_angle:.2f}°")
        print("-" * 40)

        cv2.imshow("Component 3 Input (The Neural Crop)", crop_display)

        display_main = original_img.copy()
        cv2.line(display_main,
                 (int(center[0]), int(center[1])),
                 (int(tip[0]),    int(tip[1])),
                 (0, 255, 255), 2)
        cv2.putText(display_main, f"C3: {final_angle:.1f}°",
                    (int(tip[0]), int(tip[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Full Context", display_main)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
