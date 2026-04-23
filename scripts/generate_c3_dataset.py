import csv
import cv2
import math
import os
import shutil

import numpy as np
from ultralytics import YOLO

# --- CONFIG ---
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "c2_hands_skeleton", "best.pt")
INPUT_DIR  = os.path.join(PROJECT_ROOT, "data", "straight_clocks_dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "c3_hand_crops")
DEBUG_DIR  = os.path.join(PROJECT_ROOT, "data", "c3_debug")

# Noise injected onto the true angle to simulate C2 rough-estimate error.
# C3 learns to predict (and correct) this residual.
NOISE_STD_DEG = 12   # standard deviation  (degrees)
NOISE_MAX_DEG = 25   # hard clip           (degrees)

# Minimum confidence required for ALL three keypoints.
MIN_KPT_CONF = 0.5
# --------------


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_clock_angle(center, tip):
    """Clockwise angle from 12 o'clock, range [0, 360)."""
    dx = tip[0] - center[0]
    dy = tip[1] - center[1]
    angle_deg = math.degrees(math.atan2(dx, -dy))
    return angle_deg % 360


def rotate_and_crop(img, center, angle_deg, box_size=128):
    """
    Rotate the image CCW by angle_deg (cv2 positive = CCW) so the hand at
    angle_deg CW ends up pointing straight up (0°), then crop box_size×box_size
    around the center point.

    FIX: tracks left/top padding separately from right/bottom so the crop
    origin is shifted only by the padding actually added on that side.
    """
    h, w = img.shape[:2]
    cx, cy = float(center[0]), float(center[1])

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    half = box_size // 2
    x1 = int(cx - half)
    y1 = int(cy - half)

    # Compute per-side padding independently (the old code mixed left/right
    # into a single value, causing incorrect crop origin after padding).
    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, (x1 + box_size) - w)
    pad_bottom = max(0, (y1 + box_size) - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        rotated = cv2.copyMakeBorder(
            rotated, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        # Shift origin only by the padding added on the near side.
        x1 += pad_left
        y1 += pad_top

    return rotated[y1 : y1 + box_size, x1 : x1 + box_size]


def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ C2 model not found:", MODEL_PATH)
        return

    # Fresh output folders
    for d in (OUTPUT_DIR, DEBUG_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    print("✅ C2 model found. Loading...")
    model = YOLO(MODEL_PATH)

    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".png"))]
    print(f"🚀 Processing {len(images)} images...")

    labels_path = os.path.join(OUTPUT_DIR, "labels.csv")
    count = 0

    with open(labels_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "sin_label", "cos_label"])

        for img_name in images:
            img = cv2.imread(os.path.join(INPUT_DIR, img_name))
            if img is None:
                continue

            h_img, w_img = img.shape[:2]
            img_center = (w_img / 2.0, h_img / 2.0)

            results = model(img, verbose=False)[0]
            if results.keypoints is None or len(results.keypoints.data) == 0:
                continue

            kpts = results.keypoints.data[0].cpu().numpy()  # shape [3, 3]: x, y, conf

            # Check confidence on ALL three keypoints (old code only checked index 0)
            if kpts.shape[0] < 3:
                continue
            if any(kpts[i][2] < MIN_KPT_CONF for i in range(3)):
                continue

            points = [(float(kpts[i][0]), float(kpts[i][1])) for i in range(3)]

            # Identify center (closest to image centre) and two tips
            dists_to_img_center = [dist(p, img_center) for p in points]
            center_idx = int(np.argmin(dists_to_img_center))
            center_pt  = points[center_idx]

            other = [i for i in range(3) if i != center_idx]
            tip1, tip2 = points[other[0]], points[other[1]]

            # Longer arm → minute hand
            if dist(center_pt, tip1) > dist(center_pt, tip2):
                minute_pt, hour_pt = tip1, tip2
            else:
                minute_pt, hour_pt = tip2, tip1

            # Debug visualisation for first 20 images
            if count < 20:
                dbg = img.copy()
                cv2.line(dbg, (int(center_pt[0]), int(center_pt[1])),
                         (int(hour_pt[0]),   int(hour_pt[1])),   (0, 255, 0), 2)
                cv2.line(dbg, (int(center_pt[0]), int(center_pt[1])),
                         (int(minute_pt[0]), int(minute_pt[1])), (0, 0, 255), 2)
                cv2.imwrite(os.path.join(DEBUG_DIR, f"debug_{count:04d}.jpg"), dbg)

            # ── CRITICAL FIX: noise injection + label saving ──────────────────
            #
            # The true angle is calculated from C2's detected keypoints and used
            # as a proxy for ground truth.  We then ADD a small random noise to
            # simulate the imprecision of a real C2 rough estimate.
            #
            # The crop is rotated by (true_angle + noise) — i.e. the rough angle —
            # so the hand lands at (-noise) degrees in the crop frame instead of
            # exactly 0°.  C3 must learn to predict this residual so the caller
            # can add it back and recover the true angle.
            #
            # Residual angle = -noise  (the hand's actual position in the crop)
            # We encode this as (sin, cos) to avoid the 0°/360° wrap-around
            # discontinuity that breaks a plain MSE regression on a scalar angle.
            #
            # Combination at inference time:
            #   final_angle = (rough_angle + residual) % 360
            #               = (true_angle + noise  +  (-noise)) % 360
            #               = true_angle  ✓

            for hand_name, tip_pt in [("hour", hour_pt), ("minute", minute_pt)]:
                true_angle = calculate_clock_angle(center_pt, tip_pt)

                # Inject bounded Gaussian noise
                noise = float(np.clip(
                    np.random.normal(0.0, NOISE_STD_DEG),
                    -NOISE_MAX_DEG, NOISE_MAX_DEG
                ))
                rough_angle = (true_angle + noise) % 360

                # Residual the hand sits at inside the rotated crop
                residual_rad = math.radians(-noise)
                sin_label = math.sin(residual_rad)
                cos_label = math.cos(residual_rad)

                crop = rotate_and_crop(img, center_pt, rough_angle)
                if crop is None or crop.size == 0:
                    continue

                filename = f"{hand_name}_{count:04d}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, filename), crop)
                writer.writerow([filename, f"{sin_label:.6f}", f"{cos_label:.6f}"])

            count += 1
            if count % 100 == 0:
                print(f"   Processed {count} clocks...")

    print(f"\n✅ Done — {count} clocks processed.")
    print(f"   Crops  → {OUTPUT_DIR}")
    print(f"   Labels → {labels_path}")
    print(f"   Debug  → {DEBUG_DIR}")


if __name__ == "__main__":
    main()
