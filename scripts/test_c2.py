from ultralytics import YOLO
import cv2
<<<<<<< HEAD
import os

# --- CONFIG ---
MODEL_PATH = "../models/c2_hands_skeleton/best.pt"
# Use the straight clocks dataset we generated earlier
TEST_IMAGE_DIR = "../data/straight_clocks_dataset" 
# --------------

def test_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}. Did you download it from folder 'skeleton12'?")
        return

    print(f"Loading model: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # Get first 10 images
    images = [os.path.join(TEST_IMAGE_DIR, f) for f in os.listdir(TEST_IMAGE_DIR) if f.endswith('.jpg')][:10]

    print("Controls:\n  [SPACE] Next Image\n  [Q] Quit")

=======
import numpy as np
import os
import math

# --- CONFIGURATION ---
# Auto-detect project root to prevent path errors
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Update these filenames if yours are different
MODEL_PATH = os.path.join(BASE_DIR, "best.pt") 
TEST_IMAGE_DIR = os.path.join(BASE_DIR, "testing_samples")
# ---------------------

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def mark_hands_optimized():
    print(f"📂 Looking for images in: {TEST_IMAGE_DIR}")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
    model = YOLO(MODEL_PATH)

    # 2. Find Images (Supports multiple formats)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [
        os.path.join(TEST_IMAGE_DIR, f) 
        for f in os.listdir(TEST_IMAGE_DIR) 
        if f.lower().endswith(valid_extensions)
    ]

    if not images:
        print("❌ No images found. Please check your 'testing_samples' folder.")
        return

    print("✅ Press 'SPACE' for next image, 'q' to quit.")

    # 3. Process Images
>>>>>>> 3e7cb5c564c88fda6e95b1be64ae2f71b836d184
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None: continue

<<<<<<< HEAD
        # Run AI
        results = model(img)[0]

        # Draw Skeleton
        if results.keypoints is not None and len(results.keypoints.data) > 0:
            # Get the first skeleton (x, y, confidence)
            kpts = results.keypoints.data[0].cpu().numpy() 
            
            # Extract points
            center = (int(kpts[0][0]), int(kpts[0][1]))
            hour   = (int(kpts[1][0]), int(kpts[1][1]))
            minute = (int(kpts[2][0]), int(kpts[2][1]))
            
            # Visuals
            # Center (Blue Dot)
            cv2.circle(img, center, 6, (255, 0, 0), -1) 
            
            # Hour Hand (GREEN Line)
            cv2.line(img, center, hour, (0, 255, 0), 3)
            
            # Minute Hand (RED Line)
            cv2.line(img, center, minute, (0, 0, 255), 3)

            # Labels
            cv2.putText(img, "H", hour, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, "M", minute, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("C2 Verification (Green=Hour, Red=Minute)", img)
=======
        # Run AI Inference
        results = model(img)[0]

        # Only proceed if a skeleton is detected
        if results.keypoints is not None and len(results.keypoints.data) > 0:
            # Extract raw coordinates
            kpts = results.keypoints.data[0].cpu().numpy() # Shape: [3, 3] (x, y, conf)
            
            # Map Keypoints (0=Center, 1=Point A, 2=Point B)
            center = (int(kpts[0][0]), int(kpts[0][1]))
            point_a = (int(kpts[1][0]), int(kpts[1][1]))
            point_b = (int(kpts[2][0]), int(kpts[2][1]))

            # --- OPTIMIZATION LOGIC ---
            # The AI might swap them, but Geometry never lies.
            # The Minute hand is ALWAYS longer than the Hour hand.
            len_a = calculate_distance(center, point_a)
            len_b = calculate_distance(center, point_b)

            if len_a > len_b:
                # Point A is longer -> Minute Hand
                minute_tip = point_a
                hour_tip = point_b
            else:
                # Point B is longer -> Minute Hand
                minute_tip = point_b
                hour_tip = point_a

            # --- VISUALIZATION ---
            # Draw Center (Blue Dot)
            cv2.circle(img, center, 6, (255, 0, 0), -1) 

            # Draw Hour Hand (GREEN = Short)
            cv2.line(img, center, hour_tip, (0, 255, 0), 4) # Thickness 4
            cv2.putText(img, "H", hour_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw Minute Hand (RED = Long)
            cv2.line(img, center, minute_tip, (0, 0, 255), 2) # Thickness 2
            cv2.putText(img, "M", minute_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show Result
        cv2.imshow("Optimized Hand Marking", img)
>>>>>>> 3e7cb5c564c88fda6e95b1be64ae2f71b836d184
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
<<<<<<< HEAD
    test_model()
=======
    mark_hands_optimized()
>>>>>>> 3e7cb5c564c88fda6e95b1be64ae2f71b836d184
