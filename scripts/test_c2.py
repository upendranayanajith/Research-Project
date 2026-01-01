from ultralytics import YOLO
import cv2
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

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None: continue

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
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_model()