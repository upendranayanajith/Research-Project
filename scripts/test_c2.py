from ultralytics import YOLO
import cv2
import os

# --- CONFIG ---
MODEL_PATH = "../models/c2_hands_skeleton/best.pt"
# Use the straight clocks dataset we generated earlier
TEST_IMAGE_DIR = "../data/straight_clocks_dataset" 
# --------------

# Update these filenames if yours are different
MODEL_PATH = os.path.join(BASE_DIR, "best.pt") 
TEST_IMAGE_DIR = os.path.join(BASE_DIR, "testing_samples")
# ---------------------

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def mark_hands_optimized():
    print(f"📂 Project Root: {BASE_DIR}")
    print(f"📂 Looking for model at: {MODEL_PATH}")
    print(f"📂 Looking for images in: {TEST_IMAGE_DIR}")
    
    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model not found at {MODEL_PATH}")
        return
    
    # 2. Check if image directory exists
    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"\n❌ ERROR: Image folder not found at {TEST_IMAGE_DIR}")
        print("   Please create the folder and add test images.")
        return
    
    # 3. Load Model
    print("🚀 Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully.")

    # 4. Find Images (Supports multiple formats)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    all_files = os.listdir(TEST_IMAGE_DIR)
    print(f"📁 Files in directory: {all_files}")
    
    images = [
        os.path.join(TEST_IMAGE_DIR, f) 
        for f in all_files 
        if f.lower().endswith(valid_extensions)
    ]

    print(f"🧐 Found {len(images)} images to test.")
    
    if not images:
        print("❌ No images found. Please add .jpg, .jpeg, .png, or .bmp images to the testing_samples folder.")
        print(f"   Folder location: {TEST_IMAGE_DIR}")
        return

    print("\n✅ Press 'SPACE' for next image, 'q' to quit.\n")

    # 5. Process Images
    for img_path in images:
        print(f"Processing: {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠️ Could not read: {img_path}")
            continue

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

            # Draw Minute Hand (RED = Long)
            cv2.line(img, center, minute_tip, (0, 0, 255), 2) # Thickness 2
            cv2.putText(img, "M", minute_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            print(f"  ✅ Detected keypoints - Hour hand length: {len_a if len_a < len_b else len_b:.1f}px, Minute hand length: {max(len_a, len_b):.1f}px")
        else:
            print(f"  ⚠️ No keypoints detected in this image")

        # Show Result
        cv2.imshow("Optimized Hand Marking", img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            print("\nQuitting...")
            break

    cv2.destroyAllWindows()
    print("\n✅ Processing complete!")

if __name__ == "__main__":
    test_model()