import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
# Path to the model you just trained
MODEL_PATH = r"D:\Y4S1\Research 3\Data Model\clock_cvat_project\pose_run\weights\best.pt"

# Path to an image you want to test (Pick one from your dataset for now)
# CHANGE THIS filename to one that actually exists in your train folder!
TEST_IMAGE = r"D:\Y4S1\Research 3\Data Model\dataset\images\train\000000009172.jpg" 
# ---------------------

def align_clock():
    # 1. Load your custom model
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 2. Run Inference
    results = model(TEST_IMAGE)
    result = results[0]

    # 3. Check if a clock was found
    if result.keypoints is None or len(result.keypoints) == 0:
        print("No clock detected!")
        return

    # 4. Get Keypoints
    # The model outputs: [Center, 12, 3, 6, 9] based on your CVAT order
    # Index mapping: 0=Center, 1=12, 2=3, 3=6, 4=9
    kpts = result.keypoints.xy[0].cpu().numpy()
    
    # We grab the 4 outer points to define the "Circle"
    # (We don't strictly need the center for the transformation matrix)
    pt_12 = kpts[1]
    pt_3  = kpts[2]
    pt_6  = kpts[3]
    pt_9  = kpts[4]

    print(f"Detected 12 o'clock at: {pt_12}")

    # 5. Define Destination Points (Where we WANT them to be)
    # We create a 400x400 blank image
    out_size = 400
    margin = 50
    
    # Coordinates for a perfect upright square/diamond shape
    dst_pts = np.array([
        [out_size/2, margin],            # 12 goes to Top-Middle
        [out_size-margin, out_size/2],   # 3 goes to Right-Middle
        [out_size/2, out_size-margin],   # 6 goes to Bottom-Middle
        [margin, out_size/2]             # 9 goes to Left-Middle
    ], dtype=np.float32)

    # Source points from the image
    src_pts = np.array([pt_12, pt_3, pt_6, pt_9], dtype=np.float32)

    # 6. Calculate the Homography Matrix (The Math Magic)
    M, _ = cv2.findHomography(src_pts, dst_pts)

    # 7. Warp the image
    orig_img = cv2.imread(TEST_IMAGE)
    warped_img = cv2.warpPerspective(orig_img, M, (out_size, out_size))

    # 8. Show Results
    cv2.imshow("Original", cv2.resize(orig_img, (500, 500))) # Resize for viewing
    cv2.imshow("Straightened Clock", warped_img)
    
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    align_clock()