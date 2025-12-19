import cv2
import argparse
import os
from src.c1_localization import ClockLocalizer

# --- CONFIG ---
MODEL_PATH = "models/c1_localization/best.pt"
OUTPUT_DIR = "output_results"
# --------------

def main(source_type, source_path):
    # 1. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    c1 = ClockLocalizer(MODEL_PATH, use_enhancer=True)
    
    # 2. Open Input Source
    if source_type == "webcam":
        cap = cv2.VideoCapture(0)
        print("[System] Starting Live Webcam Mode...")
    else:
        print(f"[System] Processing Static Image: {source_path}")
        image = cv2.imread(source_path)
        if image is None:
            print("Error: Could not load image.")
            return

    while True:
        # Get Frame
        if source_type == "webcam":
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = image.copy() # Use the static image

        # --- RUN COMPONENT 1 ---
        # This returns the Unblurred, Straightened, Cropped Clock
        processed_clock = c1.process_input(frame)

        # --- DISPLAY / SAVE RESULTS ---
        if processed_clock is not None:
            # Visualization
            cv2.imshow("1. Input Source", frame)
            cv2.imshow("2. C1 Output (Straight & Clean)", processed_clock)
            
            # If Static, Save and Exit (We don't need a loop)
            if source_type == "image":
                save_path = os.path.join(OUTPUT_DIR, "processed_clock.jpg")
                cv2.imwrite(save_path, processed_clock)
                print(f"✅ Output saved to: {save_path}")
                print("Press any key to exit.")
                cv2.waitKey(0)
                break
        else:
            cv2.imshow("1. Input Source", frame)

        # Exit Logic for Webcam
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if source_type == "webcam":
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create a simple command line interface
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["webcam", "image"], default="image", help="Choose input mode")
    parser.add_argument("--path", type=str, default="data/samples/test_clock.jpg", help="Path to image file")
    
    args = parser.parse_args()
    
    main(args.mode, args.path)