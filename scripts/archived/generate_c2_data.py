import sys
import os
import cv2
import glob

# 1. Fix the path so we can import from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. NOW we can import the class (This was missing!)
from src.c1_localization import ClockLocalizer

# --- CONFIGURATION ---
# We use abspath to guarantee Python finds the folders, no matter where you run this script from.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

RAW_IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "straight_clocks_dataset")
C1_MODEL = os.path.join(BASE_DIR, "models", "c1_localization", "best.pt")
# ---------------------

def generate():
    # Verify paths exist before starting
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"❌ Error: Source folder not found: {RAW_IMAGES_DIR}")
        return
    if not os.path.exists(C1_MODEL):
        print(f"❌ Error: Model not found: {C1_MODEL}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading model from: {C1_MODEL}")
    c1 = ClockLocalizer(C1_MODEL, use_enhancer=False) 
    
    # Grab all image formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(RAW_IMAGES_DIR, ext)))
    
    print(f"Found {len(files)} images in {RAW_IMAGES_DIR}. Processing...")

    count = 0
    success_count = 0
    
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
            
        # Process using your C1 class (Detect -> Straighten)
        straight = c1.process_input(img)
        
        if straight is not None:
            # Save nicely named files
            save_name = os.path.join(OUTPUT_DIR, f"clock_{count:04d}.jpg")
            cv2.imwrite(save_name, straight)
            success_count += 1
            
            if success_count % 50 == 0:
                print(f"Generated {success_count} clocks...")
        
        count += 1

    print(f"\n✅ Done! Generated {success_count} straight clocks in:\n{OUTPUT_DIR}")

if __name__ == "__main__":
    generate()