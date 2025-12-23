import os
import shutil
import random
import glob
import sys

# --- CONFIGURATION (FIXED) ---
# 1. Get the Project Root folder automatically
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 2. Path to your straight images (from generate_c2_data.py)
SOURCE_IMAGES_DIR = os.path.join(BASE_DIR, "data", "straight_clocks_dataset")

# 3. Path to your labels (Based on your screenshot)
# Note: We removed the leading '/' so it looks inside the project
SOURCE_LABELS_DIR = os.path.join(BASE_DIR, "hands_dataset", "labels", "train") 

# 4. Where the final dataset will go
DEST_DIR = os.path.join(BASE_DIR, "data", "c2_final_dataset")

# 5. Split Ratio
TRAIN_RATIO = 0.8
# ---------------------

def prepare_dataset():
    # Verify paths exist
    if not os.path.exists(SOURCE_LABELS_DIR):
        print(f"❌ Error: Labels folder not found at:\n{SOURCE_LABELS_DIR}")
        return
    if not os.path.exists(SOURCE_IMAGES_DIR):
        print(f"❌ Error: Images folder not found at:\n{SOURCE_IMAGES_DIR}")
        return

    # 1. create folders
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DEST_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, 'labels', split), exist_ok=True)

    # 2. Get all label files
    label_files = glob.glob(os.path.join(SOURCE_LABELS_DIR, "*.txt"))
    
    if len(label_files) == 0:
        print(f"❌ Error: No .txt files found in {SOURCE_LABELS_DIR}")
        return

    print(f"Found {len(label_files)} labeled samples. Shuffling and splitting...")
    
    # Shuffle to ensure random distribution
    random.shuffle(label_files)
    
    # Calculate split index
    split_idx = int(len(label_files) * TRAIN_RATIO)
    train_files = label_files[:split_idx]
    val_files = label_files[split_idx:]

    def move_files(files, split_name):
        count = 0
        missing_count = 0
        for label_path in files:
            filename = os.path.basename(label_path) # e.g., "clock_0001.txt"
            name_no_ext = os.path.splitext(filename)[0] # "clock_0001"
            
            # Find matching image (Checking jpg, png, etc.)
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_path = os.path.join(SOURCE_IMAGES_DIR, name_no_ext + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path:
                # Copy Label
                shutil.copy(label_path, os.path.join(DEST_DIR, 'labels', split_name, filename))
                # Copy Image
                shutil.copy(image_path, os.path.join(DEST_DIR, 'images', split_name, os.path.basename(image_path)))
                count += 1
            else:
                missing_count += 1
                # Optional: Print first few missing files to debug
                if missing_count < 5:
                    print(f"⚠️ Warning: Image not found for {filename}")
        
        if missing_count > 0:
            print(f"   (Skipped {missing_count} files where image was missing)")
        return count

    # 3. Process
    print("Processing Training Set...")
    train_count = move_files(train_files, 'train')
    
    print("Processing Validation Set...")
    val_count = move_files(val_files, 'val')

    print(f"\n✅ Success! Dataset prepared at: {DEST_DIR}")
    print(f"   - Train: {train_count} images")
    print(f"   - Val:   {val_count} images")
    print("\nNext Step: Zip the 'data/c2_final_dataset' folder and upload to Google Drive for training.")

if __name__ == "__main__":
    prepare_dataset()