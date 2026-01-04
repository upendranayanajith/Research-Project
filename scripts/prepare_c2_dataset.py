import os
import shutil
import random
import glob

# --- CONFIGURATION (FIXED) ---
# We calculate the paths relative to the project root so they are always correct
# (This assumes this script is inside the 'scripts' folder)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 1. Path to your images (This should be where you pasted all the clocks)
# If you pasted them into data/images_new, keep this. 
# If you pasted them into hands_dataset/images/train, change it!
SOURCE_IMAGES_DIR = os.path.join(BASE_DIR, "data", "hands_dataset", "images", "train") 

# 2. Path to your labels (FIXED based on your screenshot)
# It is NOT inside 'data'. It is in the project root.
SOURCE_LABELS_DIR = os.path.join(BASE_DIR, "data", "hands_dataset", "labels", "train") 

# 3. Where to save the final clean dataset
DEST_DIR = os.path.join(BASE_DIR, "data", "c2_final_dataset")

# 4. Split Ratio
TRAIN_RATIO = 0.8
# -------------------------------------------

def prepare_dataset():
    # Verify paths exist before starting
    if not os.path.exists(SOURCE_LABELS_DIR):
        print(f"❌ CRITICAL ERROR: Labels folder not found at:\n   {SOURCE_LABELS_DIR}")
        print("   Please check if the folder name is 'hands_dataset' or something else.")
        return

    # 1. Clean/Create Destination Folders
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
        
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DEST_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, 'labels', split), exist_ok=True)

    # 2. Find Matches (Image <-> Label)
    print(f"🔍 Scanning labels in: {SOURCE_LABELS_DIR}...")
    
    # Get all .txt files
    label_files = glob.glob(os.path.join(SOURCE_LABELS_DIR, "*.txt"))
    
    if not label_files:
        print(f"❌ Error: No .txt files found inside that folder.")
        return

    valid_pairs = []
    missing_images = 0

    print(f"   Found {len(label_files)} labels. Checking for matching images...")

    for label_path in label_files:
        filename = os.path.basename(label_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # Look for matching image (jpg, png, or jpeg)
        image_path = None
        # We check the SOURCE_IMAGES_DIR defined above
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            potential_path = os.path.join(SOURCE_IMAGES_DIR, name_no_ext + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path:
            valid_pairs.append((image_path, label_path))
        else:
            missing_images += 1

    print(f"✅ Found {len(valid_pairs)} valid image-label pairs.")
    if missing_images > 0:
        print(f"⚠️ Warning: Skipped {missing_images} labels because the image file was missing.")
        print(f"   (Checked inside: {SOURCE_IMAGES_DIR})")

    if len(valid_pairs) == 0:
        print("❌ Stopping. Zero valid pairs found.")
        return

    # 3. Shuffle and Split
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * TRAIN_RATIO)
    
    train_set = valid_pairs[:split_idx]
    val_set = valid_pairs[split_idx:]

    # 4. Copy Files
    def copy_set(pairs, split_name):
        print(f"   Copying {len(pairs)} files to {split_name}...")
        for img_src, lbl_src in pairs:
            # Copy Image
            shutil.copy(img_src, os.path.join(DEST_DIR, 'images', split_name, os.path.basename(img_src)))
            # Copy Label
            shutil.copy(lbl_src, os.path.join(DEST_DIR, 'labels', split_name, os.path.basename(lbl_src)))

    print("🚀 Creating dataset...")
    copy_set(train_set, 'train')
    copy_set(val_set, 'val')

    print(f"\n✅ DONE! Your clean dataset is here: {DEST_DIR}")
    print("👉 Next: Zip the 'c2_final_dataset' folder (inside data) and upload it to Drive.")

if __name__ == "__main__":
    prepare_dataset()