import os
import shutil
import glob

# --- CONFIGURATION ---
SOURCE_DIR = "../data/images"
DEST_DIR = "../data/images_new"
START_INDEX = 1
# ---------------------

def rename_and_copy():
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # Get all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(SOURCE_DIR, ext)))
    
    # --- THE FIX IS HERE ---
    # set() removes duplicates (e.g. if Windows found the same file for .jpg AND .JPG)
    unique_files = sorted(list(set(files)))
    
    print(f"Found {len(unique_files)} unique images. Processing...")

    count = START_INDEX
    for src_path in unique_files:
        _, ext = os.path.splitext(src_path)
        
        # Force lowercase extension for consistency
        ext = ext.lower()
        
        new_name = f"{count:06d}{ext}"
        dst_path = os.path.join(DEST_DIR, new_name)
        
        shutil.copy2(src_path, dst_path)
        
        # Optional: Print every 100th file to reduce clutter
        if count % 100 == 0:
            print(f"Copied {count} images...")
            
        count += 1

    print(f"\n✅ Success! Copied {count - START_INDEX} clean images to:\n{DEST_DIR}")

if __name__ == "__main__":
    rename_and_copy()