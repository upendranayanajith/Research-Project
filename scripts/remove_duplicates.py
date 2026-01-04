import os
import hashlib
import glob

# --- CONFIGURATION ---
# Use an absolute path or ensure this relative path is correct for where you run the terminal
# If running from 'scripts' folder, use "../data/images_new"
# If running from 'root' folder, use "data/images_new"
RELATIVE_PATH = "../data/images" 

# We convert it to a full absolute path to avoid "FileNotFound" errors
DATASET_DIR = os.path.abspath(RELATIVE_PATH)
# ---------------------

def get_file_hash(filepath):
    """Calculates the MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def clean_duplicates():
    print(f"Scanning absolute path: {DATASET_DIR}")
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: The folder '{DATASET_DIR}' does not exist.")
        return

    # Get all images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(DATASET_DIR, ext)))
    
    files.sort()

    hashes = {}
    duplicates = []

    print(f"Hashing {len(files)} files...")

    for filepath in files:
        file_hash = get_file_hash(filepath)
        
        if file_hash in hashes:
            duplicates.append(filepath)
            # Store just the filename for cleaner printing
            original_name = os.path.basename(hashes[file_hash])
            dup_name = os.path.basename(filepath)
            print(f"❌ Duplicate: {dup_name} (Same as {original_name})")
        else:
            hashes[file_hash] = filepath

    if len(duplicates) == 0:
        print("\n✅ No duplicates found.")
    else:
        print(f"\n⚠️ Found {len(duplicates)} duplicates.")
        confirm = input("Type 'yes' to delete them permanently: ")
        
        if confirm.lower() == 'yes':
            for file_path in duplicates:
                try:
                    os.remove(file_path)
                    print(f"🗑️ Deleted: {os.path.basename(file_path)}")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")
            
            print("\nDone. Run your rename script again to fix numbering gaps.")
        else:
            print("Cancelled.")

if __name__ == "__main__":
    clean_duplicates()