import os
import sys

# Calculate the Project Root (Assuming this script is in /scripts/ folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

print(f"📍 Project Root detected as: {root_dir}")

# Define expected paths
paths = {
    "C1 (YOLO)": os.path.join(root_dir, "models", "c1_localization", "best.pt"),
    "C2 (YOLO)": os.path.join(root_dir, "models", "c2_hands_skeleton", "best.pt"),
    "C3 (ResNet)": os.path.join(root_dir, "models", "c3_angle_regression", "best.pth")
}

print("\n🔎 Checking Model Files...")
all_good = True

for name, path in paths.items():
    exists = os.path.exists(path)
    status = "✅ FOUND" if exists else "❌ MISSING"
    if not exists:
        all_good = False
        print(f"{status} - {name}")
        print(f"   Looking at: {path}")
    else:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{status} - {name} ({size_mb:.2f} MB)")

if all_good:
    print("\n✅ All files exist! If it still fails, the model file might be corrupt or ultralytics is not installed.")
    try:
        import ultralytics
        print(f"✅ Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics library is NOT installed. Run: pip install ultralytics")
else:
    print("\n❌ Please fix the missing files shown above.")