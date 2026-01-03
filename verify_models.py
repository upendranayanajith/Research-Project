"""
Quick script to verify which models are loaded
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add app to path
sys.path.insert(0, BASE_DIR)

from app.core.engine import ClockEngine

print("=" * 60)
print("MODEL VERIFICATION REPORT")
print("=" * 60)

# Initialize engine
print("\n📦 Initializing ClockEngine...\n")
engine = ClockEngine(BASE_DIR)

# Check which models loaded
print("\n" + "=" * 60)
print("MODEL STATUS:")
print("=" * 60)

models = {
    "C1 (Clock Localization)": engine.c1_model,
    "C2 (Hand Skeleton Detection)": engine.c2_model,
    "C3 (Angle Regression)": engine.c3_model
}

for name, model in models.items():
    status = "✅ LOADED" if model is not None else "❌ MISSING"
    print(f"{name:30s} : {status}")

# Check file paths
print("\n" + "=" * 60)
print("MODEL FILE PATHS:")
print("=" * 60)

paths = {
    "C1 Model Path": engine.c1_path,
    "C2 Model Path": engine.c2_path,
    "C3 Model Path": engine.c3_path
}

for name, path in paths.items():
    exists = "✅ EXISTS" if os.path.exists(path) else "❌ NOT FOUND"
    size = f"({os.path.getsize(path) / 1024 / 1024:.1f} MB)" if os.path.exists(path) else ""
    print(f"{name:20s} : {exists:12s} {size}")
    print(f"  → {path}")

print("\n" + "=" * 60)
print("SYSTEM CAPABILITIES:")
print("=" * 60)

if engine.c1_model and engine.c2_model:
    print("✅ Can use FAST PATH (C1 + C2 + C4)")
elif engine.c2_model:
    print("⚠️  Can use LIMITED MODE (C2 + C4 only)")
    print("   → Will process full image (no localization)")
else:
    print("❌ CANNOT PROCESS - C2 model required")

if engine.c3_model:
    print("✅ Can use EXPERT PATH (C1 + C2 + C3 + C4)")
else:
    print("⚠️  Expert path unavailable (C3 missing)")

print("=" * 60)
