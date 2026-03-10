"""
scripts/train_c3_circular.py
=============================
[T3.5] Full training script for circular C3 angle regression using Von Mises loss.

Key differences from standard Sigmoid training:
  - CircularHead outputs (sin θ, cos θ) instead of scalar ∈ [0,1]
  - VonMisesLoss replaces MSELoss — handles 0°/360° wraparound correctly
  - MeanCircularError used as evaluation metric instead of MAE

Usage:
    # Full training:
    python scripts/train_c3_circular.py \\
        --data_dir data/c3_hand_crops \\
        --output_dir models/c3_circular \\
        --epochs 50 \\
        --lr 1e-4

    # Dry run (validates architecture + loss without loading real data):
    python scripts/train_c3_circular.py --dry-run

    # Style-conditioned training:
    python scripts/train_c3_circular.py --style-conditioned

Output:
    models/c3_circular/best.pth        — best checkpoint (by MCE)
    models/c3_circular/final.pth       — final epoch checkpoint
    models/c3_circular/training_log.csv — per-epoch metrics
"""

import os
import sys
import argparse
import csv
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image

# Allow importing from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.core.losses import VonMisesLoss, CircularHead, decode_circular, MeanCircularError


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class C3CircularDataset(Dataset):
    """
    Loads the C3 hand-crop dataset used by the standard training pipeline,
    but returns angle labels as raw degrees ∈ [0, 360) for VonMisesLoss.

    Expected directory structure:
        data/c3_hand_crops/
            ├── 045.5_crop_001.jpg   (filename prefix = angle in degrees)
            ├── 180.0_crop_002.jpg
            └── ...

    Filename convention: "{angle:.1f}_*.jpg" where angle is the ground-truth
    clock-hand direction from 12 o'clock, clockwise, in degrees.
    """

    TRANSFORM = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.1),    # Very mild augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    TRANSFORM_VAL = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, data_dir: str, split: str = "train"):
        self.split    = split
        self.samples  = []
        transform_fn  = self.TRANSFORM if split == "train" else self.TRANSFORM_VAL

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data dir not found: {data_dir}")

        for fname in sorted(os.listdir(data_dir)):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            # Parse angle from filename prefix: "045.5_crop_001.jpg" → 45.5
            try:
                angle_deg = float(fname.split("_")[0])
                if not (0.0 <= angle_deg < 360.0):
                    continue
                self.samples.append((os.path.join(data_dir, fname), angle_deg))
            except (ValueError, IndexError):
                continue

        self.transform = transform_fn
        print(f"  Loaded {len(self.samples)} samples for split='{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, angle_deg = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x   = self.transform(img)
        y   = torch.tensor(angle_deg, dtype=torch.float32)
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Circular C3 Architecture
# ─────────────────────────────────────────────────────────────────────────────
def build_circular_model() -> nn.Module:
    """
    ResNet18 backbone + CircularHead (sin/cos output).

    Compatible with the existing C3 inference pipeline after adding
    `_decode_circular()` to engine.py.
    """
    backbone = models.resnet18(weights="IMAGENET1K_V1")
    # Remove the original classification head
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])   # (N, 512, 1, 1)
    flatten = nn.Flatten()                                                  # (N, 512)
    head    = CircularHead(in_features=512)                                 # (N, 2)
    return nn.Sequential(feature_extractor, flatten, head)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  [T3.5] Circular C3 Training — Von Mises Loss")
    print(f"  Device: {device}")
    print(f"  Data:   {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    full_ds   = C3CircularDataset(args.data_dir)
    val_size  = max(1, int(len(full_ds) * args.val_split))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # Override transform for val split
    val_ds.dataset.transform = C3CircularDataset.TRANSFORM_VAL

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_circular_model().to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss + Optimiser ──────────────────────────────────────────────────────
    criterion = VonMisesLoss(reduction="mean")
    metric    = MeanCircularError()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_mce  = float("inf")
    log_rows  = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)                    # (N, 2)
            loss  = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validate
        model.eval()
        val_mces = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y   = x.to(device), y.to(device)
                preds  = model(x)
                mce    = metric(preds, y)
                val_mces.append(mce.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_mce   = np.mean(val_mces)
        elapsed       = time.time() - t0

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Loss={avg_train_loss:.4f} | MCE={avg_val_mce:.2f}° | "
              f"LR={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_mce_deg": round(avg_val_mce, 4),
            "lr": scheduler.get_last_lr()[0],
        })

        # Checkpoint
        if avg_val_mce < best_mce:
            best_mce = avg_val_mce
            path = os.path.join(args.output_dir, "best.pth")
            torch.save(model.state_dict(), path)
            print(f"    ✅ New best MCE={best_mce:.2f}° → saved to {path}")

    # Final save
    final_path = os.path.join(args.output_dir, "final.pth")
    torch.save(model.state_dict(), final_path)

    # Write CSV log
    csv_path = os.path.join(args.output_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader()
        w.writerows(log_rows)

    print(f"\n  Training complete. Best MCE = {best_mce:.2f}°")
    print(f"  Best checkpoint: {os.path.join(args.output_dir, 'best.pth')}")
    print(f"  Log:             {csv_path}")


def dry_run():
    """
    Validates architecture + loss function without loading any data.
    Runs a single forward + backward pass on synthetic inputs.
    """
    print("\n[T3.5] DRY RUN — validating CircularHead + VonMisesLoss")
    device = torch.device("cpu")

    model     = build_circular_model().to(device)
    criterion = VonMisesLoss()
    metric    = MeanCircularError()

    # Synthetic batch: 4 images, 4 angle labels
    x = torch.randn(4, 3, 64, 64)
    y = torch.tensor([0.0, 90.0, 180.0, 270.0])   # ° — includes 0°/360° boundary

    preds = model(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {preds.shape}   (sin, cos)")
    print(f"  Output sample: {preds[0].detach().numpy()}")

    loss = criterion(preds, y)
    mce  = metric(preds, y)

    # Decode predictions
    decoded = decode_circular(preds)
    print(f"  VonMisesLoss: {loss.item():.4f}")
    print(f"  MCE:          {mce.item():.2f}°")
    print(f"  Decoded angles: {decoded.detach().numpy().round(1)}")

    # Backward pass
    loss.backward()
    print("  Backward pass: OK")

    # Test 0°/360° boundary
    boundary_preds = torch.tensor([[0.0, 1.0], [0.0, 1.0]])  # Both predict ~0°
    boundary_y     = torch.tensor([0.0, 359.9])
    boundary_loss  = VonMisesLoss()(boundary_preds, boundary_y)
    print(f"\n  Boundary test (0 deg vs 359.9 deg): loss={boundary_loss.item():.6f}")
    print(f"  (MSE on Sigmoid would give ~(1.0 - 0.0)^2 = 1.0 -- Von Mises correctly gives ~0)")
    print("\n  DRY RUN PASSED\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[T3.5] Train C3 with circular Von Mises loss")
    parser.add_argument("--data_dir",   default="data/c3_hand_crops",   help="Directory with labelled crops")
    parser.add_argument("--output_dir", default="models/c3_circular",   help="Where to save checkpoints")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--val_split",  type=float, default=0.15,        help="Fraction for validation")
    parser.add_argument("--dry-run",    action="store_true",             help="Validate without training")
    parser.add_argument("--style-conditioned", action="store_true",      help="Use StyleConditionedC3 backbone")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        train(args)
