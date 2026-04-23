"""
train_c3.py  —  Train the C3 angle-regression model.

The model receives a 128×128 hand crop (aligned to roughly 0°) and predicts
(sin θ, cos θ) where θ is the residual angle the hand sits at inside that crop.
Using sin/cos avoids the 0°/360° discontinuity that breaks plain scalar MSE.

At inference time:
    residual  = atan2(sin_pred, cos_pred)          # degrees
    final_angle = (rough_angle + residual) % 360

Run:
    python scripts/train_c3.py
"""

import csv
import math
import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

# --- CONFIG ---
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR    = os.path.join(PROJECT_ROOT, "data", "c3_hand_crops")
LABELS_CSV  = os.path.join(DATA_DIR, "labels.csv")
MODEL_OUT   = os.path.join(PROJECT_ROOT, "models", "c3_angle_regression", "best.pth")

EPOCHS      = 30
BATCH_SIZE  = 32
LR          = 1e-4
VAL_SPLIT   = 0.15   # fraction held out for validation
INPUT_SIZE  = 64     # resize crops to this before feeding the network
NUM_WORKERS = 0      # set >0 on Linux/Mac for faster loading

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------


# ── Dataset ──────────────────────────────────────────────────────────────────

class HandCropDataset(Dataset):
    """Loads hand-crop images + (sin, cos) labels from the CSV produced by
    generate_c3_dataset.py."""

    def __init__(self, data_dir, labels_csv, transform=None):
        self.data_dir  = data_dir
        self.transform = transform
        self.samples   = []   # list of (path, sin_label, cos_label)

        with open(labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(data_dir, row["filename"])
                if os.path.exists(path):
                    self.samples.append((
                        path,
                        float(row["sin_label"]),
                        float(row["cos_label"]),
                    ))

        print(f"[Dataset] Loaded {len(self.samples)} samples from {labels_csv}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, sin_lbl, cos_lbl = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor([sin_lbl, cos_lbl], dtype=torch.float32)
        return img, label


# ── Model ─────────────────────────────────────────────────────────────────────

def get_c3_model():
    """
    ResNet-18 with Dropout + 2-output (sin θ, cos θ) head.

    IMPORTANT: this must exactly match HARPEngine._get_c3_arch() in
    app/core/engine.py so that weights saved here load cleanly in production.

    Architecture:
        nn.Sequential(
            resnet18(
                fc = nn.Sequential(Dropout(0.3), Linear(512, 2))
            )
        )

    The outer nn.Sequential wrapper is required because engine.py accesses
    model[0] to hand the backbone to XAI hooks (GradCAM++, IG, LIME, SHAP).
    """
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(backbone.fc.in_features, 2),   # (sin θ, cos θ)
    )
    return nn.Sequential(backbone)   # model[0] = backbone


# ── Training helpers ──────────────────────────────────────────────────────────

def angle_error_deg(sin_pred, cos_pred, sin_true, cos_true):
    """Mean absolute angular error in degrees (handles wrap-around)."""
    pred_deg = torch.atan2(sin_pred, cos_pred) * (180.0 / math.pi)
    true_deg = torch.atan2(sin_true, cos_true) * (180.0 / math.pi)
    diff = (pred_deg - true_deg + 180) % 360 - 180   # wrap to [-180, 180]
    return diff.abs().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    total_loss, total_angle_err, n = 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)

            loss = criterion(preds, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = imgs.size(0)
            total_loss      += loss.item() * bs
            total_angle_err += angle_error_deg(
                preds[:, 0], preds[:, 1],
                labels[:, 0], labels[:, 1]
            ) * bs
            n += bs

    return total_loss / n, total_angle_err / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full_dataset = HandCropDataset(DATA_DIR, LABELS_CSV, transform)
    if len(full_dataset) == 0:
        print("❌ No samples found. Run generate_c3_dataset.py first.")
        return

    val_size   = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {train_size}  Val: {val_size}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    model     = get_c3_model().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5
    )

    best_val_loss = float("inf")

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Err°':>10}  "
          f"{'Val Loss':>9}  {'Val Err°':>9}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_err = run_epoch(
            model, train_loader, criterion, optimizer, DEVICE, training=True)
        val_loss,   val_err   = run_epoch(
            model, val_loader,   criterion, optimizer, DEVICE, training=False)

        scheduler.step(val_loss)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_OUT)
            marker = " ← saved"

        print(f"{epoch:>5}  {train_loss:>10.5f}  {train_err:>9.2f}°  "
              f"{val_loss:>9.5f}  {val_err:>8.2f}°{marker}")

    print(f"\n✅ Training complete. Best model saved to:\n   {MODEL_OUT}")


if __name__ == "__main__":
    main()
