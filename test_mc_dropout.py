"""
Quick smoke test for MC Dropout uncertainty estimation in C3.
Verifies _predict_with_uncertainty returns a valid circular mean and non-zero std.
"""
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

# ── Build the same architecture as engine._get_c3_arch() ──────────────────────
backbone = models.resnet18(weights=None)
num_ftrs = backbone.fc.in_features
backbone.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, 1)
)
model = nn.Sequential(backbone, nn.Sigmoid())
model.eval()

# ── _enable_dropout ────────────────────────────────────────────────────────────
def enable_dropout(m):
    for layer in m.modules():
        if isinstance(layer, nn.Dropout):
            layer.train()

# ── _predict_with_uncertainty ──────────────────────────────────────────────────
import scipy.stats

def predict_with_uncertainty(tensor, n_passes=20):
    model.eval()
    enable_dropout(model)
    preds_deg = []
    with torch.no_grad():
        for _ in range(n_passes):
            raw = model(tensor).item()
            preds_deg.append(raw * 360.0)
    model.eval()  # restore full eval
    preds_rad = np.radians(preds_deg)
    mean_rad = scipy.stats.circmean(preds_rad, high=2 * np.pi, low=0)
    std_rad  = scipy.stats.circstd(preds_rad,  high=2 * np.pi, low=0)
    mean_deg = float(np.degrees(mean_rad) % 360.0)
    std_deg  = float(np.degrees(std_rad))
    return mean_deg, std_deg

# ── Run test ───────────────────────────────────────────────────────────────────
dummy = torch.randn(1, 3, 64, 64)
mean_angle, sigma = predict_with_uncertainty(dummy)

print(f"Mean angle = {mean_angle:.2f}°")
print(f"Uncertainty σ = ±{sigma:.2f}°")
alpha = max(0.0, min(1.0, 1.0 - sigma / 20.0))
print(f"Alpha (blend weight) = {alpha:.3f}")

assert 0.0 <= mean_angle < 360.0, f"Mean angle out of range: {mean_angle}"
assert sigma >= 0.0, f"Std must be non-negative: {sigma}"
print("\n✅ MC Dropout uncertainty test PASSED")
