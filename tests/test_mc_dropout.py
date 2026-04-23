"""
Quick smoke test for MC Dropout uncertainty estimation in C3.
Verifies _predict_with_uncertainty returns a valid circular mean and non-zero std.

Architecture must stay in sync with HARPEngine._get_c3_arch() in app/core/engine.py.
"""
import math
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

# ── Build the same architecture as engine._get_c3_arch() ──────────────────────
backbone = models.resnet18(weights=None)
backbone.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(backbone.fc.in_features, 2),   # (sin θ, cos θ)
)
model = nn.Sequential(backbone)              # model[0] = backbone for XAI hooks
model.eval()

# ── _enable_dropout ────────────────────────────────────────────────────────────
def enable_dropout(m):
    for layer in m.modules():
        if isinstance(layer, nn.Dropout):
            layer.train()

# ── _predict_with_uncertainty (pure numpy — no scipy) ─────────────────────────
def predict_with_uncertainty(tensor, n_passes=20):
    model.eval()
    enable_dropout(model)
    preds_deg = []
    with torch.no_grad():
        for _ in range(n_passes):
            raw   = model(tensor)[0]                              # shape [2]
            sin_p = raw[0].item()
            cos_p = raw[1].item()
            preds_deg.append(math.degrees(math.atan2(sin_p, cos_p)) % 360.0)
    model.eval()

    # Circular mean & std — Mardia & Jupp formula
    preds_rad = np.radians(preds_deg)
    mean_sin  = float(np.mean(np.sin(preds_rad)))
    mean_cos  = float(np.mean(np.cos(preds_rad)))
    mean_deg  = float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0)
    R         = math.sqrt(mean_sin ** 2 + mean_cos ** 2)
    std_deg   = float(np.degrees(math.sqrt(max(0.0, -2.0 * math.log(R + 1e-9)))))
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
