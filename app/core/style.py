"""
app/core/style.py
=================
[T3.2] Clock Style Embedding — Style-conditioned C3 regression head.

Classifies a clock crop into one of three style categories and uses
that style embedding to condition the C3 angle regression head.

Style Classes:
    0 — Modern Analog     (clean dial, simple hands, clear numerals)
    1 — Antique / Ornate  (ornate hands, roman numerals, aged patina)
    2 — Minimalist        (no numerals, thin hands, bare face)

Architecture:
    ClockStyleClassifier:  ResNet18 backbone → 3-class softmax
    ClockStyleEmbedding:   3-dim one-hot → 8-dim learnable embedding
    StyleConditionedC3:    ResNet18 backbone + style embedding concat → angle head

Research angle:
    Domain adaptation — clock style as an implicit domain label improves
    cross-style generalisation with minimal additional parameters.
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image

STYLE_NAMES = ["Modern Analog", "Antique/Ornate", "Minimalist"]
NUM_STYLES   = len(STYLE_NAMES)


class ClockStyleClassifier(nn.Module):
    """
    [T3.2] Lightweight 3-class clock style classifier.

    Backbone: ResNet18 (pretrained or from scratch).
    Head:     Linear(512 → 3) + Softmax.

    Without trained weights this returns uniform probabilities.
    Designed for easy fine-tuning on a small labelled style dataset (~150 images).
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])   # (N, 512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_STYLES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (N, 3) logits — use softmax for probabilities."""
        feat = self.features(x)
        return self.classifier(feat)

    @torch.no_grad()
    def predict_style(self, pil_image: Image.Image, device="cpu") -> tuple:
        """
        Classify a PIL image.

        Returns:
            (style_idx, style_name, confidence, probs_np)
        """
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        t = transform(pil_image).unsqueeze(0).to(device)
        logits = self.forward(t)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = int(probs.argmax())
        return idx, STYLE_NAMES[idx], float(probs[idx]), probs.cpu().numpy()


class ClockStyleEmbedding(nn.Module):
    """
    [T3.2] Learnable 8-dimensional style embedding layer.

    Maps a 3-class style index → 8-dim dense vector, which is concatenated
    with the ResNet backbone features before the angle regression head.

    This adds only 3×8 = 24 parameters but significantly conditions the
    angle head on the detected clock aesthetics.
    """

    EMBED_DIM = 8

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(NUM_STYLES, self.EMBED_DIM)

    def forward(self, style_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            style_idx: (N,) long tensor of style class indices.
        Returns:
            (N, 8) embedding vectors.
        """
        return self.embedding(style_idx)


class StyleConditionedC3(nn.Module):
    """
    [T3.2] Style-conditioned C3 angle regression model.

    Architecture:
        ResNet18 backbone → 512-dim feature
        +
        StyleEmbedding    →   8-dim style vector
        ──────────────────────────────────────
        concat → 520-dim → Linear(520 → 1) → Sigmoid → angle ∈ [0,1]

    The style embedding conditions the regression head without modifying
    the backbone weights (allowing backbone pre-training transfer).

    For circular output: replace the Sigmoid head with CircularHead(520)
    from app.core.losses.
    """

    def __init__(self, circular: bool = False):
        """
        Args:
            circular: If True, output (sin θ, cos θ) instead of Sigmoid scalar.
        """
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.features  = nn.Sequential(*list(backbone.children())[:-1])   # (N, 512, 1, 1)
        self.style_emb = ClockStyleEmbedding()
        self.circular  = circular

        combined_dim = 512 + ClockStyleEmbedding.EMBED_DIM  # 520

        if circular:
            from app.core.losses import CircularHead
            self.head = CircularHead(in_features=combined_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(combined_dim, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor, style_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         (N, 3, 64, 64) normalised image tensor.
            style_idx: (N,) long tensor — style class index.
        Returns:
            (N, 1) sigmoid scalar or (N, 2) sin/cos pair.
        """
        feat  = self.features(x).squeeze(-1).squeeze(-1)     # (N, 512)
        style = self.style_emb(style_idx)                     # (N, 8)
        combined = torch.cat([feat, style], dim=1)            # (N, 520)
        return self.head(combined)


class StyleAnalyser:
    """
    [T3.2] Convenience wrapper for runtime style detection in the HARP pipeline.

    Loads an optional pre-trained ClockStyleClassifier checkpoint and provides
    a simple classify() API. Falls back to style_idx=0 (Modern) if no weights.
    """

    def __init__(self, weights_path: str = None, device: str = "cpu"):
        self.device = device
        self.model  = ClockStyleClassifier(pretrained=False).to(device)
        self.model.eval()
        self._loaded = False

        if weights_path:
            import os
            if os.path.exists(weights_path):
                try:
                    self.model.load_state_dict(
                        torch.load(weights_path, map_location=device)
                    )
                    self._loaded = True
                    print(f"✅ StyleClassifier weights loaded from {weights_path}")
                except Exception as e:
                    print(f"⚠️ StyleClassifier: failed to load weights: {e}")
            else:
                print(f"⚠️ StyleClassifier: weights not found at {weights_path}, using uniform priors.")

    def classify(self, pil_image: Image.Image) -> dict:
        """
        Returns a dict: {style_idx, style_name, confidence, probs}
        """
        idx, name, conf, probs = self.model.predict_style(pil_image, self.device)
        return {
            "style_idx":   idx,
            "style_name":  name,
            "confidence":  round(conf, 4),
            "probs":       {STYLE_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_STYLES)},
            "weights_loaded": self._loaded,
        }
