"""
app/core/vit_xai.py
===================
[T3.4] ViT Backbone Architecture + Attention Rollout Visualizer.

Replaces ResNet18 with Vision Transformer (ViT-B/16) as the C3 backbone.
ViT's multi-head self-attention is inherently interpretable — no GradCAM
needed. We extract attention weights directly to produce explanation maps.

Why ViT for C3:
  - Self-attention naturally focuses on clock hand edges and tips
  - Attention maps replace GradCAM++ entirely (no backward pass required)
  - State-of-the-art for fine-grained visual recognition
  - Attention rollout (Abnar & Zuidema, 2020) provides faithful XAI

Research angle:
    "Attention is interpretable" — comparing ViT attention rollout vs
    GradCAM++ as XAI methods for analog clock reading.

References:
    Dosovitskiy et al. (2020) "An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale"
    Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers"
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models


# ---------------------------------------------------------------------------
# T3.4 — ViT-based C3 Architecture
# ---------------------------------------------------------------------------
class ViTC3Model(nn.Module):
    """
    [T3.4] ViT-B/16 backbone with angle regression head.

    Input:  (N, 3, 64, 64) — same as ResNet18 C3 (upsampled internally to 224×224)
    Output: (N, 1) scalar ∈ [0,1] via Sigmoid  OR  (N, 2) [sin, cos] via CircularHead

    NOTE: ViT-B/16 requires minimum 224×224 input. We add an interpolation
    layer to accept 64×64 crops (maintaining API compatibility with engine).
    """

    def __init__(self, circular: bool = False, pretrained: bool = True):
        """
        Args:
            circular:  If True, outputs (sin θ, cos θ) for Von Mises loss.
            pretrained: If True, loads ImageNet-pretrained ViT-B/16 weights.
        """
        super().__init__()

        # Input upsampler: 64×64 → 224×224 (ViT requirement)
        self.upsample = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)

        # ViT-B/16 backbone
        weights = "IMAGENET1K_V1" if pretrained else None
        vit = models.vit_b_16(weights=weights)

        # Extract backbone (everything except the final classification head)
        self.patch_embed = vit.conv_proj        # 3→768 patch projection
        self.encoder     = vit.encoder          # Transformer encoder (12 layers)
        self.class_token = vit.class_token      # [CLS] token
        self.hidden_dim  = 768                  # ViT-B hidden dimension

        # Angle regression head
        self.circular = circular
        if circular:
            from app.core.losses import CircularHead
            self.angle_head = CircularHead(in_features=self.hidden_dim)
        else:
            self.angle_head = nn.Sequential(
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid(),
            )

        # Store attention weights for XAI
        self._attention_weights: list = []
        self._register_attention_hooks(vit)

    def _register_attention_hooks(self, vit: nn.Module):
        """Register forward hooks to capture attention weights from all 12 heads."""
        self._attention_weights = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                # ViT attention returns (attention_output, attention_weights)
                # We capture the raw Q·K^T softmax weights
                if hasattr(module, 'weights') or isinstance(output, tuple):
                    pass   # Will use alternative approach below
            return hook

        # Note: torchvision ViT stores attention differently — we hook into
        # the SelfAttention modules to capture weights during forward pass
        for layer in self.encoder.layers:
            if hasattr(layer, 'self_attention'):
                layer.self_attention.register_forward_hook(self._attention_hook)

    def _attention_hook(self, module, input, output):
        """Capture attention weights. Called by registered hooks."""
        # torchvision ViT's MultiheadAttention returns (output, attn_weights)
        if isinstance(output, tuple) and len(output) == 2:
            _, attn = output
            if attn is not None:
                self._attention_weights.append(attn.detach().cpu())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, 64, 64) normalised tensor.
        Returns:
            (N, 1) sigmoid or (N, 2) sin/cos.
        """
        self._attention_weights = []   # Reset per forward pass

        x = self.upsample(x)     # → (N, 3, 224, 224)

        # Patch embedding
        x = self.patch_embed(x)  # → (N, 768, 14, 14)
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).permute(0, 2, 1)   # → (N, 196, 768)

        # Prepend [CLS] token
        cls = self.class_token.expand(n, -1, -1)    # (N, 1, 768)
        x   = torch.cat([cls, x], dim=1)             # (N, 197, 768)

        # Transformer encoder
        x = self.encoder(x)   # (N, 197, 768)

        # Use [CLS] token for prediction
        cls_feat = x[:, 0]    # (N, 768)
        return self.angle_head(cls_feat)


# ---------------------------------------------------------------------------
# T3.4 — Attention Rollout Visualizer
# ---------------------------------------------------------------------------
class VitAttentionVisualizer:
    """
    [T3.4] Attention Rollout XAI for ViT-B/16.

    Implements Abnar & Zuidema (2020) attention rollout:
      - Recursively multiplies attention matrices across all 12 transformer layers
      - Each layer's attention is averaged across heads
      - Identity skip-connection is added (residual attention flow)
      - CLS token's attention to all patch tokens → (196,) → (14,14) → resize

    This produces a faithful explanation map WITHOUT requiring backward passes.
    Computationally very cheap compared to GradCAM++.

    Usage:
        visualizer = VitAttentionVisualizer()
        overlay, attn_map = visualizer.generate(vit_model, input_tensor, original_image)
    """

    def __init__(self, head_fusion: str = "mean", discard_ratio: float = 0.9):
        """
        Args:
            head_fusion:   How to aggregate multi-head attention — 'mean' | 'max' | 'min'.
            discard_ratio: Fraction of lowest attention weights to zero out (noise filtering).
        """
        self.head_fusion   = head_fusion
        self.discard_ratio = discard_ratio

    def generate(
        self,
        vit_model: ViTC3Model,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
    ):
        """
        Run a forward pass and compute the attention rollout map.

        Args:
            vit_model:      ViTC3Model instance.
            input_tensor:   (1, 3, 64, 64) normalised tensor.
            original_image: (64, 64, 3) float32 [0,1] — for overlay display.

        Returns:
            visualization (np.ndarray uint8 HxWx3): attention overlay on input.
            attn_map      (np.ndarray float32 HxW): raw rollout map [0,1].
        """
        vit_model.eval()
        with torch.no_grad():
            _ = vit_model(input_tensor)

        attn_weights = vit_model._attention_weights  # List of (1, num_heads, N, N)

        if not attn_weights:
            # Fallback: return uniform map if hooks didn't capture weights
            blank = np.ones(original_image.shape[:2], dtype=np.float32) * 0.5
            overlay = (original_image * 255).astype(np.uint8)
            return overlay, blank

        # Attention rollout
        rollout = self._rollout(attn_weights)  # (N_tokens, N_tokens)

        # Extract CLS → patch attention
        n_patches  = rollout.shape[-1] - 1   # exclude CLS token itself
        grid_size  = int(n_patches ** 0.5)   # 14 for ViT-B/16
        cls_to_patch = rollout[0, 1:]        # (196,) — CLS row, skip CLS column

        # Reshape and resize to input resolution
        attn_grid   = cls_to_patch.reshape(grid_size, grid_size)
        h, w        = original_image.shape[:2]
        attn_resized = cv2.resize(attn_grid, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalise to [0, 1]
        a_min, a_max = attn_resized.min(), attn_resized.max()
        if a_max > a_min:
            attn_map = (attn_resized - a_min) / (a_max - a_min)
        else:
            attn_map = attn_resized

        # Colour overlay (blue-red heatmap)
        coloured   = cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        coloured_f = coloured.astype(np.float32) / 255.0
        img_f      = original_image.astype(np.float32)
        overlay_f  = 0.5 * img_f + 0.5 * coloured_f[:, :, ::-1]  # BGR→RGB blend
        overlay    = np.clip(overlay_f * 255, 0, 255).astype(np.uint8)

        return overlay, attn_map

    def _rollout(self, attn_weights: list) -> np.ndarray:
        """
        Compute Attention Rollout across all transformer layers.

        Recursively: rollout = (A_L + I)/2 · (A_{L-1} + I)/2 · ... · (A_1 + I)/2
        where I is the identity (skip connection).
        """
        result = None

        for attn in attn_weights:
            # attn: (batch, heads, tokens, tokens) or (batch, tokens, tokens)
            if attn.dim() == 4:
                # Multi-head: fuse across heads
                if self.head_fusion == "mean":
                    a = attn.mean(dim=1)    # (B, N, N)
                elif self.head_fusion == "max":
                    a = attn.max(dim=1).values
                else:
                    a = attn.min(dim=1).values
            else:
                a = attn

            a = a.squeeze(0).numpy()   # (N, N)

            # Add identity (residual attention flow)
            n     = a.shape[0]
            a     = a + np.eye(n)
            a    /= a.sum(axis=-1, keepdims=True) + 1e-8

            # Discard low-attention noise
            flat  = a.flatten()
            threshold = flat[int(len(flat) * self.discard_ratio)]
            a[a < threshold] = 0.0
            a /= a.sum(axis=-1, keepdims=True) + 1e-8

            result = a if result is None else np.matmul(result, a)

        return result if result is not None else np.eye(197)
