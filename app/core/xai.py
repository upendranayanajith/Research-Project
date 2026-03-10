"""
app/core/xai.py — Full 7-Layer XAI Stack for HARP C3 Angle Regressor

Layer 1 : XaiVisualizer        — GradCAM++ multi-layer fusion (L2/L3/L4)
Layer 2 : LocalExplainer       — heuristic heatmap stats, always-on
Layer 3 : SemanticExplainer    — Gemini Vision LLM fallback
Layer 4 : AdaptiveSemanticRouter — entropy-gated L2↔L3 routing
Layer 5 : ContrastiveExplainer — "Why not X:XX?" counterfactual text
Layer 6 : LimeExplainer        — LIME superpixel overlay
Layer 7 : ShapExplainer        — SHAP DeepExplainer attribution map
Util    : compute_entropy(heatmap) → float [0, 1]
"""

import os
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ── Opt-in imports (soft-fail if library not installed) ───────────────────────
try:
    import google.generativeai as genai  # type: ignore
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False

try:
    from pytorch_grad_cam import GradCAMPlusPlus          # type: ignore
    from pytorch_grad_cam.utils.image import show_cam_on_image  # type: ignore
    _GRADCAM_OK = True
except ImportError:
    _GRADCAM_OK = False

try:
    from lime.lime_image import LimeImageExplainer        # type: ignore
    _LIME_OK = True
except ImportError:
    _LIME_OK = False

try:
    import shap                                           # type: ignore
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════
def compute_entropy(heatmap: np.ndarray) -> float:
    """
    Normalised Shannon entropy of a grayscale heatmap.

    Returns a value in [0, 1]:
      0.0 → single peak (model is tightly focused)
      1.0 → uniform distribution (model has no idea where to look)
    """
    flat = heatmap.ravel().astype(np.float64)
    flat = flat - flat.min()
    total = flat.sum()
    if total == 0:
        return 1.0
    p = flat / total
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    H_max = np.log(len(flat))
    return float(H / H_max) if H_max > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — GradCAM++ Multi-layer Fusion
# ══════════════════════════════════════════════════════════════════════════════
class XaiVisualizer:
    """
    GradCAM++ fused across ResNet-18 layers 2, 3, 4 with depth-weighted blending.

    Weights:  layer2=0.20, layer3=0.30, layer4=0.50
    Returns: (visualization uint8 HxWx3, raw_fused_heatmap float32 HxW)
    The raw heatmap feeds LocalExplainer, AdaptiveRouter, Contrastive, LIME, SHAP.
    """

    LAYER_WEIGHTS = [0.20, 0.30, 0.50]

    def __init__(self, model: nn.Module):
        self.model = model
        self.available = False
        if not _GRADCAM_OK:
            print("⚠️  XaiVisualizer: pytorch-grad-cam not installed. Using fallback.")
            return
        try:
            self.cams = [
                GradCAMPlusPlus(model=model, target_layers=[model.layer2[-1]]),
                GradCAMPlusPlus(model=model, target_layers=[model.layer3[-1]]),
                GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]]),
            ]
            self.available = True
        except Exception as e:
            print(f"⚠️  XaiVisualizer init failed: {e}")

    def generate(self, input_tensor: torch.Tensor, original_image: np.ndarray):
        """
        Args:
            input_tensor:   (1, 3, 64, 64) normalised tensor
            original_image: (64, 64, 3) float32 [0, 1]
        Returns:
            visualization (np.ndarray uint8 HxWx3)
            raw_cam       (np.ndarray float32 HxW) in [0, 1]
        """
        if not self.available:
            blank = np.zeros(original_image.shape[:2], dtype=np.float32)
            vis = (original_image * 255).astype(np.uint8)
            return vis, blank

        try:
            fused = np.zeros(original_image.shape[:2], dtype=np.float32)
            for cam, weight in zip(self.cams, self.LAYER_WEIGHTS):
                raw = cam(input_tensor=input_tensor, targets=None)[0]
                fused += weight * raw

            # Normalise fused map to [0, 1]
            fused = fused - fused.min()
            if fused.max() > 0:
                fused = fused / fused.max()

            vis = show_cam_on_image(original_image, fused, use_rgb=True)
            return vis, fused

        except Exception as e:
            blank = np.zeros(original_image.shape[:2], dtype=np.float32)
            vis = (original_image * 255).astype(np.uint8)
            print(f"⚠️  GradCAM++ failed: {e}")
            return vis, blank


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — LocalExplainer (always-on, no API required)
# ══════════════════════════════════════════════════════════════════════════════
class LocalExplainer:
    """
    Analyses the raw GradCAM++ heatmap using simple heuristics:
      - Peak location  → quadrant (upper/lower × left/right)
      - Relative entropy → tight / moderate / diffuse focus
      - Peak magnitude → confidence proxy
    Returns a human-readable 1-sentence explanation without any API calls.
    """

    def explain(self, raw_heatmap: np.ndarray, predicted_angle: float,
                hand_type: str = "Hand") -> str:
        if raw_heatmap is None or raw_heatmap.size == 0:
            return f"[Local XAI] No heatmap available for {hand_type}."

        h, w = raw_heatmap.shape[:2]
        entropy = compute_entropy(raw_heatmap)

        # Focus description
        if entropy < 0.35:
            focus_desc = "tightly concentrated"
        elif entropy < 0.65:
            focus_desc = "moderately spread"
        else:
            focus_desc = "diffusely distributed (low confidence)"

        # Peak location → quadrant
        peak_y, peak_x = np.unravel_index(np.argmax(raw_heatmap), raw_heatmap.shape[:2])
        vert = "upper" if peak_y < h / 2 else "lower"
        horiz = "left" if peak_x < w / 2 else "right"
        peak_mag = float(raw_heatmap.max())

        return (
            f"[Local XAI] For the {hand_type} (predicted {predicted_angle:.1f}°), "
            f"attention is {focus_desc} in the {vert}-{horiz} region "
            f"(peak activation={peak_mag:.2f}, entropy={entropy:.2f})."
        )


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — SemanticExplainer (Gemini Vision LLM)
# ══════════════════════════════════════════════════════════════════════════════
class SemanticExplainer:
    """
    Sends the hand crop + GradCAM++ overlay to Gemini Vision for a 1-sentence
    expert explanation. Falls back to LocalExplainer on any error.
    """

    def __init__(self):
        self.available = False
        self.local = LocalExplainer()
        if not _GENAI_OK:
            return
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                import warnings
                warnings.filterwarnings("ignore")
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel("gemini-2.5-flash")
                self.available = True
            except Exception as e:
                print(f"⚠️  SemanticExplainer init failed: {e}")

    def explain(self, crop_img, heatmap_img, predicted_angle: float,
                hand_type: str = "Hour",
                raw_heatmap: np.ndarray = None,
                use_gemini: bool = False) -> str:
        if not (self.available and use_gemini):
            # local fallback
            return self.local.explain(raw_heatmap, predicted_angle, hand_type) \
                   if raw_heatmap is not None \
                   else f"[Local XAI] Predicted {hand_type} angle: {predicted_angle:.1f}°."

        try:
            if isinstance(crop_img, np.ndarray):
                crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            if isinstance(heatmap_img, np.ndarray):
                heatmap_img = Image.fromarray(heatmap_img)

            prompt = (
                f"You are an expert Computer Vision debugger. "
                f"Image 1: a crop of a clock hand. Image 2: a GradCAM++ heatmap "
                f"showing where the ResNet-18 model focused to predict the {hand_type} angle. "
                f"Predicted angle = {predicted_angle:.1f}°. "
                f"In exactly one sentence, state whether the model is looking at the correct "
                f"visual feature (hand tip or body) and whether the prediction looks reliable."
            )
            response = self.model.generate_content([prompt, crop_img, heatmap_img])
            return f"[Gemini] {response.text.strip()}"
        except Exception as e:
            return self.local.explain(raw_heatmap, predicted_angle, hand_type) \
                   if raw_heatmap is not None else f"Explanation failed: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — AdaptiveSemanticRouter (entropy-gated)
# ══════════════════════════════════════════════════════════════════════════════
class AdaptiveSemanticRouter:
    """
    Routes explanation requests to Gemini Vision when the heatmap entropy is
    high (model is confused) and sticks with LocalExplainer when the model is
    already confident (low entropy → free, always-on).

    Entropy threshold: 0.72  (configurable via class attribute)
    """

    ENTROPY_THRESHOLD: float = 0.72

    def __init__(self, semantic_explainer: SemanticExplainer):
        self.explainer = semantic_explainer
        self.local = LocalExplainer()

    def route(self, raw_heatmap: np.ndarray, crop_img, heatmap_img,
              predicted_angle: float, hand_type: str = "Hand"):
        """
        Returns: (explanation str, routing_reason str, entropy float)
        """
        entropy = compute_entropy(raw_heatmap)
        use_gemini = entropy >= self.ENTROPY_THRESHOLD and self.explainer.available

        if use_gemini:
            reason = f"entropy={entropy:.3f} ≥ {self.ENTROPY_THRESHOLD} → Gemini Vision"
            explanation = self.explainer.explain(
                crop_img, heatmap_img, predicted_angle,
                hand_type=hand_type, raw_heatmap=raw_heatmap, use_gemini=True
            )
        else:
            reason = f"entropy={entropy:.3f} < {self.ENTROPY_THRESHOLD} → LocalExplainer"
            explanation = self.local.explain(raw_heatmap, predicted_angle, hand_type)

        return explanation, reason, entropy


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — ContrastiveExplainer ("Why not X:XX?")
# ══════════════════════════════════════════════════════════════════════════════
class ContrastiveExplainer:
    """
    For each alternative time candidate, checks whether the heatmap focus
    is geometrically consistent with where the hour hand should point at
    that alternative time.

    Output (string):
        [Contrastive XAI] Predicted 9:15. Focus: left side.
          • Instead 3:15 (err=2.1°)? Hour hand should face right — inconsistent ❌
          • Instead 9:00 (err=4.3°)? Hour hand should face left  — consistent   ✅
    """

    _SIDE_ANGLES = {
        "right":  (315, 45),    # hour hand points right (3 o'clock area)
        "bottom": (45, 135),
        "left":   (135, 225),
        "top":    (225, 315),
    }

    def _centroid_side(self, raw_heatmap: np.ndarray) -> str:
        """Return the dominant side based on heatmap centroid."""
        h, w = raw_heatmap.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        total = raw_heatmap.sum()
        if total == 0:
            return "unknown"
        cy = float((ys * raw_heatmap).sum() / total)
        cx = float((xs * raw_heatmap).sum() / total)
        # Relative centroid position
        if cx > w * 0.6:   return "right"
        if cx < w * 0.4:   return "left"
        if cy < h * 0.4:   return "top"
        return "bottom"

    def _expected_side(self, hour_angle_deg: float) -> str:
        """Map hour-hand angle to the expected image side."""
        a = hour_angle_deg % 360
        if 315 <= a or a < 45:    return "top"
        if 45  <= a < 135:        return "right"
        if 135 <= a < 225:        return "bottom"
        return "left"

    def explain(self, raw_heatmap: np.ndarray, predicted_h: int, predicted_m: int,
                candidates: list, hand1_angle: float = None) -> str:
        if raw_heatmap is None:
            return "[Contrastive XAI] No heatmap available."

        focus_side = self._centroid_side(raw_heatmap)
        lines = [
            f"[Contrastive XAI] Predicted {predicted_h}:{predicted_m:02d}. "
            f"Model focus: {focus_side} side."
        ]

        for cand in candidates[:4]:
            h, m, err = cand.get("hour", 0), cand.get("minute", 0), cand.get("error", 0)
            if h == predicted_h and m == predicted_m:
                continue
            # Hour hand angle for this candidate
            h_angle = ((h % 12) + m / 60.0) / 12.0 * 360.0
            exp_side = self._expected_side(h_angle)
            consistent = (exp_side == focus_side)
            mark = "✅" if consistent else "❌"
            lines.append(
                f"  • Instead {h}:{m:02d} (err={err:.1f}°)? "
                f"Hour hand → {exp_side} — {'consistent' if consistent else 'inconsistent'} {mark}"
            )

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — LimeExplainer (superpixel perturbation)
# ══════════════════════════════════════════════════════════════════════════════
class LimeExplainer:
    """
    LIME superpixel explanation for the C3 angle regressor.
    Requires:  pip install lime

    Perturbs the hand crop 300 times, fits a local linear model, and highlights
    the top-5 superpixels that positively influence the angle prediction.

    Returns: overlay uint8 (H, W, 3) or None if unavailable.
    """

    def __init__(self):
        self.available = _LIME_OK
        if self.available:
            self._explainer = LimeImageExplainer()

    def explain(self, model: nn.Module, input_tensor: torch.Tensor,
                norm_image: np.ndarray, n_samples: int = 300) -> np.ndarray | None:
        if not self.available:
            return None

        device = next(model.parameters()).device

        def predict_fn(images: np.ndarray) -> np.ndarray:
            """LIME calls this with a batch of perturbed images (N, H, W, C) float [0,1]."""
            batch = []
            from torchvision import transforms
            tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            for img in images:
                pil = Image.fromarray((img * 255).astype(np.uint8))
                batch.append(tfm(pil))
            t = torch.stack(batch).to(device)
            with torch.no_grad():
                out = model(t).cpu().numpy()  # (N, 1) Sigmoid output
            # LIME needs (N, num_classes) — fake a 2-class output
            return np.hstack([1 - out, out])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explanation = self._explainer.explain_instance(
                    norm_image,
                    predict_fn,
                    top_labels=1,
                    hide_color=0,
                    num_samples=n_samples,
                )
            # Get top-5 superpixels for label=1 (the "angle" side)
            temp, mask = explanation.get_image_and_mask(
                label=1,
                positive_only=True,
                num_features=5,
                hide_rest=False,
            )
            overlay = (temp * 255).astype(np.uint8)
            # Draw green contours for highlighted regions
            mask_u8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
            return overlay
        except Exception as e:
            print(f"⚠️  LIME explanation failed: {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 7 — ShapExplainer (DeepExplainer attribution)
# ══════════════════════════════════════════════════════════════════════════════
class ShapExplainer:
    """
    SHAP DeepExplainer attribution map for the C3 angle regressor.
    Requires:  pip install shap

    Uses 20 zero-tensor background samples. Returns per-pixel attribution
    as a JET-coloured overlay (red = high positive SHAP value).

    Returns: overlay uint8 (H, W, 3) or None if unavailable.
    """

    def __init__(self):
        self.available = _SHAP_OK

    def explain(self, model: nn.Module, input_tensor: torch.Tensor,
                background: torch.Tensor) -> np.ndarray | None:
        if not self.available:
            return None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(input_tensor)

            # shap_values: list[ndarray] or ndarray (C, H, W) or (1, C, H, W)
            sv = np.array(shap_values)
            if sv.ndim == 5:   sv = sv[0, 0]     # (1, 1, C, H, W) → (C, H, W)
            elif sv.ndim == 4: sv = sv[0]         # (1, C, H, W)  → (C, H, W)
            elif sv.ndim == 3: pass               # (C, H, W)

            sv_mean = np.abs(sv).mean(axis=0)     # channel mean (H, W)

            # Normalise to [0, 255]
            sv_norm = sv_mean - sv_mean.min()
            if sv_norm.max() > 0:
                sv_norm = sv_norm / sv_norm.max()
            sv_u8 = (sv_norm * 255).astype(np.uint8)
            overlay = cv2.applyColorMap(sv_u8, cv2.COLORMAP_JET)
            return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"⚠️  SHAP explanation failed: {e}")
            return None