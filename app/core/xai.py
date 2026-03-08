import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import google.generativeai as genai
import os
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


class LocalExplainer:
    """
    [FIX-2] API-free XAI fallback using heatmap statistics.
    Generates plain-English explanations from Grad-CAM activation maps alone.
    No Gemini API key required.
    """

    def explain(self, heatmap: np.ndarray, predicted_angle: float, hand_type: str = "Hour") -> str:
        """
        Analyse a grayscale heatmap (H x W, float32 in [0,1]) and
        return a human-readable explanation of what the model focused on.
        """
        if heatmap is None or heatmap.size == 0:
            return "Heatmap unavailable — no explanation could be generated."

        h, w = heatmap.shape[:2]

        # --- Peak location ---
        peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        peak_y, peak_x = peak_idx  # row=y, col=x

        # Classify quadrant
        vert  = "upper" if peak_y < h / 2 else "lower"
        horiz = "left"  if peak_x < w / 2 else "right"
        region = f"{vert}-{horiz}"

        # --- Heatmap entropy (spread) ---
        flat = heatmap.flatten().astype(np.float64) + 1e-9
        flat /= flat.sum()
        entropy = -np.sum(flat * np.log(flat))
        # Maximum entropy for a uniform map of N pixels
        max_entropy = np.log(flat.size)
        relative_entropy = entropy / max_entropy  # 0 = fully focused, 1 = fully diffuse

        if relative_entropy < 0.35:
            focus_desc = "tightly concentrated"
            confidence_hint = "suggesting high confidence in the hand-tip region"
        elif relative_entropy < 0.65:
            focus_desc = "moderately spread"
            confidence_hint = "which is typical for an intermediate-length hand"
        else:
            focus_desc = "widely diffuse"
            confidence_hint = "indicating the model is uncertain about the precise hand location"

        # --- Peak activation magnitude ---
        peak_val = float(heatmap[peak_y, peak_x])
        activation_note = (
            f"peak activation is strong ({peak_val:.2f})"
            if peak_val > 0.7
            else f"peak activation is moderate ({peak_val:.2f})"
        )

        return (
            f"[Local XAI] For the {hand_type} hand (predicted {predicted_angle:.1f}°), "
            f"the model's attention is {focus_desc} in the {region} region "
            f"({activation_note}), {confidence_hint}."
        )


class XaiVisualizer:
    """
    [FIX-1] Grad-CAM++ with multi-layer fusion across ResNet18 layer2, layer3, layer4.
    Provides richer saliency maps than single-layer Grad-CAM, especially for
    fine-grained clock/gauge hand-tip detail.
    """

    # Contribution weights for each targeted layer [layer2, layer3, layer4]
    LAYER_WEIGHTS = [0.2, 0.3, 0.5]

    def __init__(self, model):
        """
        model: ResNet18 base (before the Sigmoid wrapper in nn.Sequential).
        """
        self.model = model
        self.local_explainer = LocalExplainer()

        # Multi-layer targets: earlier layers capture finer features
        target_layers = [
            model.layer2[-1],
            model.layer3[-1],
            model.layer4[-1],
        ]
        # GradCAM++ is more accurate than GradCAM for localisation tasks
        self.cams = [
            GradCAMPlusPlus(model=model, target_layers=[layer])
            for layer in target_layers
        ]

    def generate(self, input_tensor: torch.Tensor, original_image: np.ndarray):
        """
        Generate a weighted-fused GradCAM++ heatmap.

        Args:
            input_tensor: (1, 3, 64, 64) normalised torch tensor.
            original_image: (64, 64, 3) float32 [0,1] BGR or RGB image.

        Returns:
            visualization (np.ndarray uint8): heatmap overlay on the input image.
            grayscale_cam (np.ndarray float32): raw fused heatmap [0,1] for LocalExplainer.
        """
        fused_cam = np.zeros(original_image.shape[:2], dtype=np.float32)

        for cam, weight in zip(self.cams, self.LAYER_WEIGHTS):
            grayscale = cam(input_tensor=input_tensor, targets=None)
            fused_cam += weight * grayscale[0, :]

        # Normalise fused map to [0, 1]
        fused_min, fused_max = fused_cam.min(), fused_cam.max()
        if fused_max > fused_min:
            fused_cam = (fused_cam - fused_min) / (fused_max - fused_min)

        visualization = show_cam_on_image(original_image, fused_cam, use_rgb=True)
        return visualization, fused_cam


class SemanticExplainer:
    """
    Gemini Vision-based semantic explainer with LocalExplainer fallback.
    [FIX-2] When the API is unavailable, falls back to LocalExplainer instead
    of returning a blank error string.
    """

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.local_explainer = LocalExplainer()

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.available = True
            except Exception as e:
                print(f"AI Init Failed: {e}")
                self.available = False
        else:
            self.available = False

    def explain(
        self,
        crop_img,
        heatmap_img,
        predicted_angle: float,
        hand_type: str = "Hour",
        raw_heatmap: np.ndarray = None,
        use_gemini: bool = False,
    ) -> str:
        """
        Returns a human-readable explanation of the model's focus.

        Primary path  (use_gemini=True):  Gemini Vision API → rich sentence.
        Default path  (use_gemini=False): LocalExplainer only (no API call).
        Fallback:                         LocalExplainer when Gemini fails/unavailable.

        Args:
            crop_img:        Raw clock hand crop (ndarray BGR or PIL.Image).
            heatmap_img:     GradCAM++ overlay image (ndarray or PIL.Image).
            predicted_angle: C3 model prediction in degrees.
            hand_type:       "Hour" or "Minute".
            raw_heatmap:     Grayscale fused heatmap [0,1] — used by LocalExplainer.
            use_gemini:      If True and API key present, calls Gemini first.
        """
        # --- Gemini path (only when explicitly requested and API is available) ---
        if use_gemini and self.available:
            try:
                if isinstance(crop_img, np.ndarray):
                    crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                if isinstance(heatmap_img, np.ndarray):
                    heatmap_img = Image.fromarray(heatmap_img)

                prompt = (
                    f"You are an expert Computer Vision debugger. "
                    f"The first image is a crop of a clock hand. The second is a GradCAM++ heatmap "
                    f"showing where the ResNet model looked to predict the {hand_type} angle. "
                    f"The predicted angle was {predicted_angle:.1f} degrees. "
                    f"Briefly explain (in 1 sentence) if the model is focusing on the correct visual features "
                    f"(like the hand tip or edges) to make this prediction."
                )
                response = self.model.generate_content([prompt, crop_img, heatmap_img])
                return f"[Gemini] {response.text}"
            except Exception:
                pass  # Fall through to local explainer

        # --- Local fallback (always available, no API required) ---
        return self.local_explainer.explain(raw_heatmap, predicted_angle, hand_type)


# ===========================================================================
# TIER 2 ADDITIONS
# ===========================================================================

def compute_entropy(heatmap: np.ndarray) -> float:
    """
    [6.9] Compute normalised Shannon entropy of a GradCAM++ heatmap.

    A near-zero value means the model is focused (confident).
    A value close to 1.0 means activations are diffuse (confused model).

    Args:
        heatmap: float32 array [0,1] of shape (H, W).

    Returns:
        Normalised entropy in [0, 1].
    """
    if heatmap is None or heatmap.size == 0:
        return 1.0   # Treat missing heatmap as maximally uncertain

    flat = heatmap.flatten().astype(np.float64) + 1e-9
    flat /= flat.sum()
    entropy     = -np.sum(flat * np.log(flat))
    max_entropy = np.log(flat.size)
    return float(entropy / max_entropy)


class AdaptiveSemanticRouter:
    """
    [6.9] Adaptive XAI Routing — automatically escalates to Gemini when the
    GradCAM++ heatmap entropy exceeds a threshold (model is confused).

    Usage:
        router = AdaptiveSemanticRouter(explainer)
        explanation = router.explain(raw_heatmap, crop_img, heatmap_img, angle, hand_type)
    """

    ENTROPY_THRESHOLD = 0.72   # ≥ this → call Gemini (configurable)

    def __init__(self, semantic_explainer: "SemanticExplainer"):
        self.explainer = semantic_explainer

    def explain(
        self,
        raw_heatmap: np.ndarray,
        crop_img,
        heatmap_img,
        predicted_angle: float,
        hand_type: str = "Hour",
    ) -> tuple:
        """
        Returns (explanation_str, routing_reason_str).
        Routing reason is for debug_info logging.
        """
        entropy = compute_entropy(raw_heatmap)

        if entropy >= self.ENTROPY_THRESHOLD and self.explainer.available:
            explanation = self.explainer.explain(
                crop_img, heatmap_img, predicted_angle,
                hand_type=hand_type, raw_heatmap=raw_heatmap,
                use_gemini=True,
            )
            reason = f"XAI Routing: Gemini escalated (entropy={entropy:.3f} ≥ {self.ENTROPY_THRESHOLD})"
        else:
            explanation = self.explainer.explain(
                crop_img, heatmap_img, predicted_angle,
                hand_type=hand_type, raw_heatmap=raw_heatmap,
                use_gemini=False,
            )
            reason = f"XAI Routing: LocalExplainer (entropy={entropy:.3f} < {self.ENTROPY_THRESHOLD})"

        return explanation, reason, entropy


class ContrastiveExplainer:
    """
    [6.6] Contrastive XAI — "Why not X:XX?"

    Given the predicted time and top-N alternative candidates from the C4
    physics solver, and the GradCAM++ heatmap, generates a counterfactual
    explanation: e.g. "Read 3:15 instead of 9:15 because the hour hand
    focus was on the right side (expected left for 9:xx)."
    """

    # Maps coarse angle range to clock-face side label
    _ANGLE_TO_SIDE = [
        (315, 360, "top"),
        (0,   45,  "top"),
        (45,  135, "right"),
        (135, 225, "bottom"),
        (225, 315, "left"),
    ]

    @staticmethod
    def _angle_to_side(angle_deg: float) -> str:
        a = angle_deg % 360
        for lo, hi, label in ContrastiveExplainer._ANGLE_TO_SIDE:
            if lo <= a < hi:
                return label
        return "top"

    @staticmethod
    def _heatmap_to_side(heatmap: np.ndarray) -> str:
        """Identify which side the heatmap centroid falls on."""
        if heatmap is None or heatmap.size == 0:
            return "unknown"
        h, w = heatmap.shape[:2]
        flat = heatmap.flatten().astype(np.float64) + 1e-9
        flat /= flat.sum()
        ys, xs = np.mgrid[0:h, 0:w]
        cy = float(np.sum(flat.reshape(h, w) * ys))  # centroid y
        cx = float(np.sum(flat.reshape(h, w) * xs))  # centroid x
        # Map centroid to quadrant name
        if cy < h * 0.35:
            return "top"
        elif cy > h * 0.65:
            return "bottom"
        elif cx < w * 0.4:
            return "left"
        else:
            return "right"

    def explain(
        self,
        raw_heatmap: np.ndarray,
        predicted_h: int,
        predicted_m: int,
        candidates: list,         # [(h, m, error), ...]
        hand1_angle: float = None,
    ) -> str:
        """
        Args:
            raw_heatmap:  Fused GradCAM++ map [0,1] (H, W).
            predicted_h:  Predicted hour.
            predicted_m:  Predicted minute.
            candidates:   Top-N alternative (h, m, error) tuples.
            hand1_angle:  Rough hour-hand angle (from C2/C3) for reference.

        Returns:
            Human-readable contrastive explanation string.
        """
        actual_focus = self._heatmap_to_side(raw_heatmap)
        lines = [
            f"[Contrastive XAI] The model predicted **{predicted_h}:{predicted_m:02d}**. "
            f"The GradCAM++ focus was on the **{actual_focus}** side of the crop.",
        ]

        for h, m, err in candidates[:3]:
            if h == predicted_h and m == predicted_m:
                continue   # Skip the predicted time itself
            # Expected heatmap side for this candidate's hour angle
            expected_h_angle = (h % 12) * 30 + m * 0.5
            expected_side = self._angle_to_side(expected_h_angle)
            verdict = (
                "consistent ✅" if expected_side == actual_focus
                else f"inconsistent ❌ (expected focus: {expected_side})"
            )
            lines.append(
                f"  • Instead **{h}:{m:02d}** (err={err:.1f}°)? "
                f"Hour hand should face {expected_side} — {verdict}"
            )

        if len(lines) == 1:
            lines.append("  No significant alternative candidates found.")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LIME Explainer
# ---------------------------------------------------------------------------
class LimeExplainer:
    """
    [6.10] LIME — superpixel-based perturbation explainer.
    Shows which image regions most influenced the C3 prediction.

    Produces an overlay image comparable with GradCAM++ output.
    Uses lime.lime_image.LimeImageExplainer with 300 perturbation samples.
    """

    N_SAMPLES        = 300   # Perturbation samples (speed vs. accuracy)
    N_FEATURES       = 5     # Superpixels to highlight
    BATCH_SIZE       = 32

    def __init__(self):
        try:
            from lime.lime_image import LimeImageExplainer
            self.explainer = LimeImageExplainer()
            self.available = True
        except ImportError:
            self.available = False
            print("⚠️ LIME not installed. Run: pip install lime")

    def explain(self, model, input_tensor: "torch.Tensor", norm_crop: np.ndarray) -> np.ndarray:
        """
        Args:
            model:         The full C3 nn.Sequential (ResNet18 + Sigmoid).
            input_tensor:  (1, 3, 64, 64) torch tensor.
            norm_crop:     (64, 64, 3) float32 [0,1] — the original image for display.

        Returns:
            overlay (np.ndarray uint8 HxWx3): coloured LIME superpixel overlay.
            Returns None if LIME is unavailable or fails.
        """
        if not self.available:
            return None

        try:
            import torch
            from skimage.color import gray2rgb

            # LIME needs a predict_fn that takes a batch of uint8 (H,W,3) images
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            device = next(model.parameters()).device

            def predict_fn(batch_uint8):
                imgs = batch_uint8.astype(np.float32) / 255.0           # (N,H,W,C)
                imgs = (imgs - mean) / std                               # normalise
                t    = torch.tensor(imgs.transpose(0,3,1,2), dtype=torch.float32).to(device)
                with torch.no_grad():
                    out = model(t).cpu().numpy()                          # (N,1)
                return np.hstack([out, 1.0 - out])                       # fake 2-class

            # Original image in uint8 RGB
            img_uint8 = (norm_crop * 255).astype(np.uint8)

            explanation = self.explainer.explain_instance(
                img_uint8,
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=self.N_SAMPLES,
                batch_size=self.BATCH_SIZE,
            )

            # Get image + mask for the top label
            label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                label,
                positive_only=True,
                num_features=self.N_FEATURES,
                hide_rest=False,
            )

            # Colour the highlighted superpixels green
            overlay = img_uint8.copy()
            overlay[mask == 1] = np.clip(
                overlay[mask == 1] * np.array([0.3, 1.0, 0.3], dtype=np.float32), 0, 255
            ).astype(np.uint8)

            return overlay

        except Exception as e:
            print(f"⚠️ LIME explain failed: {e}")
            return None


# ---------------------------------------------------------------------------
# SHAP Explainer
# ---------------------------------------------------------------------------
class ShapExplainer:
    """
    [6.10] SHAP DeepExplainer — pixel-attribution XAI.
    Uses Shapley values to assign credit/blame to each pixel patch.

    Produces a signed heatmap (positive = increases predicted angle,
    negative = decreases predicted angle) rendered as a coloured overlay.
    """

    N_BACKGROUND = 20   # Background samples for DeepExplainer

    def __init__(self):
        try:
            import shap as _shap
            self.shap = _shap
            self.available = True
        except ImportError:
            self.available = False
            print("⚠️ SHAP not installed. Run: pip install shap")

    def explain(self, model, input_tensor: "torch.Tensor", background: "torch.Tensor") -> np.ndarray:
        """
        Args:
            model:          Full C3 nn.Sequential.
            input_tensor:   (1, 3, 64, 64) input torch tensor.
            background:     (N, 3, 64, 64) background tensor (random noise or training samples).
                            If None, a zero background is used.

        Returns:
            overlay (np.ndarray uint8 HxWx3): SHAP attribution heatmap overlay.
            Returns None if SHAP unavailable or fails.
        """
        if not self.available:
            return None

        try:
            import torch
            device = next(model.parameters()).device

            if background is None:
                background = torch.zeros(
                    self.N_BACKGROUND, *input_tensor.shape[1:], device=device
                )

            # DeepExplainer operates on the raw ResNet18 (before Sigmoid)
            resnet = model[0] if hasattr(model, '__getitem__') else model
            resnet.eval()

            explainer = self.shap.DeepExplainer(resnet, background)
            shap_vals  = explainer.shap_values(input_tensor)   # (1, 3, 64, 64) or list

            # Sum absolute SHAP values across colour channels → (64, 64)
            if isinstance(shap_vals, list):
                shap_map = np.abs(shap_vals[0]).sum(axis=1)[0]    # (H,W)
            else:
                shap_map = np.abs(shap_vals).sum(axis=1)[0]       # (H,W)

            # Normalise to [0, 1]
            s_min, s_max = shap_map.min(), shap_map.max()
            if s_max > s_min:
                shap_map = (shap_map - s_min) / (s_max - s_min)

            # Apply a red-blue colourmap (blue = low, red = high attribution)
            shap_uint8 = (shap_map * 255).astype(np.uint8)
            coloured   = cv2.applyColorMap(shap_uint8, cv2.COLORMAP_JET)
            coloured_rgb = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
            return coloured_rgb

        except Exception as e:
            print(f"⚠️ SHAP explain failed: {e}")
            return None
