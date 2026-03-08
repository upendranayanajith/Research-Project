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