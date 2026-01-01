from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import cv2

class XaiVisualizer:
    def __init__(self, model):
        self.model = model
        # Target the last ResNet layer
        self.target_layers = [model.layer4[-1]]
        self.cam = GradCAM(model=model, target_layers=self.target_layers)

    def generate(self, input_tensor, rgb_img_normalized):
        """
        Generates a sharp Grad-CAM heatmap.
        """
        # 1. Generate Raw Heatmap
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # 2. Sharpening Logic (The Fix)
        # We clip low values to remove the "Cyan Fog" background noise
        grayscale_cam[grayscale_cam < 0.3] = 0  # Remove weak activations
        
        # Re-normalize to 0-1 range after clipping
        if np.max(grayscale_cam) > 0:
            grayscale_cam = grayscale_cam / np.max(grayscale_cam)

        # 3. Overlay on Image
        visualization = show_cam_on_image(rgb_img_normalized, grayscale_cam, use_rgb=True)
        return visualization