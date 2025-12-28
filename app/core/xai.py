from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import cv2

class XaiVisualizer:
    def __init__(self, model):
        self.model = model
        # Target the last layer of ResNet18
        self.target_layers = [model.layer4[-1]]
        self.cam = GradCAM(model=model, target_layers=self.target_layers)

    def generate(self, input_tensor, rgb_img_normalized):
        """
        input_tensor: (1, 3, 64, 64) torch tensor
        rgb_img_normalized: (64, 64, 3) numpy array float32 (0-1)
        """
        # Since we do regression, we don't have classes. 
        # We target the output itself.
        targets = [ClassifierOutputTarget(0)] 
        
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        visualization = show_cam_on_image(rgb_img_normalized, grayscale_cam, use_rgb=True)
        return visualization