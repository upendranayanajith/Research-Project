import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class XaiVisualizer:
    def __init__(self, model):
        """
        Initialize XAI visualizer with Grad-CAM
        model: The feature extractor (e.g., ResNet base)
        """
        self.model = model
        # Target the last convolutional layer
        target_layers = [model.layer4[-1]]
        self.cam = GradCAM(model=model, target_layers=target_layers)
    
    def generate(self, input_tensor, original_image):
        """
        Generate Grad-CAM heatmap
        input_tensor: Preprocessed tensor input
        original_image: Original image (normalized 0-1)
        Returns: Heatmap overlaid on original image
        """
        # Generate CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay on image
        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
        
        return visualization
