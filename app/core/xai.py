import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import google.generativeai as genai
import os
from PIL import Image

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
        """
        # Generate CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay on image
        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
        
        return visualization

class SemanticExplainer:
    def __init__(self):
        # --- API KEY CONFIGURATION ---
        self.api_key = "AIzaSyANdJ_TYftpxzYYAyyqbqtvKbYcqs-zM3c"
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.available = True
            except Exception as e:
                print(f"AI Init Failed: {e}")
                self.available = False
        else:
            self.available = False

    def explain(self, crop_img, heatmap_img, predicted_angle, hand_type="Hour"):
        """
        Sends the images to the AI to interpret the model's focus.
        """
        if not self.available:
            return "AI Explanation unavailable (API Key missing)."

        # Convert arrays to PIL Images if they aren't already
        if isinstance(crop_img, np.ndarray):
            crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        if isinstance(heatmap_img, np.ndarray):
            heatmap_img = Image.fromarray(heatmap_img)

        prompt = (
            f"You are an expert Computer Vision debugger. "
            f"The first image is a crop of a clock hand. The second is a Grad-CAM heatmap "
            f"showing where the ResNet model looked to predict the {hand_type} angle. "
            f"The predicted angle was {predicted_angle:.1f} degrees. "
            f"Briefly explain (in 1 sentence) if the model is focusing on the correct visual features "
            f"(like the hand tip or edges) to make this prediction."
        )

        try:
            response = self.model.generate_content([prompt, crop_img, heatmap_img])
            return response.text
        except Exception as e:
            return f"Explanation failed: {str(e)}"