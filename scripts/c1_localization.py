import cv2
import numpy as np
from ultralytics import YOLO

class ClockLocalizer:
    def __init__(self, model_path, use_enhancer=True):
        print(f"[C1] Loading Localization Model: {model_path}")
        self.model = YOLO(model_path)
        self.use_enhancer = use_enhancer

    def process_input(self, image):
        """
        Accepts ANY image (Video Frame or Static File).
        Returns: The straightened clock image or None.
        """
        results = self.model(image, verbose=False)[0]

        if not results.keypoints or len(results.keypoints) == 0:
            return None

        kpts = results.keypoints.xy[0].cpu().numpy()
