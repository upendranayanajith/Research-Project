import cv2
import numpy as np
from ultralytics import YOLO

class ClockLocalizer:
    def __init__(self, model_path, use_enhancer=True):
        print(f"[C1] Loading Localization Model: {model_path}")
        self.model = YOLO(model_path)
        self.use_enhancer = use_enhancer

