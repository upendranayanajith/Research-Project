"""
C3 Angle Refinement Service — ResNet18 + Grad-CAM Expert Refinement
Runs on port 8003
Owner: Member 3
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
import base64
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
from services.c3_angle_refinement.xai import XaiVisualizer

app = FastAPI(title="C3 - Angle Refinement Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Device & Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pth")


def _build_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = nn.Sequential(model, nn.Sigmoid())
    return model


c3_model = None
xai_viz = None
try:
    print(f"[C3] Loading ResNet18 model: {MODEL_PATH}")
    c3_model = _build_model().to(device)
    if os.path.exists(MODEL_PATH):
        c3_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        c3_model.eval()
        xai_viz = XaiVisualizer(c3_model[0])
        print("[C3] Model loaded successfully.")
    else:
        print("[C3] ⚠️ Weights file not found.")
        c3_model = None
except Exception as e:
    print(f"[C3] ⚠️ Model load failed: {e}")

# Preprocessing
c3_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def _decode_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


def _get_crop(img, center, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    s = 128 // 2
    y1, y2 = int(center[1] - s), int(center[1] + s)
    x1, x2 = int(center[0] - s), int(center[0] + s)
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return np.array([])
    return rotated[y1:y2, x1:x2]


def _resize_small(img, size=500):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _draw_angles_on_img(img, center, tip1, tip2, a1, a2):
    img_copy = img.copy()
    center_pt = (int(center[0]), int(center[1]))
    tip1_pt = (int(tip1[0]), int(tip1[1]))
    tip2_pt = (int(tip2[0]), int(tip2[1]))

    cv2.line(img_copy, center_pt, tip1_pt, (0, 255, 0), 4)
    cv2.line(img_copy, center_pt, tip2_pt, (0, 0, 255), 4)
    cv2.circle(img_copy, center_pt, 8, (255, 0, 0), -1)
    cv2.circle(img_copy, tip1_pt, 8, (0, 255, 0), -1)
    cv2.circle(img_copy, tip2_pt, 8, (0, 0, 255), -1)

    cv2.putText(img_copy, f"H: {a1:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img_copy, f"M: {a2:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return _resize_small(img_copy)


# --- Request Model ---
class RefineRequest(BaseModel):
    image: str  # base64 of cropped clock
    keypoints: dict  # {"center": [x,y], "tip1": [x,y], "tip2": [x,y]}
    rough_angles: dict  # {"hand1": float, "hand2": float}


@app.get("/health")
async def health():
    return {"service": "C3-AngleRefinement", "status": "ok", "model_loaded": c3_model is not None}


@app.post("/refine-angles")
async def refine_angles(req: RefineRequest):
    if c3_model is None:
        return {
            "refined_angles": req.rough_angles,
            "crops": [],
            "heatmap": None,
            "angle_visualization": None,
            "debug": ["C3 model not loaded — returning rough angles"],
            "refined": False
        }

    img = _decode_image(req.image)
    center = req.keypoints["center"]
    tip1 = req.keypoints["tip1"]
    tip2 = req.keypoints["tip2"]
    a1 = req.rough_angles["hand1"]
    a2 = req.rough_angles["hand2"]

    refined_angles = []
    debug_info = []
    heatmap_b64 = None
    crop_images = []

    tips = [tip1, tip2]
    rough = [a1, a2]

    for i, (tip, rough_angle) in enumerate(zip(tips, rough)):
        crop = _get_crop(img, center, rough_angle)
        if crop.size == 0:
            refined_angles.append(rough_angle)
            debug_info.append(f"Hand {i}: Crop failed, keeping rough angle")
            continue

        crop_images.append(_encode_image(crop))

        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pil_resized = pil_crop.resize((64, 64))
        t_input = c3_transform(pil_resized).unsqueeze(0).to(device)

        # Generate XAI heatmap for first hand only
        if heatmap_b64 is None and xai_viz is not None:
            norm_crop = np.array(pil_resized, dtype=np.float32) / 255.0
            heatmap_img = xai_viz.generate(t_input, norm_crop)
            heatmap_b64 = _encode_image(heatmap_img)

        with torch.no_grad():
            pred = c3_model(t_input).item()

        c3_angle = pred * 360.0
        delta = c3_angle - 360 if c3_angle > 180 else c3_angle

        if abs(delta) > 20.0:
            debug_info.append(f"Hand {i}: Rejected C3 delta {delta:.1f}°")
            refined_angles.append(rough_angle)
        else:
            debug_info.append(f"Hand {i}: Accepted C3 delta {delta:.1f}°")
            refined_angles.append((rough_angle + delta) % 360)

    # Generate angle visualization
    angle_viz = _draw_angles_on_img(img, center, tip1, tip2, refined_angles[0], refined_angles[1])

    return {
        "refined_angles": {
            "hand1": round(refined_angles[0], 2),
            "hand2": round(refined_angles[1], 2)
        },
        "crops": crop_images,
        "heatmap": heatmap_b64,
        "angle_visualization": _encode_image(angle_viz),
        "debug": debug_info,
        "refined": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
