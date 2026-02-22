"""
C2 Skeleton Service — Hand Keypoint Extraction via YOLO-Pose
Runs on port 8002
Owner: Member 2
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import math
import os
from ultralytics import YOLO

app = FastAPI(title="C2 - Hand Skeleton Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Model Loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

c2_model = None
try:
    print(f"[C2] Loading YOLO-Pose model: {MODEL_PATH}")
    c2_model = YOLO(MODEL_PATH)
    print("[C2] Model loaded successfully.")
except Exception as e:
    print(f"[C2] ⚠️ Model load failed: {e}")


def _resize_small(img, size=500):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


def _get_angle(center, point):
    """Calculate angle from 12-o'clock position (clockwise)."""
    dx, dy = point[0] - center[0], point[1] - center[1]
    angle = math.degrees(math.atan2(dx, -dy))
    return angle + 360 if angle < 0 else angle


def _draw_skeleton(img, center, tip1, tip2):
    """Draw skeleton overlay on the image."""
    img_copy = img.copy()
    center_pt = (int(center[0]), int(center[1]))
    tip1_pt = (int(tip1[0]), int(tip1[1]))
    tip2_pt = (int(tip2[0]), int(tip2[1]))

    cv2.line(img_copy, center_pt, tip1_pt, (0, 255, 0), 4)
    cv2.line(img_copy, center_pt, tip2_pt, (0, 0, 255), 4)
    cv2.circle(img_copy, center_pt, 8, (255, 0, 0), -1)
    cv2.circle(img_copy, tip1_pt, 8, (0, 255, 0), -1)
    cv2.circle(img_copy, tip2_pt, 8, (0, 0, 255), -1)

    return _resize_small(img_copy)


@app.get("/health")
async def health():
    return {"service": "C2-Skeleton", "status": "ok", "model_loaded": c2_model is not None}


@app.post("/extract-skeleton")
async def extract_skeleton(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if c2_model is None:
        return {"error": "C2 model not loaded"}

    results = c2_model(img, verbose=False)[0]

    if not results.keypoints or len(results.keypoints.data) == 0:
        return {"error": "No hands/keypoints detected"}

    kpts = results.keypoints.data[0].cpu().numpy()
    center = kpts[0][:2].tolist()
    tip1 = kpts[1][:2].tolist()
    tip2 = kpts[2][:2].tolist()

    # Calculate geometric angles
    angle1 = _get_angle(center, tip1)
    angle2 = _get_angle(center, tip2)

    # Generate visualization
    viz = _draw_skeleton(img, center, tip1, tip2)

    return {
        "keypoints": {
            "center": center,
            "tip1": tip1,
            "tip2": tip2
        },
        "angles": {
            "hand1": round(angle1, 2),
            "hand2": round(angle2, 2)
        },
        "visualization": _encode_image(viz)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
