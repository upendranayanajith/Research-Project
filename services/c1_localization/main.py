"""
C1 Localization Service — Clock Detection via YOLO
Runs on port 8001
Owner: Member 1
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import os
from ultralytics import YOLO

app = FastAPI(title="C1 - Clock Localization Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Model Loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

c1_model = None
try:
    print(f"[C1] Loading YOLO model: {MODEL_PATH}")
    c1_model = YOLO(MODEL_PATH)
    print("[C1] Model loaded successfully.")
except Exception as e:
    print(f"[C1] ⚠️ Model load failed: {e}")


def _resize_small(img, size=500):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


def _enhance_image(img):
    """CLAHE contrast enhancement + unsharp masking sharpening."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    # Unsharp masking
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    return sharpened


def _deskew_clock(crop):
    """Detect tilt angle via contour fitting and rotate crop to upright."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[2]
    # Normalize: minAreaRect returns angle in (-90, 0]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 2.0:  # Skip tiny corrections
        return crop
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


@app.get("/health")
async def health():
    return {"service": "C1-Localization", "status": "ok", "model_loaded": c1_model is not None}


@app.post("/localize")
async def localize(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if c1_model is None:
        return {
            "found": False,
            "bbox": None,
            "cropped_image": _encode_image(img),
            "visualization": _encode_image(_resize_small(img)),
            "error": "C1 model not loaded"
        }

    results = c1_model(img, verbose=False)[0]

    if len(results.boxes) == 0:
        return {
            "found": False,
            "bbox": None,
            "cropped_image": _encode_image(img),
            "visualization": _encode_image(_resize_small(img)),
            "error": "No clock face detected in the uploaded image. Please upload a clear photo of an analog clock."
        }

    # Best detection
    best_box = results.boxes[0]
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    h, w = img.shape[:2]
    pad = 30
    x1_p, y1_p = max(0, x1 - pad), max(0, y1 - pad)
    x2_p, y2_p = min(w, x2 + pad), min(h, y2 + pad)

    cropped = img[y1_p:y2_p, x1_p:x2_p]

    # Step 1: Deskew (correct tilt)
    cropped = _deskew_clock(cropped)

    # Step 2: Enhance image quality
    cropped = _enhance_image(cropped)

    # Visualization with bounding box
    viz = img.copy()
    cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(viz, "Clock Detected", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return {
        "found": True,
        "bbox": [x1_p, y1_p, x2_p, y2_p],
        "cropped_image": _encode_image(cropped),
        "visualization": _encode_image(_resize_small(viz))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
