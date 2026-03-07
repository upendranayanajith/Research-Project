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
    print(f"[C1] Model load failed: {e}")


def _resize_small(img, size=500):
    h, w = img.shape[:2]
    if max(h, w) <= size:
        return img  
    scale = size / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


def _enhance_image(img):
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


# def _deskew_clock(crop):
#     """Detect tilt angle via contour fitting and rotate crop to upright."""
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return crop
#     largest = max(contours, key=cv2.contourArea)
#     rect = cv2.minAreaRect(largest)
#     angle = rect[2]
#     # Normalize: minAreaRect returns angle in (-90, 0]
#     if angle < -45:
#         angle = 90 + angle
#     if abs(angle) < 2.0:  # Skip tiny corrections
#         return crop
#     h, w = crop.shape[:2]
#     M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
#     rotated = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
#     return rotated


def _compute_quality_metrics(img):
    """Compute per-metric image quality scores: blur, brightness, contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur score — Laplacian variance (higher = sharper image)
    blur_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = round(min(100.0, (blur_raw / 800.0) * 100.0), 1)  # 800+ var → score 100

    # Brightness score — mean pixel value; ideal near 128
    brightness_raw = float(np.mean(gray))
    brightness_score = round(max(0.0, 100.0 - abs(brightness_raw - 128.0) / 1.28), 1)

    # Contrast score — std deviation of pixel values (higher = richer contrast)
    contrast_raw = float(np.std(gray))
    contrast_score = round(min(100.0, (contrast_raw / 80.0) * 100.0), 1)  # 80+ std → score 100

    # Overall quality: weighted composite
    overall = round(0.40 * blur_score + 0.30 * brightness_score + 0.30 * contrast_score, 1)

    return {
        "blur_score": blur_score,
        "blur_raw": round(blur_raw, 2),
        "brightness_score": brightness_score,
        "brightness_raw": round(brightness_raw, 1),
        "contrast_score": contrast_score,
        "contrast_raw": round(contrast_raw, 1),
        "overall_quality": overall,
    }


def _hough_circle_validation(crop):
    """Use Hough Circle Transform to validate that the detected region is a clock face."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape
    min_dim = min(h, w)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim // 3,
        param1=80,
        param2=35,
        minRadius=max(10, min_dim // 6),
        maxRadius=min_dim // 2,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        best = circles[0]
        cx, cy, r = int(best[0]), int(best[1]), int(best[2])
        # Center alignment score (100 = perfectly centred)
        center_dist = float(np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2))
        max_off = float(np.sqrt((w / 2) ** 2 + (h / 2) ** 2))
        center_score = round(max(0.0, 100.0 * (1.0 - center_dist / max_off)), 1)
        # Radius coverage: what fraction of the crop the circle fills
        coverage_score = round(min(100.0, (2 * r / min_dim) * 100.0), 1)
        return {
            "validated": True,
            "circle_count": int(len(circles)),
            "best_circle": {"cx": cx, "cy": cy, "radius": r},
            "center_score": center_score,
            "coverage_score": coverage_score,
        }
    return {
        "validated": False,
        "circle_count": 0,
        "best_circle": None,
        "center_score": 0.0,
        "coverage_score": 0.0,
    }


def _draw_hough_overlay(img, hough_data):
    """Draw detected Hough circle on a copy of img."""
    vis = img.copy()
    if hough_data["validated"] and hough_data["best_circle"]:
        bc = hough_data["best_circle"]
        cv2.circle(vis, (bc["cx"], bc["cy"]), bc["radius"], (0, 255, 0), 2)
        cv2.circle(vis, (bc["cx"], bc["cy"]), 4, (0, 0, 255), -1)
        label = f"Hough r={bc['radius']}px"
        cv2.putText(vis, label, (bc["cx"] - bc["radius"], bc["cy"] - bc["radius"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    else:
        cv2.putText(vis, "No circle found", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis


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
    confidence = round(float(best_box.conf[0]), 4)
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    h, w = img.shape[:2]
    pad = 30
    x1_p, y1_p = max(0, x1 - pad), max(0, y1 - pad)
    x2_p, y2_p = min(w, x2 + pad), min(h, y2 + pad)

    raw_crop = img[y1_p:y2_p, x1_p:x2_p]

    # --- Analysis on RAW crop (before any processing) ---
    quality_metrics = _compute_quality_metrics(raw_crop)
    hough_data = _hough_circle_validation(raw_crop)

    # Step 1: Deskew (correct tilt) — temporarily disabled
    # cropped = _deskew_clock(raw_crop)
    cropped = raw_crop

    # Step 2: Enhance image quality
    cropped = _enhance_image(cropped)

    # --- Hough overlay on the enhanced/processed crop ---
    hough_viz = _draw_hough_overlay(cropped, hough_data)

    # Confidence label colour: green ≥ 0.80, orange ≥ 0.60, red < 0.60
    if confidence >= 0.80:
        conf_color = (0, 220, 0)
        conf_label = "HIGH"
    elif confidence >= 0.60:
        conf_color = (0, 165, 255)
        conf_label = "MED"
    else:
        conf_color = (0, 0, 220)
        conf_label = "LOW"

    # Visualization with bounding box + confidence annotation
    viz = img.copy()
    cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(viz, f"Clock  conf:{confidence:.2f} [{conf_label}]", (x1, max(y1 - 10, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, conf_color, 2)
    # Quality badge in bottom-left corner of viz
    q = quality_metrics["overall_quality"]
    cv2.putText(viz, f"Quality: {q:.0f}/100", (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    return {
        "found": True,
        "bbox": [x1_p, y1_p, x2_p, y2_p],
        "confidence": confidence,
        "quality": quality_metrics,
        "hough_validation": hough_data,
        "cropped_image": _encode_image(cropped),
        "hough_visualization": _encode_image(_resize_small(hough_viz, size=300)),
        "visualization": _encode_image(_resize_small(viz)),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
