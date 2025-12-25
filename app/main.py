from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import os
import io

# Import our core modules
from app.core.utils import calculate_angle, get_aligned_crop
from app.core.c4_inference import TimeInferenceEngine

app = FastAPI(title="Clock Reader Research API")

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C2_MODEL_PATH = os.path.join(BASE_DIR, "models", "c2_hands_skeleton", "best.pt")
C3_MODEL_PATH = os.path.join(BASE_DIR, "models", "c3_angle_regression", "best.pth")

# --- MODEL LOADING ---
print("🚀 Loading Models...")

# 1. Load C2 (YOLO)
c2_model = YOLO(C2_MODEL_PATH)

# 2. Load C3 (ResNet Regression)
# We must redefine the architecture to load weights
def get_c3_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = nn.Sequential(model, nn.Sigmoid())
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
c3_model = get_c3_model().to(device)

if os.path.exists(C3_MODEL_PATH):
    c3_model.load_state_dict(torch.load(C3_MODEL_PATH, map_location=device))
    c3_model.eval()
    print("✅ C3 Model Loaded")
else:
    print("⚠️ WARNING: C3 Model not found at path!")

# 3. Load C4 (Reasoning Engine)
c4_engine = TimeInferenceEngine()

# C3 Preprocessing Transforms
c3_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

@app.post("/analyze")
async def analyze_clock(file: UploadFile = File(...)):
    # 1. READ IMAGE
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(content={"error": "Invalid image"}, status_code=400)

    # 2. RUN C2 (Skeleton Detection)
    results = c2_model(img, verbose=False)[0]
    
    if results.keypoints is None or len(results.keypoints.data) == 0:
        return JSONResponse(content={"error": "No clock hands found by C2"}, status_code=400)

    kpts = results.keypoints.data[0].cpu().numpy()
    
    # Extract Center and Tips
    points = [(kpts[i][0], kpts[i][1]) for i in range(3)]
    
    # Simple heuristic: Center is the point closest to image center
    h_img, w_img = img.shape[:2]
    img_center = (w_img/2, h_img/2)
    dists_to_center = [dist(p, img_center) for p in points]
    center_idx = np.argmin(dists_to_center)
    center_pt = points[center_idx]
    
    tips = [points[i] for i in range(3) if i != center_idx]
    if len(tips) < 2:
        return JSONResponse(content={"error": "Could not identify two hands"}, status_code=400)

    # 3. RUN C3 (Angle Regression) for both hands
    raw_angles = []
    
    for tip in tips:
        # A. Get Rough Angle (Geometry)
        rough_angle = calculate_angle(center_pt, tip)
        
        # B. Get Normalized Crop
        crop = get_aligned_crop(img, center_pt, rough_angle)
        
        # C. Refine with AI
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = c3_transform(crop_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = c3_model(input_tensor).item() # 0.0 to 1.0
            
        c3_adjustment = prediction * 360
        final_angle = (rough_angle + c3_adjustment) % 360
        raw_angles.append(final_angle)

    # 4. RUN C4 (Reasoning)
    # We pass the two angles. C4 figures out which is which.
    decision = c4_engine.analyze(raw_angles[0], raw_angles[1])
    
    return decision

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)