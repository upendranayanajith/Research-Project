from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import base64
import os
from app.core.engine import ClockEngine

# Initialize App & Engine
app = FastAPI(title="Clock AI Research")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
engine = ClockEngine(BASE_DIR)

@app.post("/analyze")
async def analyze_clock(file: UploadFile = File(...), force_expert: bool = Form(False)):
    # 1. Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Run Engine
    result = engine.analyze(img, force_expert=force_expert)
    
    # 3. Handle Heatmap Image (Convert to Base64 for JSON transfer)
    heatmap_b64 = None
    if result.get("heatmap") is not None:
        _, buffer = cv2.imencode('.jpg', (result["heatmap"] * 255).astype(np.uint8))
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        result["heatmap"] = None # Remove raw array from JSON
    
    return {
        "result": result,
        "heatmap_b64": heatmap_b64
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)