from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import base64
import os
import time
from typing import List
from app.core.engine import HARPEngine 
from app.core.metrics import metrics_tracker

# --- [C4] API SETUP ---
app = FastAPI(title="HARP Research - Multi-Model Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize Engine (Default to Clock mode, but it can switch dynamically)
engine = HARPEngine(BASE_DIR, mode='clock')

# --- [C4] MAIN ANALYSIS ENDPOINT ---
@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...), 
    force_expert: bool = Form(False),
    mode: str = Form("clock") # New Parameter: 'clock' or 'gauge'
):
    start_time = time.time()
    
    # 1. Decode Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "result": {"error": "Invalid or corrupted image file."},
            "processing_time": time.time() - start_time
        }


    # 2. Switch Engine Mode Dynamically
    if engine.mode != mode:
        engine.set_mode(mode)

    # 3. Call the Engine
    result = engine.analyze(img, force_expert=force_expert)
    
    processing_time = time.time() - start_time
    
    # [C4] Log to Database
    metrics_tracker.record_analysis(result, processing_time, file.filename)
    
    if "error" in result and result["error"]:
        return {"result": result, "processing_time": processing_time}
    
    # [C4] Process Visualizations for API Response
    viz_base64 = {}
    if "visualizations" in result:
        for stage_name, stage_val in result["visualizations"].items():
            # Handle list of crops (C3 output)
            if isinstance(stage_val, list):
                crops_b64 = []
                for crop in stage_val:
                    if isinstance(crop, np.ndarray) and crop.size > 0:
                        _, buffer = cv2.imencode('.jpg', crop)
                        crops_b64.append(base64.b64encode(buffer).decode('utf-8'))
                viz_base64[stage_name] = crops_b64
            
            # Handle single image (C1/C2 output)
            elif isinstance(stage_val, np.ndarray) and stage_val.size > 0:
                _, buffer = cv2.imencode('.jpg', stage_val)
                viz_base64[stage_name] = base64.b64encode(buffer).decode('utf-8')

        # Remove raw arrays from result to make it JSON serializable
        result.pop("visualizations", None)
    
    # Handle Heatmap (C3 XAI)
    heatmap_b64 = None
    if result.get("heatmap") is not None:
        # Normalize 0-1 float to 0-255 uint8
        heatmap_uint8 = (result["heatmap"] * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.jpg', heatmap_uint8)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        result["heatmap"] = None
    
    return {
        "result": result,
        "visualizations": viz_base64,
        "heatmap_b64": heatmap_b64,
        "processing_time": processing_time
    }

# --- [C4] BATCH PROCESSING ---
@app.post("/analyze_batch")
async def analyze_batch(
    files: List[UploadFile] = File(...), 
    force_expert: bool = Form(False),
    mode: str = Form("clock")
):
    results = []
    
    # Ensure correct mode for the batch
    if engine.mode != mode:
        engine.set_mode(mode)

    for file in files:
        try:
            start_time = time.time()
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Invalid or corrupted image file.")
                
            result = engine.analyze(img, force_expert=force_expert)
            processing_time = time.time() - start_time
            
            metrics_tracker.record_analysis(result, processing_time, file.filename)
            
            results.append({
                "filename": file.filename,
                "success": "error" not in result or not result["error"],
                "value": result.get("time", "N/A"), # 'time' key holds both Time and Gauge %
                "method": result.get("method", "Unknown"),
                "processing_time": processing_time
            })
        except Exception as e:
            results.append({"filename": file.filename, "success": False, "error": str(e)})
    
    return {"total_images": len(files), "mode": mode, "results": results}

# --- [C4] METRICS ENDPOINTS ---
@app.get("/metrics")
async def get_metrics():
    return metrics_tracker.get_metrics()

@app.get("/metrics/history")
async def get_metrics_history():
    return metrics_tracker.get_history(limit=50)

@app.get("/metrics/export")
async def export_metrics():
    csv_data = metrics_tracker.export_to_csv()
    return PlainTextResponse(content=csv_data, media_type="text/csv")

@app.post("/metrics/clear")
async def clear_metrics():
    metrics_tracker.clear_metrics()
    return {"message": "Metrics cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    