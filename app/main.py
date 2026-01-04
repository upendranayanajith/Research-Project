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
from app.core.engine import ClockEngine
from app.core.metrics import metrics_tracker

app = FastAPI(title="Clock AI Research - Multi-Stage Visualization")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
engine = ClockEngine(BASE_DIR)

@app.post("/analyze")
async def analyze_clock(file: UploadFile = File(...), force_expert: bool = Form(False)):
    start_time = time.time()
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = engine.analyze(img, force_expert=force_expert)
    
    processing_time = time.time() - start_time
    
    # DB Record
    metrics_tracker.record_analysis(result, processing_time, file.filename)
    
    if "error" in result:
        return {"result": result, "processing_time": processing_time}
    
    # Handle Visualization Images
    viz_base64 = {}
    if "visualizations" in result:
        for stage_name, stage_img in result["visualizations"].items():
            if stage_name == "c3_crops":
                crops_b64 = []
                for crop in stage_img:
                    _, buffer = cv2.imencode('.jpg', crop)
                    crops_b64.append(base64.b64encode(buffer).decode('utf-8'))
                viz_base64[stage_name] = crops_b64
            else:
                _, buffer = cv2.imencode('.jpg', stage_img)
                viz_base64[stage_name] = base64.b64encode(buffer).decode('utf-8')
        result.pop("visualizations", None)
    
    heatmap_b64 = None
    if result.get("heatmap") is not None:
        _, buffer = cv2.imencode('.jpg', (result["heatmap"] * 255).astype(np.uint8))
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        result["heatmap"] = None
    
    return {
        "result": result,
        "visualizations": viz_base64,
        "heatmap_b64": heatmap_b64,
        "processing_time": processing_time
    }

@app.post("/analyze_batch")
async def analyze_batch(files: List[UploadFile] = File(...), force_expert: bool = Form(False)):
    results = []
    for file in files:
        try:
            start_time = time.time()
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            result = engine.analyze(img, force_expert=force_expert)
            processing_time = time.time() - start_time
            
            metrics_tracker.record_analysis(result, processing_time, file.filename)
            
            results.append({
                "filename": file.filename,
                "success": "error" not in result,
                "time": result.get("time", "N/A"),
                "method": result.get("method", "Unknown"),
                "processing_time": processing_time
            })
        except Exception as e:
            results.append({"filename": file.filename, "success": False, "error": str(e)})
    
    return {"total_images": len(files), "results": results}

@app.get("/metrics")
async def get_metrics():
    return metrics_tracker.get_metrics()

@app.get("/metrics/history")
async def get_metrics_history():
    """New endpoint to fetch raw rows for table"""
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