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

# Initialize Engine
engine = HARPEngine(BASE_DIR)

# --- [C4] MAIN ANALYSIS ENDPOINT ---
@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...), 
    force_expert: bool = Form(False),
    manual_min_val: str = Form(""),
    manual_max_val: str = Form(""),
    device_time_str: str = Form(None)
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


    # 2. Call the Engine
    result = engine.analyze(
        img, 
        force_expert=force_expert, 
        manual_min_val=manual_min_val, 
        manual_max_val=manual_max_val,
        device_time_str=device_time_str
    )
    
    processing_time = time.time() - start_time

    # [FIX-7] Extract C3 angle correction magnitude for tracking
    angles_info = result.get("angles", {})
    correction_deg = None
    if "hand1" in angles_info and result.get("method", "").startswith("Expert"):
        # Compute from uncertainty_deg if available, else leave None
        uncertainty_str = result.get("uncertainty_deg", "")
        if uncertainty_str and uncertainty_str != "N/A":
            try:
                # Parse first hand uncertainty e.g. "H1=±3.2°, H2=±1.1°"
                first_part = uncertainty_str.split(",")[0]
                correction_deg = float(first_part.split("±")[1].replace("°", "").strip())
            except Exception:
                pass

    # [C4] Log to Database
    metrics_tracker.record_analysis(result, processing_time, file.filename, correction_deg)
    
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

# --- COMPARATOR ENDPOINT ---
@app.post("/compare_times")
async def compare_times(
    file_before: UploadFile = File(...),
    file_after: UploadFile = File(...)
):
    start_time = time.time()
    
    async def process_file(upload_file):
        contents = await upload_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("Invalid image")
        res = engine.analyze(img, force_expert=False)
        return res
        
    try:
        res_before = await process_file(file_before)
        res_after = await process_file(file_after)
        
        if res_before.get("error") or res_after.get("error"):
            return JSONResponse(status_code=400, content={"error": "Failed to analyze one or both images."})
            
        tb = res_before.get("time", "")
        ta = res_after.get("time", "")
        
        if ":" not in tb or ":" not in ta:
            return JSONResponse(status_code=400, content={"error": "Both images must be analog clocks."})
            
        hb, mb = map(int, tb.split(":"))
        ha, ma = map(int, ta.split(":"))
        
        min_b = (hb % 12) * 60 + mb
        min_a = (ha % 12) * 60 + ma
        
        diff = min_a - min_b
        if diff < 0:
            diff += 720
            
        elapsed_h = diff // 60
        elapsed_m = diff % 60
        
        return {
            "time_before": tb,
            "time_after": ta,
            "elapsed_minutes": diff,
            "elapsed_text": f"From {tb} to {ta} → {diff} minutes elapsed ({elapsed_h}h {elapsed_m}m)",
            "processing_time": time.time() - start_time
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- [C4] BATCH PROCESSING ---
@app.post("/analyze_batch")
async def analyze_batch(
    files: List[UploadFile] = File(...), 
    force_expert: bool = Form(False),
    manual_min_val: str = Form(""),
    manual_max_val: str = Form("")
):
    results = []

    for file in files:
        try:
            start_time = time.time()
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Invalid or corrupted image file.")
                
            result = engine.analyze(
                img, 
                force_expert=force_expert,
                manual_min_val=manual_min_val, 
                manual_max_val=manual_max_val
            )
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
    
    return {"total_images": len(files), "results": results}

# --- [C4] METRICS ENDPOINTS ---
@app.get("/metrics")
async def get_metrics():
    return metrics_tracker.get_metrics()

@app.get("/metrics/c3")
async def get_c3_metrics():
    """[FIX-8] C3-specific performance statistics: trigger rate and avg correction."""
    return metrics_tracker.get_c3_stats()

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
    