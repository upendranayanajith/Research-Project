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

def sanitize_result(obj):
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_result(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_result(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return None # Should have been handled/encoded already
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int8, np.int16, np.uint8, np.uint16, np.uint32)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

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
    
    # [C4] Process Visualizations for API Response (Must happen before return to prevent 500 error)
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

        # Remove raw arrays from result
        result.pop("visualizations", None)
    
    # Handle Heatmap (C3 XAI)
    heatmap_b64 = None
    if result.get("heatmap") is not None:
        if isinstance(result["heatmap"], np.ndarray):
            heatmap_uint8 = (result["heatmap"] * 255).astype(np.uint8)
            _, buffer = cv2.imencode('.jpg', heatmap_uint8)
            heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        result.pop("heatmap", None) # Always remove raw array/none
    
    # 3. Final Sanitization (Deep convert numpy types to python types)
    clean_result = sanitize_result(result)
    
    # Handle C2 Research images (convert numpy arrays to base64)
    c2_research_b64 = None
    if result.get("c2_research"):
        c2_research_b64 = _serialize_c2_research(result.pop("c2_research"))
    
    return {
        "result": clean_result,
        "visualizations": viz_base64,
        "heatmap_b64": heatmap_b64,
        "c2_research": c2_research_b64,
        "processing_time": processing_time
    }

# --- [C1] IDENTIFICATION ENDPOINT ---
@app.post("/identify")
async def identify_type(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"type": "none", "error": "Invalid image"}
        
    # Reuse engine localization
    c_crop, c_bbox, c_conf, g_crop, g_bbox, g_conf = engine._localize_all(img)
    
    if c_conf == -1.0 and g_conf == -1.0:
        return {"type": "none"}
        
    return {"type": "clock" if c_conf > g_conf else "gauge"}


def _serialize_c2_research(data):
    """Convert any numpy image arrays in c2_research dict to base64 strings."""
    if data is None:
        return None
    
    def _encode_img(img):
        if isinstance(img, np.ndarray) and img.size > 0:
            _, buffer = cv2.imencode('.jpg', img)
            return base64.b64encode(buffer).decode('utf-8')
        return None

    # Skeleton image
    if 'skeleton' in data and 'image' in data['skeleton']:
        data['skeleton']['image'] = _encode_img(data['skeleton']['image'])
    
    # Scale pyramid images
    if 'scale_analysis' in data and 'pyramid_images' in data['scale_analysis']:
        data['scale_analysis']['pyramid_images'] = [
            _encode_img(img) for img in data['scale_analysis']['pyramid_images']
        ]
    
    # Manifold image
    if 'manifold' in data and 'manifold_image' in data['manifold']:
        data['manifold']['manifold_image'] = _encode_img(data['manifold']['manifold_image'])
    
    return data

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


# =============================================================================
# [T3.2] CLOCK STYLE ENDPOINT
# =============================================================================
@app.post("/style/classify")
async def classify_style(file: UploadFile = File(...)):
    """
    [T3.2] Classify the visual style of a clock image.
    Returns: style_idx, style_name, confidence, per-class probabilities.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    from PIL import Image as PILImage
    pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = engine.style_analyser.classify(pil)
    return {"style": result}


# =============================================================================
# [T3.3] FEDERATED LEARNING ENDPOINTS
# =============================================================================
from app.core.federated import FederatedCoordinator, FederatedNode
import copy

# Global coordinator — shared across requests (in-memory for prototype)
_fed_coordinator: FederatedCoordinator = None

def _get_coordinator() -> FederatedCoordinator:
    global _fed_coordinator
    if _fed_coordinator is None and engine.c3_model is not None:
        _fed_coordinator = FederatedCoordinator(engine.c3_model)
        print("✅ FederatedCoordinator initialized")
    return _fed_coordinator


@app.post("/federated/register")
async def federated_register(node_id: str = Form(...), n_samples: int = Form(100)):
    """
    [T3.3] Register a new camera node with the federated coordinator.

    Args:
        node_id:   Unique node identifier (e.g. 'rtsp_cam_01').
        n_samples: Approximate number of local labelled samples.
    """
    coord = _get_coordinator()
    if coord is None:
        return JSONResponse({"error": "C3 model not loaded — coordinator unavailable."}, status_code=503)
    msg = coord.register_node(node_id, n_samples)
    return {"message": msg, "coordinator_status": coord.status()}


@app.get("/federated/pull_weights")
async def federated_pull_weights():
    """
    [T3.3] Pull current global model weights from the coordinator.
    Returns weights as a dict of {param_name: shape} (not the raw tensors —
    actual weight transfer would use a binary stream in production).
    """
    coord = _get_coordinator()
    if coord is None:
        return JSONResponse({"error": "Coordinator unavailable."}, status_code=503)
    weights = coord.get_global_weights()
    return {
        "round":       coord._round,
        "n_params":    sum(v.numel() for v in weights.values()),
        "param_shapes": {k: list(v.shape) for k, v in weights.items()},
        "message":     "Use POST /federated/push_update to send your weight delta after local training.",
    }


@app.get("/federated/status")
async def federated_status():
    """[T3.3] Get current federated coordinator status."""
    coord = _get_coordinator()
    if coord is None:
        return {"status": "not_initialized", "reason": "C3 model not loaded"}
    return coord.status()


# =============================================================================
# [T3.5] RESEARCH ARCHITECTURES INFO ENDPOINT
# =============================================================================
@app.get("/research/architectures")
async def get_research_architectures():
    """
    [T3.4, T3.5] Returns information about available Tier 3 architectures.
    """
    return {
        "standard_c3": {
            "backbone":     "ResNet18",
            "head":         "Sigmoid scalar",
            "loss":         "MSELoss",
            "limitation":   "0°/360° wraparound discontinuity",
            "checkpoint":   "models/c3_angle_regression/best.pth",
            "status":       "active",
        },
        "circular_c3": {
            "backbone":     "ResNet18",
            "head":         "CircularHead (sin θ, cos θ)",
            "loss":         "VonMisesLoss (1 - cos(δθ))",
            "advantage":    "No wraparound ambiguity — 0° and 360° identical",
            "training":     "scripts/train_c3_circular.py",
            "checkpoint":   "models/c3_circular/best.pth",
            "status":       "architecture ready, needs retraining",
            "tier":         "T3.5",
        },
        "vit_c3": {
            "backbone":     "ViT-B/16 (Vision Transformer)",
            "head":         "Sigmoid or CircularHead",
            "xai":          "Attention Rollout (no GradCAM++ needed)",
            "advantage":    "Built-in interpretable attention, 14×14 patch attention map",
            "training":     "scripts/train_c3_circular.py --backbone vit",
            "checkpoint":   "models/c3_vit/best.pth",
            "status":       "architecture ready, needs retraining",
            "tier":         "T3.4",
        },
        "style_conditioned_c3": {
            "backbone":     "ResNet18 + ClockStyleEmbedding (8-dim)",
            "head":         "Sigmoid scalar, conditioned on style",
            "styles":       ["Modern Analog", "Antique/Ornate", "Minimalist"],
            "advantage":    "Domain-adaptive — clock aesthetics improve cross-style accuracy",
            "status":       "architecture ready, needs retraining",
            "tier":         "T3.2",
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
