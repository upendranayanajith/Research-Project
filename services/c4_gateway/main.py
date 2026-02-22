"""
C4 Gateway — API Gateway & Physics Service
Orchestrates C1 → C2 → C3 → C4 pipeline via HTTP
Runs on port 8000 (same as original — frontend doesn't change)
Owner: Member 4
"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from typing import List

from services.c4_gateway.physics import physics_solver
from services.c4_gateway.metrics import metrics_tracker
from services.c4_gateway.orchestrator import call_c1, call_c2, call_c3, check_services

app = FastAPI(title="C4 - Clock AI Gateway (Microservice Architecture)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    """Gateway health + downstream service status."""
    return {
        "service": "C4-Gateway",
        "status": "ok",
        "downstream": check_services()
    }


@app.post("/analyze")
async def analyze_clock(file: UploadFile = File(...), force_expert: bool = Form(False)):
    """Full pipeline: C1 → C2 → (optionally C3) → C4 Physics."""
    start_time = time.time()
    debug_info = []
    visualizations = {}

    # --- Read image ---
    contents = await file.read()

    # ========== STEP 1: C1 Localization ==========
    c1_result = call_c1(contents, file.filename)
    if "error" in c1_result and "found" not in c1_result:
        return {"result": {"error": f"C1 Failed: {c1_result['error']}"}, "processing_time": time.time() - start_time}

    debug_info.append("C1: Clock Found" if c1_result.get("found") else "C1: Full Scan")
    visualizations["c1_detection"] = c1_result.get("visualization")
    cropped_b64 = c1_result.get("cropped_image")

    # ========== STEP 2: C2 Skeleton Extraction ==========
    c2_result = call_c2(cropped_b64)
    if "error" in c2_result:
        return {"result": {"error": f"C2 Failed: {c2_result['error']}"}, "processing_time": time.time() - start_time}

    keypoints = c2_result["keypoints"]
    angles = c2_result["angles"]
    visualizations["c2_skeleton"] = c2_result.get("visualization")
    debug_info.append("C2: Keypoints extracted")

    # ========== STEP 3: C4 Physics (Fast Path) ==========
    a1 = angles["hand1"]
    a2 = angles["hand2"]
    physics_result = physics_solver.solve(a1, a2)

    if physics_result["error"] < 8.0 and not force_expert:
        # Fast Path — skip C3
        result = {
            "time": physics_result["time"],
            "method": "Fast Path (C1+C2+C4)",
            "confidence": "High",
            "heatmap": None,
            "debug": debug_info,
            "angles": {"hand1": a1, "hand2": a2},
            "reasoning": physics_result["reasoning"]
        }
        processing_time = time.time() - start_time
        metrics_tracker.record_analysis(result, processing_time, file.filename)

        return {
            "result": result,
            "visualizations": visualizations,
            "heatmap_b64": None,
            "processing_time": processing_time
        }

    # ========== STEP 4: C3 Expert Refinement ==========
    debug_info.append("C4: Error too high or expert forced — calling C3")
    c3_result = call_c3(cropped_b64, keypoints, angles)

    if c3_result.get("refined") and "error" not in c3_result:
        refined = c3_result["refined_angles"]
        debug_info.extend(c3_result.get("debug", []))

        # Update visualizations from C3
        if c3_result.get("angle_visualization"):
            visualizations["c3_angles"] = c3_result["angle_visualization"]
        if c3_result.get("crops"):
            visualizations["c3_crops"] = c3_result["crops"]

        # Re-solve with refined angles
        refined_physics = physics_solver.solve(refined["hand1"], refined["hand2"])

        result = {
            "time": refined_physics["time"],
            "method": "Expert Path (C1+C2+C3+C4)",
            "confidence": "Refined",
            "heatmap": None,
            "debug": debug_info,
            "angles": {"hand1": refined["hand1"], "hand2": refined["hand2"]},
            "reasoning": f"Refined: H={refined['hand1']:.1f}°, M={refined['hand2']:.1f}° → Time={refined_physics['time']}"
        }
        heatmap_b64 = c3_result.get("heatmap")
    else:
        # C3 failed — fall back to fast path result
        debug_info.append(f"C3 unavailable: {c3_result.get('error', 'unknown')}")
        result = {
            "time": physics_result["time"],
            "method": "Fast Path (C3 Unavailable)",
            "confidence": "Low",
            "heatmap": None,
            "debug": debug_info,
            "angles": {"hand1": a1, "hand2": a2},
            "reasoning": physics_result["reasoning"]
        }
        heatmap_b64 = None

    processing_time = time.time() - start_time
    metrics_tracker.record_analysis(result, processing_time, file.filename)

    return {
        "result": result,
        "visualizations": visualizations,
        "heatmap_b64": heatmap_b64,
        "processing_time": processing_time
    }


@app.post("/analyze_batch")
async def analyze_batch(files: List[UploadFile] = File(...), force_expert: bool = Form(False)):
    """Batch processing — calls the full pipeline for each image."""
    results = []
    for f in files:
        try:
            start_time = time.time()
            contents = await f.read()

            c1_result = call_c1(contents, f.filename)
            if c1_result.get("error") and not c1_result.get("found", False):
                results.append({"filename": f.filename, "success": False, "error": c1_result["error"]})
                continue

            c2_result = call_c2(c1_result.get("cropped_image"))
            if "error" in c2_result:
                results.append({"filename": f.filename, "success": False, "error": c2_result["error"]})
                continue

            a1 = c2_result["angles"]["hand1"]
            a2 = c2_result["angles"]["hand2"]
            physics_result = physics_solver.solve(a1, a2)
            processing_time = time.time() - start_time

            method = "Fast Path (C1+C2+C4)"
            if physics_result["error"] >= 8.0 or force_expert:
                c3_result = call_c3(c1_result.get("cropped_image"), c2_result["keypoints"], c2_result["angles"])
                if c3_result.get("refined"):
                    refined = c3_result["refined_angles"]
                    physics_result = physics_solver.solve(refined["hand1"], refined["hand2"])
                    method = "Expert Path (C1+C2+C3+C4)"

            res = {"time": physics_result["time"], "method": method}
            metrics_tracker.record_analysis(res, processing_time, f.filename)
            results.append({
                "filename": f.filename, "success": True,
                "time": physics_result["time"], "method": method,
                "processing_time": processing_time
            })
        except Exception as e:
            results.append({"filename": f.filename, "success": False, "error": str(e)})

    return {"total_images": len(files), "results": results}


# --- C4 Physics Direct Endpoint ---
@app.post("/solve-time")
async def solve_time(data: dict):
    """Direct physics solver endpoint for testing."""
    a1 = data.get("hand1_angle", 0)
    a2 = data.get("hand2_angle", 0)
    return physics_solver.solve(a1, a2)


# --- Metrics Endpoints ---
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
