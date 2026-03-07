"""
C4 Gateway — API Gateway & Physics Service
Orchestrates C1 → C2 → C3 → C4 pipeline via HTTP
Runs on port 8000
Owner: Member 4

Scope:
  CLOCK (original):
  • AM/PM Inference    (/analyze enriched + /infer-ampm)
  • Ambiguity Resolver (/analyze enriched + /resolve-ambiguity)
  • Accuracy Checker   (/check-accuracy)
  • Elapsed Time       (/compare-clocks)
  • Report Generator   (/analyze enriched + /report)
  • Reading History    (/readings/history)

  GAUGE (new extension):
  • Single image       POST /gauge/analyze
  • Live CCTV stream   POST /gauge/stream-frame
  • Multiple gauges    POST /gauge/analyze-multiple
  • Register gauge     POST /gauge/register
  • Reset stream       POST /gauge/reset-stream
  • List gauge types   GET  /gauge/types
"""

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from typing import List, Optional
from pydantic import BaseModel

from services.c4_gateway.physics import physics_solver
from services.c4_gateway.metrics import metrics_tracker
from services.c4_gateway.orchestrator import call_c1, call_c2, call_c2_enhanced, call_c3, check_services

# --- Clock C4 modules ---
from services.c4_gateway.ampm_inference import ampm_engine
from services.c4_gateway.ambiguity_resolver import ambiguity_resolver
from services.c4_gateway.accuracy_checker import accuracy_checker
from services.c4_gateway.elapsed_time import elapsed_calculator
from services.c4_gateway.report_generator import report_generator

# --- Gauge C4 modules (new) ---
from services.c4_gateway.gauge_orchestrator import GaugeOrchestrator
from services.c4_gateway.gauge_config import list_gauge_types

_gauge_orch = GaugeOrchestrator()


# ── Request models — Clock ──────────────────────────────────────────────────
class SolveTimeRequest(BaseModel):
    hand1_angle: float = 0.0
    hand2_angle: float = 0.0


class AMPMRequest(BaseModel):
    hour: int
    minute: int
    hand1_angle: float
    hand2_angle: float
    user_hint: Optional[str] = None


class AmbiguityRequest(BaseModel):
    hand1_angle: float
    hand2_angle: float
    top_n: int = 5


class AccuracyRequest(BaseModel):
    hour: int
    minute: int
    period: Optional[str] = None
    tz_offset_hours: float = 0.0


class CompareRequest(BaseModel):
    start_hour: int
    start_minute: int
    end_hour: int
    end_minute: int
    start_period: Optional[str] = None
    end_period:   Optional[str] = None


class ReportRequest(BaseModel):
    analysis_result: dict
    ampm_result:     Optional[dict] = None
    ambiguity_result:Optional[dict] = None
    accuracy_result: Optional[dict] = None


# ── Request models — Gauge ──────────────────────────────────────────────────
class GaugeAnalyzeRequest(BaseModel):
    """Payload for a single uploaded gauge image."""
    c3_output: dict                          # {"angle": float, "confidence": float}
    c1_metadata: Optional[dict] = None       # {"class": "...", "unit": "...", ...}
    gauge_type_override: Optional[str] = None
    ocr_text: Optional[str] = None


class GaugeStreamRequest(BaseModel):
    """Payload for a single CCTV stream frame."""
    c3_output: dict
    c1_metadata: Optional[dict] = None
    gauge_type_override: Optional[str] = None
    ocr_text: Optional[str] = None
    frame_id: Optional[int] = None
    stream_id: str = "default"
    smooth: bool = True


class GaugeMultipleRequest(BaseModel):
    """Payload for multiple gauges in one scene."""
    gauge_list: List[dict]
    source: str = "image"
    frame_id: Optional[int] = None
    stream_id: str = "default"


class GaugeRegisterRequest(BaseModel):
    """Register a new custom gauge type."""
    gauge_type: str
    start_angle: float
    end_angle: float
    min_value: float
    max_value: float
    unit: str
    description: str = ""
    warning_high: Optional[float] = None
    warning_low: Optional[float] = None
    danger_high: Optional[float] = None
    danger_low: Optional[float] = None
    decimal_places: int = 1


class GaugeResetRequest(BaseModel):
    stream_id: str = "default"


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="C4 - Clock & Gauge AI Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ============================================================
# HEALTH
# ============================================================
@app.get("/health")
async def health():
    """Gateway health + downstream service status."""
    return {
        "service": "C4-Gateway",
        "status": "ok",
        "downstream": check_services(),
        "gauge_types_registered": len(list_gauge_types()),
    }


# ============================================================
# CLOCK ENDPOINTS  (original — unchanged)
# ============================================================

@app.post("/analyze")
async def analyze_clock(file: UploadFile = File(...), force_expert: bool = Form(False)):
    """Full pipeline: C1 -> C2 -> (optionally C3) -> C4 Physics."""
    start_time = time.time()
    debug_info = []
    visualizations = {}

    contents = await file.read()

    # ── C1 Localization ──────────────────────────────────────────────────────
    c1_result = call_c1(contents, file.filename)
    if "error" in c1_result and "found" not in c1_result:
        return {"result": {"error": c1_result['error']}, "processing_time": time.time() - start_time}

    if not c1_result.get("found"):
        return {
            "result": {"error": "No clock face detected in the uploaded image. Please upload a clear photo of an analog clock."},
            "processing_time": time.time() - start_time,
        }

    debug_info.append("C1: Clock Found" if c1_result.get("found") else "C1: Full Scan")
    visualizations["c1_detection"] = c1_result.get("visualization")
    visualizations["c1_hough"]     = c1_result.get("hough_visualization")
    cropped_b64 = c1_result.get("cropped_image")
    c1_quality = {
        "confidence":       c1_result.get("confidence"),
        "quality":          c1_result.get("quality"),
        "hough_validation": c1_result.get("hough_validation"),
    }

    # ── C2 Skeleton Extraction (Enhanced) ────────────────────────────────────
    c2_result = call_c2_enhanced(cropped_b64)
    if "error" in c2_result:
        return {"result": {"error": c2_result['error']}, "processing_time": time.time() - start_time}

    keypoints           = c2_result["keypoints"]
    angles              = c2_result["angles"]
    visualizations["c2_skeleton"] = c2_result.get("visualization")
    c2_enhanced         = c2_result.get("enhanced", {})
    c2_research_visuals = c2_result.get("research_visuals", {})
    debug_info.append("C2: Enhanced analysis complete")

    # ── C4 Physics (Fast Path) ────────────────────────────────────────────────
    a1 = angles["hand1"]
    a2 = angles["hand2"]
    physics_result = physics_solver.solve(a1, a2)

    if physics_result["error"] < 8.0 and not force_expert:
        c2_conf         = c2_enhanced.get("reconstruction_3d", {}).get("confidence", 0.5)
        c2_occ          = c2_enhanced.get("reconstruction_3d", {}).get("occlusion_risk", "UNKNOWN")
        c2_ha           = c2_enhanced.get("reconstruction_3d", {}).get("hand_assignment", {})
        uncertainty_min = max(1, int((1 - c2_conf) * 10))

        result = {
            "time": physics_result["time"],
            "method": "Fast Path (C1+C2+C4)",
            "confidence": "High",
            "heatmap": None,
            "debug": debug_info,
            "angles": {"hand1": a1, "hand2": a2},
            "reasoning": physics_result["reasoning"],
            "c2_confidence": round(c2_conf, 3),
            "c2_occlusion_risk": c2_occ,
            "c2_hand_assignment": c2_ha,
            "uncertainty": f"+-{uncertainty_min} min",
        }

        ampm_res   = ampm_engine.to_dict(ampm_engine.infer(physics_result["hour"], physics_result["minute"], a1, a2))
        amb_res    = ambiguity_resolver.to_dict(ambiguity_resolver.resolve(a1, a2))
        report_res = report_generator.to_dict(report_generator.generate(result, ampm_res, amb_res))

        processing_time = time.time() - start_time
        metrics_tracker.record_analysis(result, processing_time, file.filename)

        return {
            "result": result,
            "visualizations": visualizations,
            "c2_enhanced": c2_enhanced,
            "c2_research_visuals": c2_research_visuals,
            "heatmap_b64": None,
            "c1_quality": c1_quality,
            "processing_time": processing_time,
            "ampm": ampm_res,
            "ambiguity": amb_res,
            "report": report_res,
        }

    # ── C3 Expert Refinement ──────────────────────────────────────────────────
    debug_info.append("C4: Error too high or expert forced — calling C3")
    c3_result = call_c3(cropped_b64, keypoints, angles)

    if c3_result.get("refined") and "error" not in c3_result:
        refined = c3_result["refined_angles"]
        debug_info.extend(c3_result.get("debug", []))

        if c3_result.get("angle_visualization"):
            visualizations["c3_angles"] = c3_result["angle_visualization"]
        if c3_result.get("crops"):
            visualizations["c3_crops"] = c3_result["crops"]

        refined_physics = physics_solver.solve(refined["hand1"], refined["hand2"])
        result = {
            "time": refined_physics["time"],
            "method": "Expert Path (C1+C2+C3+C4)",
            "confidence": "Refined",
            "heatmap": None,
            "debug": debug_info,
            "angles": {"hand1": refined["hand1"], "hand2": refined["hand2"]},
            "reasoning": f"Refined: H={refined['hand1']:.1f}, M={refined['hand2']:.1f} -> Time={refined_physics['time']}",
            "c2_confidence": round(c2_enhanced.get("reconstruction_3d", {}).get("confidence", 0.5), 3),
            "c2_occlusion_risk": c2_enhanced.get("reconstruction_3d", {}).get("occlusion_risk", "UNKNOWN"),
            "c2_hand_assignment": c2_enhanced.get("reconstruction_3d", {}).get("hand_assignment", {}),
            "uncertainty": f"+-{max(1, int((1 - c2_enhanced.get('reconstruction_3d', {}).get('confidence', 0.5)) * 10))} min",
        }
        heatmap_b64 = c3_result.get("heatmap")
    else:
        debug_info.append(f"C3 unavailable: {c3_result.get('error', 'unknown')}")
        result = {
            "time": physics_result["time"],
            "method": "Fast Path (C3 Unavailable)",
            "confidence": "Low",
            "heatmap": None,
            "debug": debug_info,
            "angles": {"hand1": a1, "hand2": a2},
            "reasoning": physics_result["reasoning"],
        }
        heatmap_b64 = None

    processing_time = time.time() - start_time
    metrics_tracker.record_analysis(result, processing_time, file.filename)

    final_a1 = result["angles"]["hand1"]
    final_a2 = result["angles"]["hand2"]
    try:
        _h = int(result["time"].split(":")[0])
        _m = int(result["time"].split(":")[1])
    except Exception:
        _h, _m = 12, 0

    ampm_res   = ampm_engine.to_dict(ampm_engine.infer(_h, _m, final_a1, final_a2))
    amb_res    = ambiguity_resolver.to_dict(ambiguity_resolver.resolve(final_a1, final_a2))
    report_res = report_generator.to_dict(report_generator.generate(result, ampm_res, amb_res))

    return {
        "result": result,
        "visualizations": visualizations,
        "c2_enhanced": c2_enhanced,
        "c2_research_visuals": c2_research_visuals,
        "heatmap_b64": heatmap_b64,
        "c1_quality": c1_quality,
        "processing_time": processing_time,
        "ampm": ampm_res,
        "ambiguity": amb_res,
        "report": report_res,
    }


@app.post("/analyze_batch")
async def analyze_batch(files: List[UploadFile] = File(...), force_expert: bool = Form(False)):
    """Batch processing — calls the full pipeline for each image."""
    results = []
    for f in files:
        try:
            start_time = time.time()
            contents   = await f.read()

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
            physics_result  = physics_solver.solve(a1, a2)
            processing_time = time.time() - start_time
            method          = "Fast Path (C1+C2+C4)"

            if physics_result["error"] >= 8.0 or force_expert:
                c3_result = call_c3(c1_result.get("cropped_image"), c2_result["keypoints"], c2_result["angles"])
                if c3_result.get("refined"):
                    refined        = c3_result["refined_angles"]
                    physics_result = physics_solver.solve(refined["hand1"], refined["hand2"])
                    method         = "Expert Path (C1+C2+C3+C4)"

            res = {"time": physics_result["time"], "method": method}
            metrics_tracker.record_analysis(res, processing_time, f.filename)
            results.append({
                "filename": f.filename, "success": True,
                "time": physics_result["time"], "method": method,
                "processing_time": processing_time,
            })
        except Exception as e:
            results.append({"filename": f.filename, "success": False, "error": str(e)})

    return {"total_images": len(files), "results": results}


@app.post("/solve-time")
async def solve_time(data: SolveTimeRequest):
    """Direct physics solver endpoint for testing."""
    return physics_solver.solve(data.hand1_angle, data.hand2_angle)


@app.post("/infer-ampm")
async def infer_ampm(req: AMPMRequest):
    result = ampm_engine.infer(req.hour, req.minute, req.hand1_angle, req.hand2_angle, req.user_hint)
    return ampm_engine.to_dict(result)


@app.post("/resolve-ambiguity")
async def resolve_ambiguity(req: AmbiguityRequest):
    result = ambiguity_resolver.resolve(req.hand1_angle, req.hand2_angle, req.top_n)
    return ambiguity_resolver.to_dict(result)


@app.post("/check-accuracy")
async def check_accuracy(req: AccuracyRequest):
    result = accuracy_checker.check(req.hour, req.minute, req.period, req.tz_offset_hours)
    return accuracy_checker.to_dict(result)


@app.post("/analyze-with-accuracy")
async def analyze_with_accuracy(
    file: UploadFile = File(...),
    force_expert: bool = Form(False),
    period: Optional[str] = Form(None),
    tz_offset_hours: float = Form(0.0),
):
    start_time = time.time()
    contents   = await file.read()

    c1_result = call_c1(contents, file.filename)
    if "error" in c1_result and "found" not in c1_result:
        return {"error": f"C1 Failed: {c1_result['error']}"}

    c2_result = call_c2(c1_result.get("cropped_image"))
    if "error" in c2_result:
        return {"error": f"C2 Failed: {c2_result['error']}"}

    a1             = c2_result["angles"]["hand1"]
    a2             = c2_result["angles"]["hand2"]
    physics_result = physics_solver.solve(a1, a2)
    h, m           = physics_result["hour"], physics_result["minute"]

    return {
        "detected_time":   physics_result["time"],
        "angles":          {"hand1": a1, "hand2": a2},
        "ampm":            ampm_engine.to_dict(ampm_engine.infer(h, m, a1, a2)),
        "accuracy":        accuracy_checker.to_dict(accuracy_checker.check(h, m, period, tz_offset_hours)),
        "processing_time": round(time.time() - start_time, 3),
    }


@app.post("/compare-clocks")
async def compare_clocks(
    file_start: UploadFile = File(...),
    file_end:   UploadFile = File(...),
    start_period: Optional[str] = Form(None),
    end_period:   Optional[str] = Form(None),
):
    start_time = time.time()

    c1_s = call_c1(await file_start.read(), file_start.filename)
    c2_s = call_c2(c1_s.get("cropped_image", ""))
    if "error" in c2_s:
        return {"error": f"Could not process Start image: {c2_s['error']}"}
    ps = physics_solver.solve(c2_s["angles"]["hand1"], c2_s["angles"]["hand2"])

    c1_e = call_c1(await file_end.read(), file_end.filename)
    c2_e = call_c2(c1_e.get("cropped_image", ""))
    if "error" in c2_e:
        return {"error": f"Could not process End image: {c2_e['error']}"}
    pe = physics_solver.solve(c2_e["angles"]["hand1"], c2_e["angles"]["hand2"])

    elapsed_result = elapsed_calculator.calculate(
        start_hour=ps["hour"],   start_minute=ps["minute"],
        end_hour=pe["hour"],     end_minute=pe["minute"],
        start_period=start_period, end_period=end_period,
    )
    return {
        "start_clock_reading": ps["time"],
        "end_clock_reading":   pe["time"],
        "elapsed":             elapsed_calculator.to_dict(elapsed_result),
        "processing_time":     round(time.time() - start_time, 3),
    }


@app.post("/elapsed-from-readings")
async def elapsed_from_readings(req: CompareRequest):
    result = elapsed_calculator.calculate(
        req.start_hour, req.start_minute,
        req.end_hour,   req.end_minute,
        req.start_period, req.end_period,
    )
    return elapsed_calculator.to_dict(result)


@app.post("/report")
async def generate_report(req: ReportRequest):
    report = report_generator.generate(
        req.analysis_result, req.ampm_result, req.ambiguity_result, req.accuracy_result,
    )
    return report_generator.to_dict(report)


@app.get("/readings/history")
async def readings_history(
    limit:       int   = Query(50, ge=1, le=500),
    confidence:  Optional[str]   = Query(None, description="Filter: High | Refined | Low"),
    method:      Optional[str]   = Query(None, description="Filter by method substring"),
    since_hours: float = Query(0, description="Only show readings from last N hours (0 = all)"),
):
    history = metrics_tracker.get_history(
        limit=limit,
        confidence_filter=confidence,
        method_filter=method,
        since_hours=since_hours if since_hours > 0 else None,
    )
    return {"count": len(history), "readings": history}


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


# ============================================================
# GAUGE ENDPOINTS  (new extension)
# ============================================================

@app.post("/gauge/analyze")
async def gauge_analyze(req: GaugeAnalyzeRequest):
    """
    Gauge reading from a SINGLE uploaded image.

    C4 receives the needle angle from C3, determines the gauge type
    automatically (or uses gauge_type_override), and returns the
    real-world measurement value.

    Request body:
        {
            "c3_output":           {"angle": 45.0, "confidence": 0.95},
            "c1_metadata":         {"class": "pressure_gauge"},
            "gauge_type_override": null,
            "ocr_text":            null
        }

    Response:
        {
            "value": 66.7,  "unit": "PSI",
            "gauge_type": "pressure_psi",
            "percentage": 66.7,
            "status": "normal",  "status_detail": "...",
            "confidence": 0.95,
            "display": "66.7 PSI",
            "raw_angle_deg": 45.0,
            "source": "image"
        }
    """
    reading = _gauge_orch.process_image(
        c3_output=req.c3_output,
        c1_metadata=req.c1_metadata,
        gauge_type_override=req.gauge_type_override,
        ocr_text=req.ocr_text,
    )
    return reading.to_dict()


@app.post("/gauge/stream-frame")
async def gauge_stream_frame(req: GaugeStreamRequest):
    """
    Gauge reading from a LIVE CCTV stream frame.

    Applies temporal smoothing across frames for the same stream_id
    to reduce needle jitter. Call /gauge/reset-stream when the camera
    moves to a different gauge.

    Request body:
        {
            "c3_output":           {"angle": -30.0, "confidence": 0.9},
            "c1_metadata":         {"unit": "PSI"},
            "frame_id":            1042,
            "stream_id":           "boiler_cam",
            "smooth":              true
        }
    """
    reading = _gauge_orch.process_stream_frame(
        c3_output=req.c3_output,
        c1_metadata=req.c1_metadata,
        gauge_type_override=req.gauge_type_override,
        ocr_text=req.ocr_text,
        frame_id=req.frame_id,
        stream_id=req.stream_id,
        smooth=req.smooth,
    )
    return reading.to_dict()


@app.post("/gauge/analyze-multiple")
async def gauge_analyze_multiple(req: GaugeMultipleRequest):
    """
    Process MULTIPLE gauges detected in a single image or CCTV frame.

    Use this when C1 detects more than one gauge in the same scene.

    Request body:
        {
            "gauge_list": [
                {"c3_output": {"angle": 30},  "gauge_type_override": "pressure_psi"},
                {"c3_output": {"angle": -10}, "gauge_type_override": "temperature_celsius"}
            ],
            "source": "image"
        }

    Response:
        {"count": 2, "readings": [ {...}, {...} ]}
    """
    readings = _gauge_orch.process_multiple_gauges(
        gauge_list=req.gauge_list,
        source=req.source,
        frame_id=req.frame_id,
        stream_id=req.stream_id,
    )
    return {"count": len(readings), "readings": [r.to_dict() for r in readings]}


@app.post("/gauge/register")
async def gauge_register(req: GaugeRegisterRequest):
    """
    Register a NEW custom gauge type at runtime.
    After calling this, pass the gauge_type key in any /gauge/* request.

    Request body:
        {
            "gauge_type":   "boiler_pressure",
            "start_angle":  -135,  "end_angle": 135,
            "min_value":    0,     "max_value": 16,
            "unit":         "bar",
            "danger_high":  14
        }
    """
    _gauge_orch.register_gauge(
        gauge_type=req.gauge_type,
        start_angle=req.start_angle,   end_angle=req.end_angle,
        min_value=req.min_value,       max_value=req.max_value,
        unit=req.unit,                 description=req.description,
        warning_high=req.warning_high, warning_low=req.warning_low,
        danger_high=req.danger_high,   danger_low=req.danger_low,
        decimal_places=req.decimal_places,
    )
    return {"status": "registered", "gauge_type": req.gauge_type}


@app.post("/gauge/reset-stream")
async def gauge_reset_stream(req: GaugeResetRequest):
    """
    Clear the temporal smoothing history for a CCTV stream.
    Call this when the camera moves to a different gauge.
    """
    _gauge_orch.reset_stream(req.stream_id)
    return {"status": "reset", "stream_id": req.stream_id}


@app.get("/gauge/types")
async def gauge_list_types():
    """Return all registered gauge type keys (built-in + custom)."""
    return {"gauge_types": list_gauge_types()}


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)