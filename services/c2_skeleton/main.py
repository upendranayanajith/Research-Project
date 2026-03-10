"""
C2 Skeleton Service — Extended v4.0 (Dual-Mode: Clock + Gauge)
===============================================================
Keypoint Extraction + Probabilistic 3D + Temporal Dynamics
+ Multi-Scale + Manifold + Causal + LVM Temporal + Combined Pipeline

Supports TWO instrument types via `mode` query parameter:
  - mode=clock  (default) → 3 keypoints: center, tip1 (hour), tip2 (minute)
  - mode=gauge  → 2 keypoints: center, needle_tip

Port   : 8002
Owner  : Member 2 (extended for research)

Endpoints
---------
Original (v1):
  GET  /health              — service health
  POST /extract-skeleton    — 2D keypoint extraction (YOLO)

Research v2 (GAP 1 & 2):
  POST /extract-skeleton-enhanced — full research analysis
  POST /extract-skeleton-3d  — Bayesian 3D reconstruction
  POST /track-temporal       — persistent homology tracking
  POST /analyze-sequence     — batch temporal analysis
  DELETE /reset-tracker      — reset tracker state
  GET  /tracker-summary      — session topology summary

Research v3 (GAP 3, 4, 5 + combination):
  POST /extract-multiscale   — GAP 3: scale-space LVM oracle
  POST /extract-manifold     — GAP 4: Riemannian geodesic skeleton
  POST /analyze-causal       — GAP 5: Granger causality discovery
  POST /smooth-temporal      — LVM-gated temporal smoothing
  POST /extract-combined     — RECOMMENDED: full pipeline
  GET  /pipeline-stats       — combined pipeline session stats
  DELETE /reset-pipeline     — reset combined pipeline state

All POST endpoints accept ?mode=clock (default) or ?mode=gauge
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import math
import os
from typing import List, Optional, Literal, Tuple, Union
from enum import Enum

# ── YOLO model ──────────────────────────────────────────────────────────────
from ultralytics import YOLO

# ── Research extension modules ───────────────────────────────────────────────
try:
    from .probabilistic_3d import BayesianGraphInference
    from .temporal_dynamics import TemporalGraphTracker
    EXTENSIONS_AVAILABLE = True
except ImportError:
    from probabilistic_3d import BayesianGraphInference
    from temporal_dynamics import TemporalGraphTracker
    EXTENSIONS_AVAILABLE = True


# ═══════════════════════════════════════════════════════════════════════════════
# Constants & Mode Definitions
# ═══════════════════════════════════════════════════════════════════════════════

VALID_MODES = ("clock", "gauge")

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="C2 - Skeleton Service (Clock + Gauge)",
    description=(
        "Dual-mode keypoint extraction and research analysis. "
        "Supports clocks (3 keypoints) and gauges (2 keypoints)."
    ),
    version="4.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLOCK_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
GAUGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "gauge_best.pt")

c2_clock_model = None
c2_gauge_model = None

try:
    print(f"[C2] Loading clock YOLO-Pose model: {CLOCK_MODEL_PATH}")
    c2_clock_model = YOLO(CLOCK_MODEL_PATH)
    print("[C2] Clock model loaded successfully.")
except Exception as e:
    print(f"[C2] ⚠️ Clock model load failed: {e}")

try:
    if os.path.exists(GAUGE_MODEL_PATH):
        print(f"[C2] Loading gauge YOLO-Pose model: {GAUGE_MODEL_PATH}")
        c2_gauge_model = YOLO(GAUGE_MODEL_PATH)
        print("[C2] Gauge model loaded successfully.")
    else:
        print(f"[C2] ℹ️ Gauge model not found at {GAUGE_MODEL_PATH} — "
              "gauge mode will use clock model as fallback")
except Exception as e:
    print(f"[C2] ⚠️ Gauge model load failed: {e}")

# ── Singleton trackers (per-session state) ────────────────────────────────────
_bayesian_engine = BayesianGraphInference(k_hypotheses=10, image_size=500)
_temporal_tracker = TemporalGraphTracker(max_history=100)


# ═══════════════════════════════════════════════════════════════════════════════
# Request / Response schemas
# ═══════════════════════════════════════════════════════════════════════════════

class FrameKeypointsRequest(BaseModel):
    """Keypoints for a single frame.

    For clock mode: center, tip1, tip2 (all required)
    For gauge mode: center and tip1 (needle_tip); tip2 is ignored
    """
    center: Optional[List[float]] = None    # [x, y]
    tip1:   Optional[List[float]] = None    # [x, y] — needle_tip in gauge mode
    tip2:   Optional[List[float]] = None    # [x, y] — unused in gauge mode


class SequenceRequest(BaseModel):
    """Batch of frames for sequence analysis."""
    frames: List[FrameKeypointsRequest]


class SkeletonSequenceRequest(BaseModel):
    """Batch of skeletons for temporal smoothing or causal analysis."""
    frames: List[FrameKeypointsRequest]


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helper functions — dual-mode aware
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_mode(mode: str) -> str:
    """Normalise and validate the mode parameter."""
    mode = mode.strip().lower()
    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Must be one of: {VALID_MODES}",
        )
    return mode


def _resize_small(img, size=500):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _encode_image(img) -> str:
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


def _get_angle(center, point) -> float:
    """Calculate clockwise angle from 12-o'clock position.

    Works identically for both clock hands and gauge needles.
    """
    dx, dy = point[0] - center[0], point[1] - center[1]
    angle = math.degrees(math.atan2(dx, -dy))
    return angle + 360 if angle < 0 else angle


def _draw_skeleton(img, center, tip1, tip2=None, mode="clock"):
    """Draw skeleton overlay on the image.

    Clock mode: 2 lines (green=hand1, red=hand2) + 3 circles
    Gauge mode: 1 line (orange=needle) + 2 circles
    """
    img_copy = img.copy()
    center_pt = (int(center[0]), int(center[1]))
    tip1_pt = (int(tip1[0]), int(tip1[1]))

    if mode == "gauge":
        # Single needle — orange line
        cv2.line(img_copy, center_pt, tip1_pt, (0, 165, 255), 4)
        cv2.circle(img_copy, center_pt, 8, (255, 0, 0), -1)   # blue center
        cv2.circle(img_copy, tip1_pt, 8, (0, 165, 255), -1)    # orange tip
    else:
        # Clock — two hands
        tip2_pt = (int(tip2[0]), int(tip2[1]))
        cv2.line(img_copy, center_pt, tip1_pt, (0, 255, 0), 4)   # green
        cv2.line(img_copy, center_pt, tip2_pt, (0, 0, 255), 4)   # red
        cv2.circle(img_copy, center_pt, 8, (255, 0, 0), -1)
        cv2.circle(img_copy, tip1_pt, 8, (0, 255, 0), -1)
        cv2.circle(img_copy, tip2_pt, 8, (0, 0, 255), -1)

    return _resize_small(img_copy)


def _get_model_for_mode(mode: str):
    """Return the appropriate YOLO model based on mode."""
    if mode == "gauge":
        return c2_gauge_model if c2_gauge_model is not None else c2_clock_model
    return c2_clock_model


def _extract_keypoints_from_image(img, mode="clock"):
    """Run YOLO-Pose on image and extract keypoints.

    Clock mode: returns (center, tip1, tip2), None
    Gauge mode: returns (center, needle_tip), None

    The gauge model may produce 2 or 3 keypoints — we always take
    center + first tip. If only 2 keypoints exist, that's acceptable.
    """
    model = _get_model_for_mode(mode)
    if model is None:
        return None, f"No YOLO model loaded for {mode} mode"

    results = model(img, verbose=False)[0]
    if not results.keypoints or len(results.keypoints.data) == 0:
        label = "needle" if mode == "gauge" else "hands"
        return None, f"No {label}/keypoints detected"

    kpts = results.keypoints.data[0].cpu().numpy()
    num_kpts = kpts.shape[0]

    center = kpts[0][:2].tolist()

    if mode == "gauge":
        # Gauge: center + needle_tip (1 needle)
        if num_kpts < 2:
            return None, "Gauge detection found center but no needle tip"
        needle_tip = kpts[1][:2].tolist()
        return (center, needle_tip), None
    else:
        # Clock: center + tip1 + tip2 (2 hands)
        if num_kpts < 3:
            return None, "Clock detection requires 3 keypoints (center + 2 hands)"
        tip1 = kpts[1][:2].tolist()
        tip2 = kpts[2][:2].tolist()
        return (center, tip1, tip2), None


def _build_keypoints_response(keypoints, mode="clock") -> dict:
    """Build a standardised keypoints dict for the response."""
    if mode == "gauge":
        center, needle_tip = keypoints
        return {"center": center, "needle_tip": needle_tip}
    else:
        center, tip1, tip2 = keypoints
        return {"center": center, "tip1": tip1, "tip2": tip2}


def _build_angles_response(keypoints, mode="clock") -> dict:
    """Build a standardised angles dict for the response."""
    if mode == "gauge":
        center, needle_tip = keypoints
        angle = _get_angle(center, needle_tip)
        return {"needle": round(angle, 2)}
    else:
        center, tip1, tip2 = keypoints
        return {
            "hand1": round(_get_angle(center, tip1), 2),
            "hand2": round(_get_angle(center, tip2), 2),
        }


def _build_visualization(img, keypoints, mode="clock") -> str:
    """Draw skeleton and encode as base64."""
    if mode == "gauge":
        center, needle_tip = keypoints
        viz = _draw_skeleton(img, center, needle_tip, mode="gauge")
    else:
        center, tip1, tip2 = keypoints
        viz = _draw_skeleton(img, center, tip1, tip2, mode="clock")
    return _encode_image(viz)


def _unpack_for_3_point(keypoints, mode="clock"):
    """Unpack keypoints to (center, tip1, tip2) for sub-modules that expect 3 points.

    For gauge mode, tip2 is set to the same as tip1 so sub-modules don't crash.
    """
    if mode == "gauge":
        center, needle_tip = keypoints
        return center, needle_tip, needle_tip  # duplicate tip for compatibility
    else:
        return keypoints  # already (center, tip1, tip2)


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 1 — Health
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Extended health check — reports model and module availability."""
    return {
        "service": "C2-Skeleton",
        "version": "4.0.0",
        "status": "ok",
        "supported_modes": list(VALID_MODES),
        "models": {
            "clock": c2_clock_model is not None,
            "gauge": c2_gauge_model is not None,
            "gauge_fallback_to_clock": (
                c2_gauge_model is None and c2_clock_model is not None
            ),
        },
        "modules": {
            "probabilistic_3d": EXTENSIONS_AVAILABLE,
            "temporal_dynamics": EXTENSIONS_AVAILABLE,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 2 — Basic Skeleton Extraction
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/extract-skeleton")
async def extract_skeleton(
    file: UploadFile = File(...),
    mode: str = Query("clock", description="Instrument type: 'clock' or 'gauge'"),
):
    """
    Extract 2D keypoints and compute angles.

    - **clock** (default): 3 keypoints (center, tip1, tip2) + 2 angles
    - **gauge**: 2 keypoints (center, needle_tip) + 1 angle
    """
    mode = _validate_mode(mode)

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    keypoints, error = _extract_keypoints_from_image(img, mode=mode)
    if error:
        return {"error": error, "mode": mode}

    return {
        "mode": mode,
        "keypoints": _build_keypoints_response(keypoints, mode),
        "angles": _build_angles_response(keypoints, mode),
        "visualization": _build_visualization(img, keypoints, mode),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 3 — Enhanced (Full Research Analysis)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/extract-skeleton-enhanced")
async def extract_skeleton_enhanced(
    file: UploadFile = File(...),
    mode: str = Query("clock", description="Instrument type: 'clock' or 'gauge'"),
):
    """
    ENHANCED — Runs all C2 research algorithms.
    Returns keypoints, angles, 3D uncertainty, multi-scale analysis,
    manifold curvature, Betti topology, and 6 visualisation images.

    Works for both clocks and gauges — adapts analysis automatically.
    """
    mode = _validate_mode(mode)

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    keypoints, error = _extract_keypoints_from_image(img, mode=mode)
    if error:
        return {"error": error, "mode": mode}

    center, tip1, tip2 = _unpack_for_3_point(keypoints, mode)

    # ── Base result ──────────────────────────────────────────────────
    result = {
        "mode": mode,
        "keypoints": _build_keypoints_response(keypoints, mode),
        "angles": _build_angles_response(keypoints, mode),
        "visualization": _build_visualization(img, keypoints, mode),
    }

    # ── Research enhancements (each wrapped in try/except) ──────────
    enhanced = {}
    visuals = {}

    # Import visualizer
    try:
        try:
            from .viz import ResearchVisualizer
        except ImportError:
            from viz import ResearchVisualizer
        viz_available = True
    except ImportError:
        viz_available = False

    # --- 3D Bayesian Reconstruction ---
    try:
        bayes = _bayesian_engine.infer(center, tip1, tip2)
        if mode == "gauge":
            # Simplify: no hand assignment for gauges
            enhanced["reconstruction_3d"] = {
                "needle_depth": bayes.get("hand_depths", {}),
                "occlusion_risk": "N/A",
                "confidence": bayes.get("uncertainty", {}).get("confidence_score", 0.5),
                "credible_intervals": bayes.get("uncertainty", {}).get("credible_intervals"),
            }
        else:
            enhanced["reconstruction_3d"] = {
                "hand_assignment": bayes.get("hand_assignment"),
                "hand_depths": bayes.get("hand_depths"),
                "occlusion_risk": bayes.get("occlusion_risk", "UNKNOWN"),
                "confidence": bayes.get("uncertainty", {}).get("confidence_score", 0.5),
                "credible_intervals": bayes.get("uncertainty", {}).get("credible_intervals"),
            }
    except Exception as e:
        enhanced["reconstruction_3d"] = {
            "error": str(e), "confidence": 0.5,
            "occlusion_risk": "N/A" if mode == "gauge" else "UNKNOWN",
        }

    # --- Multi-Scale LVM Analysis ---
    try:
        if V3_EXTENSIONS_AVAILABLE:
            ms_result = _multiscale_extractor.extract(img)
            enhanced["scale_analysis"] = {
                "best_sigma": ms_result.get("best_sigma"),
                "scale_scores": ms_result.get("scale_scores", {}),
                "confidence_margin": ms_result.get("confidence", 0),
                "interpretation": ms_result.get("interpretation", ""),
            }
            if viz_available:
                visuals["scale_pyramid"] = ResearchVisualizer.render_scale_pyramid(
                    img,
                    ms_result.get("scale_scores", {}),
                    ms_result.get("best_sigma", 1.0),
                )
        else:
            enhanced["scale_analysis"] = {
                "best_sigma": 1.0, "scale_scores": {}, "confidence_margin": 0,
            }
    except Exception as e:
        enhanced["scale_analysis"] = {"error": str(e), "best_sigma": 1.0}

    # --- Manifold Curvature (downscaled for speed) ---
    try:
        if V3_EXTENSIONS_AVAILABLE:
            mf_size = 200
            H_orig, W_orig = img.shape[:2]
            sx, sy = mf_size / W_orig, mf_size / H_orig
            img_small = cv2.resize(img, (mf_size, mf_size))
            c_s = [center[0] * sx, center[1] * sy]
            t1_s = [tip1[0] * sx, tip1[1] * sy]
            t2_s = [tip2[0] * sx, tip2[1] * sy]
            mf_result = _manifold_detector.detect(img_small, c_s, t1_s, t2_s)
            enhanced["manifold"] = {
                "surface_classification": mf_result.get("manifold_analysis", {}).get(
                    "surface_classification", "FLAT"),
                "average_curvature_ratio": mf_result.get("manifold_analysis", {}).get(
                    "average_curvature_ratio", 1.0),
                "recommendation": mf_result.get("manifold_analysis", {}).get(
                    "recommendation", ""),
                "curvature_ratios": mf_result.get("curvature_ratios", {}),
            }
            if viz_available:
                visuals["curvature_heatmap"] = ResearchVisualizer.render_curvature_heatmap(
                    img,
                    mf_result.get("curvature_ratios", {}),
                    mf_result.get("manifold_analysis", {}).get(
                        "surface_classification", "FLAT"),
                )
        else:
            enhanced["manifold"] = {
                "surface_classification": "FLAT",
                "average_curvature_ratio": 1.0,
            }
    except Exception as e:
        enhanced["manifold"] = {"error": str(e), "surface_classification": "FLAT"}

    # --- Temporal Topology (single-frame snapshot) ---
    try:
        tracker_result = _temporal_tracker.add_frame(center, tip1, tip2)
        enhanced["temporal"] = {
            "beta0": tracker_result.get("topology", {}).get("betti_0", 1),
            "beta1": tracker_result.get("topology", {}).get("betti_1", 0),
            "topology_status": tracker_result.get("topology_change", "NOMINAL"),
            "occlusion_event": tracker_result.get("occlusion_event", False),
        }
    except Exception as e:
        enhanced["temporal"] = {
            "beta0": 1, "beta1": 0,
            "topology_status": "NOMINAL", "error": str(e),
        }

    # ── Generate visualization images ───────────────────────────────
    if viz_available:
        conf = enhanced.get("reconstruction_3d", {}).get("confidence", 0.5)
        occ = enhanced.get("reconstruction_3d", {}).get("occlusion_risk", "UNKNOWN")
        ha = enhanced.get("reconstruction_3d", {}).get("hand_assignment")
        best_s = enhanced.get("scale_analysis", {}).get("best_sigma", 1.0)
        surf = enhanced.get("manifold", {}).get("surface_classification", "FLAT")
        b0 = enhanced.get("temporal", {}).get("beta0", 1)
        b1 = enhanced.get("temporal", {}).get("beta1", 0)
        topo_status = enhanced.get("temporal", {}).get("topology_status", "NOMINAL")

        try:
            visuals["confidence_gauge"] = ResearchVisualizer.render_confidence_gauge(
                conf, occ, ha)
        except Exception:
            pass
        try:
            visuals["comparison"] = ResearchVisualizer.render_comparison(
                img, center, tip1, tip2, conf, occ, best_s)
        except Exception:
            pass
        try:
            visuals["betti_badge"] = ResearchVisualizer.render_betti_badge(
                b0, b1, topo_status)
        except Exception:
            pass
        try:
            visuals["impact_kpis"] = ResearchVisualizer.render_impact_kpis(
                conf, best_s, occ, surf, b0)
        except Exception:
            pass

    result["enhanced"] = enhanced
    result["research_visuals"] = visuals
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 4 — Probabilistic 3D Reconstruction (GAP 1)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/extract-skeleton-3d")
async def extract_skeleton_3d(
    file: UploadFile = File(...),
    mode: str = Query("clock", description="Instrument type: 'clock' or 'gauge'"),
):
    """
    Probabilistic 3D reconstruction with uncertainty quantification.

    For gauges: simplified — single needle depth estimation,
    no hand assignment, no occlusion risk between hands.
    """
    mode = _validate_mode(mode)

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    keypoints, error = _extract_keypoints_from_image(img, mode=mode)
    if error:
        return {"error": error, "mode": mode}

    center, tip1, tip2 = _unpack_for_3_point(keypoints, mode)

    # Run Bayesian inference
    result_3d = _bayesian_engine.infer(center, tip1, tip2)

    return {
        "mode": mode,
        "keypoints_2d": _build_keypoints_response(keypoints, mode),
        "angles_2d": _build_angles_response(keypoints, mode),
        "reconstruction_3d": result_3d,
        "visualization": _build_visualization(img, keypoints, mode),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 5 — Temporal Graph Dynamics (single frame, GAP 2)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/track-temporal")
async def track_temporal(frame: FrameKeypointsRequest):
    """
    Add a single frame to the temporal tracker and get topology analysis.

    For gauge mode: pass tip2 as None — tracker handles 2-node graphs.
    """
    result = _temporal_tracker.add_frame(
        center=frame.center,
        tip1=frame.tip1,
        tip2=frame.tip2,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 6 — Batch Sequence Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze-sequence")
async def analyze_sequence(sequence: SequenceRequest):
    """Analyze a complete sequence of frames at once."""
    tracker = TemporalGraphTracker(max_history=len(sequence.frames) + 10)
    frame_results = []

    for frame in sequence.frames:
        res = tracker.add_frame(
            center=frame.center,
            tip1=frame.tip1,
            tip2=frame.tip2,
        )
        frame_results.append(res)

    return {
        "frame_results": frame_results,
        "session_summary": tracker.get_session_summary(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 7 — Reset / Summary
# ═══════════════════════════════════════════════════════════════════════════════

@app.delete("/reset-tracker")
async def reset_tracker():
    """Reset the session temporal tracker state."""
    _temporal_tracker.reset()
    return {"message": "Temporal tracker reset successfully.", "frame_count": 0}


@app.get("/tracker-summary")
async def tracker_summary():
    """Get cumulative session summary from the temporal tracker."""
    return _temporal_tracker.get_session_summary()


# ═══════════════════════════════════════════════════════════════════════════════
# v3 Imports — Research Module Suite
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from .multiscale import MultiScaleSkeletonExtractor
    from .manifold import ManifoldSkeletonDetector
    from .causal import CausalSkeletonDiscovery
    from .lvm_temporal import LVMTemporalSmoother
    from .combination import LVMMultiScaleDetector
    V3_EXTENSIONS_AVAILABLE = True
except ImportError:
    try:
        from multiscale import MultiScaleSkeletonExtractor
        from manifold import ManifoldSkeletonDetector
        from causal import CausalSkeletonDiscovery
        from lvm_temporal import LVMTemporalSmoother
        from combination import LVMMultiScaleDetector
        V3_EXTENSIONS_AVAILABLE = True
    except ImportError as _e:
        print(f"[C2] v3 modules not available: {_e}")
        V3_EXTENSIONS_AVAILABLE = False

# ── v3 Singleton instances ───────────────────────────────────────────────────
if V3_EXTENSIONS_AVAILABLE:
    _multiscale_extractor = MultiScaleSkeletonExtractor()
    _manifold_detector = ManifoldSkeletonDetector()
    _causal_discovery = CausalSkeletonDiscovery()
    _combined_pipeline = LVMMultiScaleDetector()


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 8 — GAP 3: Multi-Scale LVM Extraction
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/extract-multiscale")
async def extract_multiscale(
    file: UploadFile = File(...),
    mode: str = Query("clock", description="Instrument type: 'clock' or 'gauge'"),
):
    """GAP 3 — Multi-scale extraction with LVM scale oracle (σ* selection).

    Domain-agnostic algorithm — works identically for clocks and gauges.
    """
    mode = _validate_mode(mode)
    if not V3_EXTENSIONS_AVAILABLE:
        return {"error": "v3 modules not available"}

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    keypoints, error = _extract_keypoints_from_image(img, mode=mode)
    if error:
        return {"error": error, "mode": mode}

    center, tip1, tip2 = _unpack_for_3_point(keypoints, mode)
    result = _multiscale_extractor.extract_with_yolo_keypoints(
        img, center, tip1, tip2)
    result.pop("all_graphs", None)
    result["mode"] = mode
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 9 — GAP 4: Non-Euclidean Manifold Skeleton
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/extract-manifold")
async def extract_manifold(
    file: UploadFile = File(...),
    mode: str = Query("clock", description="Instrument type: 'clock' or 'gauge'"),
):
    """GAP 4 — Riemannian geodesic skeleton with curvature analysis.

    Domain-agnostic — works for both curved dashboards (clocks)
    and cylindrical tanks (gauges).
    """
    mode = _validate_mode(mode)
    if not V3_EXTENSIONS_AVAILABLE:
        return {"error": "v3 modules not available"}

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    keypoints, error = _extract_keypoints_from_image(img, mode=mode)
    if error:
        return {"error": error, "mode": mode}

    center, tip1, tip2 = _unpack_for_3_point(keypoints, mode)
    result = _manifold_detector.detect(img, center, tip1, tip2)
    result["mode"] = mode
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 10 — GAP 5: Granger Causal Discovery
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze-causal")
async def analyze_causal(sequence: SequenceRequest):
    """GAP 5 — Granger causality from keypoint trajectories.

    Requires >= 20 frames. Works with both 2 and 3 keypoint layouts.
    """
    if not V3_EXTENSIONS_AVAILABLE:
        return {"error": "v3 modules not available"}

    frame_dicts = [
        {"center": f.center, "tip1": f.tip1, "tip2": f.tip2}
        for f in sequence.frames
    ]
    return _causal_discovery.discover(frame_dicts)


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 11 — LVM Temporal Smoothing
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/smooth-temporal")
async def smooth_temporal(sequence: SkeletonSequenceRequest):
    """LVM-gated temporal smoothing: ACCEPTED / INTERPOLATED / REJECTED."""
    if not V3_EXTENSIONS_AVAILABLE:
        return {"error": "v3 modules not available"}

    raw = [
        {"center": f.center, "tip1": f.tip1, "tip2": f.tip2}
        for f in sequence.frames
        if f.center and f.tip1
    ]
    smoother = LVMTemporalSmoother()
    smoothed_frames = smoother.process_sequence(raw)
    return {
        "smoothed_frames": smoothed_frames,
        "smoothing_stats": smoother.get_smoothing_stats(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 12 — RECOMMENDED: Combined Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/extract-combined")
async def extract_combined(
    file: UploadFile = File(...),
    mode: str = Query("clock", description="Instrument type: 'clock' or 'gauge'"),
):
    """
    RECOMMENDED — Full LVM-Guided Multi-Scale pipeline:
      scale oracle → Bayesian 3D → temporal smoothing → pipeline confidence.
    """
    mode = _validate_mode(mode)
    if not V3_EXTENSIONS_AVAILABLE:
        return {"error": "v3 modules not available"}

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    keypoints, error = _extract_keypoints_from_image(img, mode=mode)
    if error:
        return {"error": error, "mode": mode}

    center, tip1, tip2 = _unpack_for_3_point(keypoints, mode)
    combined = _combined_pipeline.process(img, center, tip1, tip2)
    response = combined.to_dict()
    response["mode"] = mode
    response["visualization"] = _build_visualization(img, keypoints, mode)
    return response


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 13/14 — Pipeline Stats & Reset
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/pipeline-stats")
async def pipeline_stats():
    """Combined pipeline session statistics."""
    if not V3_EXTENSIONS_AVAILABLE:
        return {"error": "v3 modules not available"}
    return _combined_pipeline.get_session_stats()


@app.delete("/reset-pipeline")
async def reset_pipeline():
    """Reset combined pipeline temporal state."""
    if not V3_EXTENSIONS_AVAILABLE:
        return {"error": "v3 modules not available"}
    _combined_pipeline.reset()
    return {"message": "Combined pipeline reset."}


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
