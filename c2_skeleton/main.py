"""
C2 Skeleton Service — Extended v3.0
=====================================
Hand Keypoint Extraction + Probabilistic 3D + Temporal Dynamics
+ Multi-Scale + Manifold + Causal + LVM Temporal + Combined Pipeline

Port   : 8002
Owner  : Member 2 (extended for research)

Endpoints
---------
Original (v1):
  GET  /health              — service health
  POST /extract-skeleton    — 2D keypoint extraction (YOLO)

Research v2 (GAP 1 & 2):
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
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import math
import os
from typing import List, Optional

# ── YOLO model ──────────────────────────────────────────────────────────────
from ultralytics import YOLO

# ── Research extension modules ───────────────────────────────────────────────
try:
    from .probabilistic_3d import BayesianGraphInference
    from .temporal_dynamics import TemporalGraphTracker
    EXTENSIONS_AVAILABLE = True
except ImportError:
    # Fallback for running main.py directly (not as package)
    from probabilistic_3d import BayesianGraphInference
    from temporal_dynamics import TemporalGraphTracker
    EXTENSIONS_AVAILABLE = True

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="C2 - Hand Skeleton Service (Extended)",
    description="Original 2D skeleton extraction + Probabilistic 3D Reconstruction + Temporal Graph Dynamics",
    version="2.0.0",
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
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

c2_model = None
try:
    print(f"[C2] Loading YOLO-Pose model: {MODEL_PATH}")
    c2_model = YOLO(MODEL_PATH)
    print("[C2] Model loaded successfully.")
except Exception as e:
    print(f"[C2] ⚠️ Model load failed: {e}")

# ── Singleton trackers (per-session state) ────────────────────────────────────
_bayesian_engine = BayesianGraphInference(k_hypotheses=10, image_size=500)
_temporal_tracker = TemporalGraphTracker(max_history=100)


# ═══════════════════════════════════════════════════════════════════════════════
# Request / Response schemas
# ═══════════════════════════════════════════════════════════════════════════════

class FrameKeypointsRequest(BaseModel):
    """Keypoints for a single frame, used for temporal tracking."""
    center: Optional[List[float]] = None    # [x, y]
    tip1:   Optional[List[float]] = None    # [x, y]
    tip2:   Optional[List[float]] = None    # [x, y]


class SequenceRequest(BaseModel):
    """Batch of frames for sequence analysis."""
    frames: List[FrameKeypointsRequest]


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _resize_small(img, size=500):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _encode_image(img) -> str:
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


def _get_angle(center, point) -> float:
    """Calculate clockwise angle from 12-o'clock position."""
    dx, dy = point[0] - center[0], point[1] - center[1]
    angle = math.degrees(math.atan2(dx, -dy))
    return angle + 360 if angle < 0 else angle


def _draw_skeleton(img, center, tip1, tip2):
    """Draw skeleton overlay on the image."""
    img_copy = img.copy()
    center_pt = (int(center[0]), int(center[1]))
    tip1_pt   = (int(tip1[0]),   int(tip1[1]))
    tip2_pt   = (int(tip2[0]),   int(tip2[1]))

    cv2.line(img_copy, center_pt, tip1_pt, (0, 255, 0), 4)
    cv2.line(img_copy, center_pt, tip2_pt, (0, 0, 255), 4)
    cv2.circle(img_copy, center_pt, 8, (255, 0, 0), -1)
    cv2.circle(img_copy, tip1_pt,   8, (0, 255, 0), -1)
    cv2.circle(img_copy, tip2_pt,   8, (0, 0, 255), -1)

    return _resize_small(img_copy)


def _extract_keypoints_from_image(img):
    """Run YOLO-Pose on image and return (center, tip1, tip2) or None."""
    if c2_model is None:
        return None, "C2 model not loaded"

    results = c2_model(img, verbose=False)[0]
    if not results.keypoints or len(results.keypoints.data) == 0:
        return None, "No hands/keypoints detected"

    kpts = results.keypoints.data[0].cpu().numpy()
    center = kpts[0][:2].tolist()
    tip1   = kpts[1][:2].tolist()
    tip2   = kpts[2][:2].tolist()
    return (center, tip1, tip2), None


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 1 (ORIGINAL — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Extended health check — also reports research module availability."""
    return {
        "service": "C2-Skeleton",
        "version": "2.0.0",
        "status": "ok",
        "model_loaded": c2_model is not None,
        "modules": {
            "probabilistic_3d": EXTENSIONS_AVAILABLE,
            "temporal_dynamics": EXTENSIONS_AVAILABLE,
        },
    }


@app.post("/extract-skeleton")
async def extract_skeleton(file: UploadFile = File(...)):
    """
    ORIGINAL ENDPOINT — Unchanged for backward compatibility.
    Extracts 2D keypoints and computes flat angles.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    keypoints, error = _extract_keypoints_from_image(img)
    if error:
        return {"error": error}

    center, tip1, tip2 = keypoints
    angle1 = _get_angle(center, tip1)
    angle2 = _get_angle(center, tip2)
    viz    = _draw_skeleton(img, center, tip1, tip2)

    return {
        "keypoints": {"center": center, "tip1": tip1, "tip2": tip2},
        "angles": {"hand1": round(angle1, 2), "hand2": round(angle2, 2)},
        "visualization": _encode_image(viz),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 1b (ENHANCED) — Full Research Analysis + Visual Outputs
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/extract-skeleton-enhanced")
async def extract_skeleton_enhanced(file: UploadFile = File(...)):
    """
    ENHANCED — Runs all C2 research algorithms and returns:
      - keypoints + angles (same as original)
      - 3D uncertainty (Bayesian inference)
      - Multi-scale LVM analysis (optimal σ*)
      - Manifold curvature (geodesic vs Euclidean)
      - Betti topology numbers
      - 6 pre-rendered base64 visualization images

    Used by C4 gateway to populate the Structure tab in the frontend.
    Falls back gracefully if any sub-module fails.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    keypoints, error = _extract_keypoints_from_image(img)
    if error:
        return {"error": error}

    center, tip1, tip2 = keypoints
    angle1 = _get_angle(center, tip1)
    angle2 = _get_angle(center, tip2)
    viz    = _draw_skeleton(img, center, tip1, tip2)

    # ── Base result (same as original) ──────────────────────────────
    result = {
        "keypoints": {"center": center, "tip1": tip1, "tip2": tip2},
        "angles": {"hand1": round(angle1, 2), "hand2": round(angle2, 2)},
        "visualization": _encode_image(viz),
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
        enhanced["reconstruction_3d"] = {
            "hand_assignment": bayes.get("hand_assignment"),
            "hand_depths": bayes.get("hand_depths"),
            "occlusion_risk": bayes.get("occlusion_risk", "UNKNOWN"),
            "confidence": bayes.get("uncertainty", {}).get("confidence_score", 0.5),
            "credible_intervals": bayes.get("uncertainty", {}).get("credible_intervals"),
        }
    except Exception as e:
        enhanced["reconstruction_3d"] = {"error": str(e), "confidence": 0.5, "occlusion_risk": "UNKNOWN"}

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
            enhanced["scale_analysis"] = {"best_sigma": 1.0, "scale_scores": {}, "confidence_margin": 0}
    except Exception as e:
        enhanced["scale_analysis"] = {"error": str(e), "best_sigma": 1.0}

    # --- Manifold Curvature (downscaled for speed) ---
    try:
        if V3_EXTENSIONS_AVAILABLE:
            # Downscale image + keypoints for fast Dijkstra
            mf_size = 200
            H_orig, W_orig = img.shape[:2]
            sx, sy = mf_size / W_orig, mf_size / H_orig
            img_small = cv2.resize(img, (mf_size, mf_size))
            c_s = [center[0]*sx, center[1]*sy]
            t1_s = [tip1[0]*sx, tip1[1]*sy]
            t2_s = [tip2[0]*sx, tip2[1]*sy]
            mf_result = _manifold_detector.detect(img_small, c_s, t1_s, t2_s)
            enhanced["manifold"] = {
                "surface_classification": mf_result.get("manifold_analysis", {}).get("surface_classification", "FLAT"),
                "average_curvature_ratio": mf_result.get("manifold_analysis", {}).get("average_curvature_ratio", 1.0),
                "recommendation": mf_result.get("manifold_analysis", {}).get("recommendation", ""),
                "curvature_ratios": mf_result.get("curvature_ratios", {}),
            }
            if viz_available:
                visuals["curvature_heatmap"] = ResearchVisualizer.render_curvature_heatmap(
                    img,
                    mf_result.get("curvature_ratios", {}),
                    mf_result.get("manifold_analysis", {}).get("surface_classification", "FLAT"),
                )
        else:
            enhanced["manifold"] = {"surface_classification": "FLAT", "average_curvature_ratio": 1.0}
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
        enhanced["temporal"] = {"beta0": 1, "beta1": 0, "topology_status": "NOMINAL", "error": str(e)}

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
            visuals["confidence_gauge"] = ResearchVisualizer.render_confidence_gauge(conf, occ, ha)
        except Exception:
            pass
        try:
            visuals["comparison"] = ResearchVisualizer.render_comparison(
                img, center, tip1, tip2, conf, occ, best_s
            )
        except Exception:
            pass
        try:
            visuals["betti_badge"] = ResearchVisualizer.render_betti_badge(b0, b1, topo_status)
        except Exception:
            pass
        try:
            visuals["impact_kpis"] = ResearchVisualizer.render_impact_kpis(
                conf, best_s, occ, surf, b0
            )
        except Exception:
            pass

    result["enhanced"] = enhanced
    result["research_visuals"] = visuals
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 2 (NEW) — Probabilistic 3D Reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/extract-skeleton-3d")
async def extract_skeleton_3d(file: UploadFile = File(...)):
    """
    NEW — Probabilistic 3D reconstruction with uncertainty quantification.

    Algorithm (GAP 1):
      1. Extract 2D keypoints via YOLO-Pose
      2. Sample K=10 candidate 3D structures from the graph prior
      3. Score each via rendering likelihood P(I|G)
      4. Return MAP estimate + full posterior uncertainty

    Returns
    -------
    JSON with:
      keypoints_2d       — observed 2D keypoints (from YOLO)
      angles_2d          — standard 2D angles
      reconstruction_3d  — MAP 3D structure (including Z depth)
      uncertainty        — credible intervals, angle std, confidence
      hand_assignment    — which hand is hour vs minute
      occlusion_risk     — LOW / MEDIUM / HIGH
      summary            — human-readable interpretation
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    keypoints, error = _extract_keypoints_from_image(img)
    if error:
        return {"error": error}

    center, tip1, tip2 = keypoints

    # Run Bayesian inference
    result_3d = _bayesian_engine.infer(center, tip1, tip2)

    # Standard 2D angles (for comparison)
    angle1 = _get_angle(center, tip1)
    angle2 = _get_angle(center, tip2)
    viz    = _draw_skeleton(img, center, tip1, tip2)

    return {
        "keypoints_2d": {"center": center, "tip1": tip1, "tip2": tip2},
        "angles_2d": {"hand1": round(angle1, 2), "hand2": round(angle2, 2)},
        "reconstruction_3d": result_3d,
        "visualization": _encode_image(viz),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 3 (NEW) — Temporal Graph Dynamics (single frame)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/track-temporal")
async def track_temporal(frame: FrameKeypointsRequest):
    """
    NEW — Add a single frame to the temporal tracker and get topology analysis.

    Algorithm (GAP 2):
      - Updates Betti numbers (β₀, β₁) for this frame
      - Detects birth/death of topological features
      - If connectivity changed, classifies as OCCLUSION or TOPOLOGY_CHANGE
      - Returns motion velocity/acceleration analysis

    Call this endpoint for each frame in a video for live tracking.
    Use DELETE /reset-tracker to start a new video sequence.

    Body (JSON):
      { "center": [x, y], "tip1": [x, y], "tip2": [x, y] }
    """
    result = _temporal_tracker.add_frame(
        center=frame.center,
        tip1=frame.tip1,
        tip2=frame.tip2,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 4 (NEW) — Batch sequence analysis
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze-sequence")
async def analyze_sequence(sequence: SequenceRequest):
    """
    NEW — Analyze a complete sequence of frames at once.

    Resets the tracker, processes all frames, and returns a full session summary
    with the Betti number time series, persistence diagram, and events log.

    Body (JSON):
      { "frames": [ {"center": [...], "tip1": [...], "tip2": [...]}, ... ] }
    """
    # Fresh tracker for this sequence
    tracker = TemporalGraphTracker(max_history=len(sequence.frames) + 10)
    frame_results = []

    for frame in sequence.frames:
        res = tracker.add_frame(
            center=frame.center,
            tip1=frame.tip1,
            tip2=frame.tip2,
        )
        frame_results.append(res)

    session_summary = tracker.get_session_summary()

    return {
        "frame_results": frame_results,
        "session_summary": session_summary,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 5 (NEW) — Reset temporal tracker
# ═══════════════════════════════════════════════════════════════════════════════

@app.delete("/reset-tracker")
async def reset_tracker():
    """
    NEW — Reset the session temporal tracker state.
    Call this when starting a new video sequence.
    """
    _temporal_tracker.reset()
    return {"message": "Temporal tracker reset successfully.", "frame_count": 0}


@app.get("/tracker-summary")
async def tracker_summary():
    """NEW — Get cumulative session summary from the temporal tracker."""
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
    _manifold_detector    = ManifoldSkeletonDetector()
    _causal_discovery     = CausalSkeletonDiscovery()
    _combined_pipeline    = LVMMultiScaleDetector()


# ── v3 Request Schemas ───────────────────────────────────────────────────────

class SkeletonSequenceRequest(BaseModel):
    """Batch of skeletons for temporal smoothing or causal analysis."""
    frames: List[FrameKeypointsRequest]


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 6 — GAP 3: Multi-Scale LVM Extraction
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/extract-multiscale")
async def extract_multiscale(file: UploadFile = File(...)):
    """GAP 3 — Multi-scale extraction with LVM scale oracle (σ* selection)."""
    if not V3_EXTENSIONS_AVAILABLE:
        return {"error": "v3 modules not available"}
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    keypoints, error = _extract_keypoints_from_image(img)
    if error:
        return {"error": error}
    center, tip1, tip2 = keypoints
    result = _multiscale_extractor.extract_with_yolo_keypoints(img, center, tip1, tip2)
    result.pop("all_graphs", None)   # remove heavy nested data
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 7 — GAP 4: Non-Euclidean Manifold Skeleton
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/extract-manifold")
async def extract_manifold(file: UploadFile = File(...)):
    """GAP 4 — Riemannian geodesic skeleton with curvature analysis."""
    if not V3_EXTENSIONS_AVAILABLE:
        return {"error": "v3 modules not available"}
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    keypoints, error = _extract_keypoints_from_image(img)
    if error:
        return {"error": error}
    center, tip1, tip2 = keypoints
    return _manifold_detector.detect(img, center, tip1, tip2)


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 8 — GAP 5: Granger Causal Discovery
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze-causal")
async def analyze_causal(sequence: SequenceRequest):
    """
    GAP 5 — Granger causality from keypoint trajectories.
    Requires >= 20 frames. Body: {"frames": [{center, tip1, tip2}, ...]}
    """
    if not V3_EXTENSIONS_AVAILABLE:
        return {"error": "v3 modules not available"}
    frame_dicts = [
        {"center": f.center, "tip1": f.tip1, "tip2": f.tip2}
        for f in sequence.frames
    ]
    return _causal_discovery.discover(frame_dicts)


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 9 — LVM Temporal Smoothing
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/smooth-temporal")
async def smooth_temporal(sequence: SkeletonSequenceRequest):
    """
    LVM-gated temporal smoothing: ACCEPTED / INTERPOLATED / REJECTED per frame.
    Body: {"frames": [{center, tip1, tip2}, ...]}
    """
    if not V3_EXTENSIONS_AVAILABLE:
        return {"error": "v3 modules not available"}
    raw = [
        {"center": f.center, "tip1": f.tip1, "tip2": f.tip2}
        for f in sequence.frames
        if f.center and f.tip1 and f.tip2
    ]
    smoother = LVMTemporalSmoother()
    smoothed_frames = smoother.process_sequence(raw)
    return {
        "smoothed_frames": smoothed_frames,
        "smoothing_stats": smoother.get_smoothing_stats(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 10 — RECOMMENDED: Combined Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/extract-combined")
async def extract_combined(file: UploadFile = File(...)):
    """
    RECOMMENDED — Full LVM-Guided Multi-Scale pipeline:
      scale oracle → Bayesian 3D → temporal smoothing → pipeline confidence.
    """
    if not V3_EXTENSIONS_AVAILABLE:
        return {"error": "v3 modules not available"}
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    keypoints, error = _extract_keypoints_from_image(img)
    if error:
        return {"error": error}
    center, tip1, tip2 = keypoints
    combined = _combined_pipeline.process(img, center, tip1, tip2)
    response = combined.to_dict()
    response["visualization"] = _encode_image(_draw_skeleton(img, center, tip1, tip2))
    return response


@app.get("/pipeline-stats")
async def pipeline_stats():
    """v3 — Combined pipeline session statistics."""
    if not V3_EXTENSIONS_AVAILABLE:
        return {"error": "v3 modules not available"}
    return _combined_pipeline.get_session_stats()


@app.delete("/reset-pipeline")
async def reset_pipeline():
    """v3 — Reset combined pipeline temporal state."""
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
