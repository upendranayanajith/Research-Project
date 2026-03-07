import streamlit as st
import requests
import base64
from PIL import Image
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av
import cv2
import numpy as np
import time
import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"
st.set_page_config(page_title="Chronos Vision", layout="wide", page_icon="static/favicon.ico")

# ── Google Material Symbols ───────────────────────────────────────────────────
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">', unsafe_allow_html=True)

def icon(name, size=24, color="inherit", vertical_align="middle"):
    return f'<span class="material-symbols-outlined" style="font-size:{size}px; color:{color}; vertical-align:{vertical_align};">{name}</span>'


# ── Status colour helpers ─────────────────────────────────────────────────────
STATUS_COLORS = {
    "normal":       "#22c55e",   # green
    "warning":      "#f59e0b",   # amber
    "danger":       "#ef4444",   # red
    "out_of_range": "#8b5cf6",   # purple
    "error":        "#6b7280",   # grey
}
STATUS_ICONS = {
    "normal":       "check_circle",
    "warning":      "warning",
    "danger":       "error",
    "out_of_range": "help_outline",
    "error":        "bug_report",
}


# ============================================================
# [C1 & C2] REAL-TIME PROCESSOR  (clock — unchanged)
# ============================================================
class ClockProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.fps         = 0
        self.last_time   = time.time()
        self.force_expert = False
        self.last_result  = None

        from app.core.engine import ClockEngine
        self.engine = ClockEngine(parent_dir)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        now = time.time()
        if now - self.last_time > 1:
            self.fps         = self.frame_count
            self.frame_count = 0
            self.last_time   = now

        if self.frame_count % 5 == 0:
            try:
                self.last_result = self.engine.analyze(img, force_expert=self.force_expert)
            except Exception as e:
                print(f"AI Error: {e}")

        if self.last_result:
            res = self.last_result
            cv2.putText(img, f"TIME: {res.get('time', '--:--')}", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
            method = res.get('method', 'Unknown')
            color  = (0, 255, 0) if "Fast" in method else (0, 0, 255)
            cv2.putText(img, f"Mode: {method}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if "angles" in res:
                a1 = res["angles"]["hand1"]
                a2 = res["angles"]["hand2"]
                cv2.putText(img, f"H:{a1:.0f} M:{a2:.0f}", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(img, f"FPS: {self.fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================
# SHARED HELPERS
# ============================================================
def display_results(data):
    """Render full clock analysis result (original — unchanged)."""
    res         = data["result"]
    viz         = data.get("visualizations", {})
    ampm_data   = data.get("ampm", {})
    amb_data    = data.get("ambiguity", {})
    report_data = data.get("report", {})

    if "error" in res:
        st.markdown(f"#### {icon('error', color='red')} Analysis Failed", unsafe_allow_html=True)
        st.error(res['error'])
        return

    st.markdown(f"#### {icon('check_circle', color='green')} Analysis Complete ({data['processing_time']:.3f}s)", unsafe_allow_html=True)

    is_fast      = "Fast Path" in res["method"]
    method_icon  = "bolt" if is_fast else "psychology"
    method_color = "green" if is_fast else "orange"
    st.markdown(f"**Method Used:** <span style='color:{method_color}'>{icon(method_icon, size=20)} {res['method']}</span>", unsafe_allow_html=True)

    stages = [
        ("C1 Localization", "crop_free",      ["C1", "C2", "C4"]),
        ("C2 Structure",    "timeline",        ["C1", "C2", "C4"]),
        ("C3 Expert AI",    "model_training",  ["Expert"]),
        ("C4 Physics",      "functions",       ["C1", "C2", "C4"]),
    ]
    cols = st.columns(4)
    for col, (name, icn, active_list) in zip(cols, stages):
        is_active = False
        if "Expert" in res["method"]:
            is_active = True
        elif name.split()[0] in active_list and "Fast" in res["method"] and "Expert" not in name:
            is_active = True
        color = "green" if is_active else "grey"
        col.markdown(f"{icon(icn, color=color)} {name}", unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Localization", "Structure", "Expert AI", "Result", "Report"])

    with tab1:
        st.markdown(f"{icon('crop_free')} **YOLO Localization**", unsafe_allow_html=True)
        c1q      = data.get("c1_quality") or {}
        det_conf = c1q.get("confidence")
        quality  = c1q.get("quality") or {}

        col_img, col_stats = st.columns([1, 1])
        with col_img:
            if "c1_detection" in viz:
                st.image(base64.b64decode(viz["c1_detection"]), use_container_width=True)
        with col_stats:
            if det_conf is not None:
                pct = int(det_conf * 100)
                color = "#3782eb" if pct >= 80 else ("#39F91FE9" if pct >= 60 else "#ef4444")
                tier  = "HIGH"    if pct >= 80 else ("MEDIUM"    if pct >= 60 else "LOW")
                st.markdown(
                    f"""Detection Confidence :
                    <span style='font-size:1rem; font-weight:700'>{pct}%</span><br><br>
                    <span style='font-weight:400'>{tier}</span>
                    <div style='background:#e0e0e0; border-radius:6px; height:8px; margin:4px 0 14px'>
                      <div style='background:{color}; width:{pct}%; height:8px; border-radius:6px'></div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        st.markdown("---")
        if quality:
            overall = quality.get("overall_quality", 0)
            st.markdown(f"""Image Quality Score : <span style='font-weight:500;'><b>{overall:.0f} / 100</b></span>""", unsafe_allow_html=True)
            metrics_info = [
                ("Sharpness (Blur)",  quality.get("blur_score", 0),       quality.get("blur_raw", 0),       "Laplacian variance"),
                ("Brightness",        quality.get("brightness_score", 0), quality.get("brightness_raw", 0), "Mean pixel value"),
                ("Contrast",          quality.get("contrast_score", 0),   quality.get("contrast_raw", 0),   "Pixel std deviation"),
            ]
            cols_m = st.columns(3)
            for col_m, (label, score, raw, raw_label) in zip(cols_m, metrics_info):
                bar_c = "#3782eb" if score >= 75 else ("#39F91FE9" if score >= 45 else "#ef4444")
                col_m.markdown(
                    f"**{label}**<br>"
                    f"<span style='font-size:1.3rem; font-weight:700'>{score:.0f}</span>"
                    f"<span style='font-size:1rem; color:#334155'>/100</span><br>"
                    f"<span style='font-size:.9rem; color:#4b5563'>{raw_label}: {raw:.1f}</span>",
                    unsafe_allow_html=True,
                )
                col_m.markdown(
                    f"""<div style='background:#1e293b; border-radius:6px; height:8px; margin:4px 0'>
                      <div style='background:{bar_c}; width:{score}%; height:8px; border-radius:6px'></div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    with tab2:
        st.markdown(f"{icon('timeline')} **C2 — Skeleton Structure Analysis**", unsafe_allow_html=True)
        c2e = data.get("c2_enhanced", {})
        c2v = data.get("c2_research_visuals", {})
        if not c2e:
            if "c2_skeleton" in viz:
                st.image(base64.b64decode(viz["c2_skeleton"]), width=350)
            st.info("Enhanced C2 data not available.")
        else:
            s1, s2, s3, s4, s5, s6 = st.tabs(["🦴 Skeleton", "🔭 Scale Analysis", "🧊 3D Reconstruction", "🌍 Manifold", "⏱ Temporal", "📊 Impact Summary"])
            with s1:
                col_l, col_r = st.columns([2, 1])
                with col_l:
                    if "c2_skeleton" in viz:
                        st.image(base64.b64decode(viz["c2_skeleton"]), caption="YOLO-Pose Skeleton", width=350)
                with col_r:
                    kp = data.get("result", {}).get("angles", res.get("angles", {}))
                    if kp:
                        st.metric("Hand 1 Angle", f"{kp.get('hand1', 0):.1f}°")
                        st.metric("Hand 2 Angle", f"{kp.get('hand2', 0):.1f}°")
            with s2:
                sa = c2e.get("scale_analysis", {})
                st.markdown("**GAP 3 — Multi-Scale LVM Oracle**")
                if c2v.get("scale_pyramid"):
                    st.image(base64.b64decode(c2v["scale_pyramid"]), caption="Scale Pyramid + LVM Scores", use_container_width=True)
                col_a, col_b = st.columns(2)
                col_a.metric("Optimal Scale σ*", sa.get("best_sigma", "—"))
                col_b.metric("Confidence Margin", f"{sa.get('confidence_margin', 0):.3f}")
                if sa.get("interpretation"):
                    st.info(sa["interpretation"])
            with s3:
                r3d = c2e.get("reconstruction_3d", {})
                st.markdown("**GAP 1 — Bayesian 3D Reconstruction**")
                if c2v.get("confidence_gauge"):
                    st.image(base64.b64decode(c2v["confidence_gauge"]), caption="Confidence Gauge", width=350)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Confidence", f"{r3d.get('confidence', 0):.3f}")
                col_b.metric("Occlusion Risk", r3d.get("occlusion_risk", "—"))
                ha = r3d.get("hand_assignment", {})
                if ha:
                    col_c.metric("Hour Hand", ha.get("hour", "—"))
                depths = r3d.get("hand_depths", {})
                if depths:
                    with st.expander("Depth Estimates"):
                        st.json(depths)
            with s4:
                mf = c2e.get("manifold", {})
                st.markdown("**GAP 4 — Non-Euclidean Manifold Skeleton**")
                if c2v.get("curvature_heatmap"):
                    st.image(base64.b64decode(c2v["curvature_heatmap"]), caption="Curvature Analysis", use_container_width=True)
                col_a, col_b = st.columns(2)
                col_a.metric("Surface Class", mf.get("surface_classification", "—"))
                col_b.metric("Avg Curvature Ratio", f"{mf.get('average_curvature_ratio', 1.0):.3f}")
                if mf.get("recommendation"):
                    st.info(mf["recommendation"])
            with s5:
                tmp = c2e.get("temporal", {})
                st.markdown("**GAP 2 — Persistent Homology Tracking**")
                if c2v.get("betti_badge"):
                    st.image(base64.b64decode(c2v["betti_badge"]), caption="Topology Status", width=300)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("β₀ (Components)", tmp.get("beta0", "—"))
                col_b.metric("β₁ (Loops)", tmp.get("beta1", "—"))
                col_c.metric("Status", tmp.get("topology_status", "—"))
            with s6:
                st.markdown("**Research Impact Summary**")
                if c2v.get("comparison"):
                    st.image(base64.b64decode(c2v["comparison"]), use_container_width=True)
                if c2v.get("impact_kpis"):
                    st.image(base64.b64decode(c2v["impact_kpis"]), use_container_width=True)

    with tab3:
        st.markdown(f"{icon('psychology')} **Angle Predictions**", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if "c3_angles" in viz:
                st.image(base64.b64decode(viz["c3_angles"]), caption="Angle Visual", width=300)
        with col_b:
            if "angles" in res:
                st.markdown(f"**H:** {res['angles']['hand1']:.1f}°")
                st.markdown(f"**M:** {res['angles']['hand2']:.1f}°")
        if "c3_crops" in viz and viz["c3_crops"]:
            st.markdown("---")
            c_cols = st.columns(len(viz["c3_crops"]))
            for col, crop in zip(c_cols, viz["c3_crops"]):
                col.image(base64.b64decode(crop), width=100)
            if data.get("heatmap_b64"):
                st.image(base64.b64decode(data["heatmap_b64"]), width=300)
        else:
            st.info("Fast Path Used — Expert AI skipped.")

    with tab4:
        st.markdown(f"# {icon('schedule')} {res['time']}", unsafe_allow_html=True)
        unc    = res.get('uncertainty', '')
        c2_conf = res.get('c2_confidence', 0)
        c2_occ  = res.get('c2_occlusion_risk', 'UNKNOWN')
        c2_ha   = res.get('c2_hand_assignment', {})
        if unc:
            st.markdown(f"**Uncertainty:** `{res['time']} {unc}`", unsafe_allow_html=True)
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("C2 Confidence", f"{c2_conf:.2f}" if c2_conf else "—")
        occ_icon = "✅" if c2_occ == "LOW" else ("⚠️" if c2_occ == "MEDIUM" else "🔴")
        col_r2.metric("Occlusion Risk", f"{occ_icon} {c2_occ}")
        col_r3.metric("Hour Hand", c2_ha.get('hour', '—'))
        col_r4.metric("Minute Hand", c2_ha.get('minute', '—'))
        if c2_occ == "HIGH":
            st.warning("High occlusion risk — result may be less reliable.")
        st.markdown("---")
        st.markdown(f"**Reasoning:** `{res.get('reasoning', 'N/A')}`")
        st.markdown("---")
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"#### {icon('wb_sunny')} AM/PM Inference", unsafe_allow_html=True)
            if ampm_data:
                period   = ampm_data.get("period", "Unknown")
                p_conf   = ampm_data.get("confidence", 0)
                p_reason = ampm_data.get("reason", "")
                p_color  = "orange" if period == "Unknown" else "green"
                st.markdown(f"**Period:** <span style='color:{p_color};font-size:1.3em'>{period}</span>  ({p_conf:.0f}% confidence)", unsafe_allow_html=True)
                st.caption(p_reason)
            else:
                st.info("Not available.")
        with col_r:
            st.markdown(f"#### {icon('help_outline')} Ambiguity Analysis", unsafe_allow_html=True)
            if amb_data:
                is_amb    = amb_data.get("is_ambiguous", False)
                amb_color = "orange" if is_amb else "green"
                st.markdown(f"{icon('warning' if is_amb else 'check_circle', color=amb_color)} {amb_data.get('ambiguity_reason', '')}", unsafe_allow_html=True)
                candidates = amb_data.get("top_candidates", [])
                if candidates:
                    cdf = pd.DataFrame([{
                        "Time": c["time"], "Error (°)": c["angular_error"],
                        "Confidence %": c["confidence_pct"], "Fit": c["fit_quality"],
                    } for c in candidates])
                    st.dataframe(cdf, use_container_width=True, hide_index=True)
            else:
                st.info("Not available.")

    with tab5:
        if report_data:
            st.markdown(f"## {icon('description')} {report_data.get('title','')}", unsafe_allow_html=True)
            st.caption(f"Generated at {report_data.get('generated_at','')} — {report_data.get('one_liner','')}")
            st.markdown("---")
            for section in report_data.get("sections", []):
                st.markdown(f"### {section['heading']}")
                st.markdown(section["body"])
        else:
            st.info("Report not available for this analysis.")


def display_gauge_reading(reading: dict, show_title: bool = True):
    """
    Render a single GaugeReading dict in a clean card-style layout.
    Reusable across both image upload and CCTV pages.
    """
    status      = reading.get("status", "normal")
    status_col  = STATUS_COLORS.get(status, "#6b7280")
    status_icn  = STATUS_ICONS.get(status, "info")
    value       = reading.get("value", 0)
    unit        = reading.get("unit", "")
    gauge_type  = reading.get("gauge_type", "—")
    percentage  = reading.get("percentage", 0)
    confidence  = reading.get("confidence", 0)
    raw_angle   = reading.get("raw_angle_deg", 0)
    detail      = reading.get("status_detail", "")

    if show_title:
        st.markdown(
            f"#### {icon(status_icn, color=status_col)} Gauge Reading",
            unsafe_allow_html=True,
        )

    # ── Big value display ─────────────────────────────────────────────────────
    st.markdown(
        f"""<div style='
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border: 2px solid {status_col};
            border-radius: 16px;
            padding: 28px 32px;
            text-align: center;
            margin-bottom: 16px;
        '>
            <div style='font-size: 3.5rem; font-weight: 800; color: {status_col}; line-height: 1;'>
                {value} <span style='font-size: 1.8rem; color: #94a3b8;'>{unit}</span>
            </div>
            <div style='font-size: 0.9rem; color: #64748b; margin-top: 8px;'>{gauge_type}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Percentage bar ────────────────────────────────────────────────────────
    st.markdown(
        f"""<div style='margin-bottom:12px;'>
            <div style='font-size:0.85rem; color:#94a3b8; margin-bottom:4px;'>
                Scale position: <b style='color:#e2e8f0;'>{percentage}%</b>
            </div>
            <div style='background:#1e293b; border-radius:8px; height:12px;'>
                <div style='background:{status_col}; width:{min(percentage,100)}%;
                            height:12px; border-radius:8px;'></div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status",      status.capitalize())
    m2.metric("Confidence",  f"{confidence:.2f}")
    m3.metric("Needle Angle", f"{raw_angle:.1f}°")
    m4.metric("Scale %",     f"{percentage}%")

    if detail:
        if status == "danger":
            st.error(detail)
        elif status == "warning":
            st.warning(detail)
        else:
            st.success(detail)


# ============================================================
# NAVIGATION
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "analysis"

def nav_button(page_key, label, icon_name):
    c1, c2 = st.sidebar.columns([1, 4])
    with c1:
        st.markdown(f"<div style='text-align:center; padding-top:5px;'>{icon(icon_name)}</div>", unsafe_allow_html=True)
    with c2:
        btn_type = "primary" if st.session_state.page == page_key else "secondary"
        if st.button(label, key=f"nav_{page_key}", type=btn_type, use_container_width=True):
            st.session_state.page = page_key
            st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
logo_path = os.path.join(current_dir, "..", "assets", "images", "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.markdown(f"### {icon('menu')} Navigation", unsafe_allow_html=True)
st.sidebar.markdown("---")

# ── Clock section ─────────────────────────────────────────────────────────────
st.sidebar.markdown(
    f"<div style='font-size:0.75rem; color:#64748b; text-transform:uppercase; "
    f"letter-spacing:0.08em; padding: 4px 0 2px 4px;'>⏱ Clock Reading</div>",
    unsafe_allow_html=True,
)
nav_button("analysis",   "File Analysis",    "cloud_upload")
nav_button("webcam",     "Live Webcam",      "videocam")
nav_button("batch",      "Batch Processing", "perm_media")
nav_button("comparator", "Clock Comparator", "compare_arrows")
nav_button("accuracy",   "Accuracy Checker", "verified")
nav_button("dashboard",  "Analytics",        "monitoring")

# ── Gauge section ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div style='font-size:0.75rem; color:#64748b; text-transform:uppercase; "
    f"letter-spacing:0.08em; padding: 4px 0 2px 4px;'>🔧 Gauge Reading</div>",
    unsafe_allow_html=True,
)
nav_button("gauge_image",    "Gauge Image",       "image_search")
nav_button("gauge_stream",   "Gauge Live Stream", "sensors")
nav_button("gauge_multi",    "Multi-Gauge Scene", "dashboard")
nav_button("gauge_register", "Register Gauge",    "add_circle")

st.sidebar.markdown("---")


# ============================================================
# CLOCK PAGES  (original — unchanged)
# ============================================================

if st.session_state.page == "analysis":
    st.markdown(f"## {icon('cloud_upload')} File Analysis", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    st.markdown("---")
    st.markdown(f"#### {icon('settings')} Configuration", unsafe_allow_html=True)
    force_expert = st.checkbox("Force Expert Path (Activate C3 + XAI)", value=False)
    if uploaded_file and st.button("Run Analysis", type="primary"):
        with st.spinner("Processing..."):
            try:
                image        = Image.open(uploaded_file)
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                files     = {"file": ("image.jpg", img_byte_arr.getvalue(), "image/jpeg")}
                data_form = {"force_expert": str(force_expert)}
                response  = requests.post(f"{API_URL}/analyze", files=files, data=data_form)
                if response.status_code == 200:
                    display_results(response.json())
                else:
                    st.error(f"Server Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

elif st.session_state.page == "webcam":
    st.markdown(f"## {icon('videocam')} Real-Time Analysis", unsafe_allow_html=True)
    st.info("Running C1 (Localization) + C2 (Pose) locally. C4 runs on every 5th frame.")
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    col1, col2 = st.columns([3, 1])
    with col1:
        ctx = webrtc_streamer(
            key="clock-ai", video_processor_factory=ClockProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    with col2:
        st.markdown(f"### {icon('tune')} Controls", unsafe_allow_html=True)
        if ctx.video_processor:
            st.markdown(f"{icon('military_tech')} **Force Expert Mode**", unsafe_allow_html=True)
            ctx.video_processor.force_expert = st.checkbox("", value=False)
        st.markdown("---")
        if st.button("Reset Connection"):
            st.cache_resource.clear()
            st.rerun()

elif st.session_state.page == "batch":
    st.markdown(f"## {icon('perm_media')} Batch Processing", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)
    if uploaded_files and st.button("Process All"):
        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
        with st.spinner("Processing Batch..."):
            try:
                res = requests.post(f"{API_URL}/analyze_batch", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.markdown(f"#### {icon('check_circle')} Processed {data['total_images']} images", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(data["results"]), use_container_width=True)
                else:
                    st.error("Batch failed.")
            except Exception as e:
                st.error(f"Error: {e}")

elif st.session_state.page == "comparator":
    st.markdown(f"## {icon('compare_arrows')} Clock Comparator — Elapsed Time", unsafe_allow_html=True)
    st.info("Upload two clock images (a **Before** and an **After** shot). C4 will compute how much time elapsed between them.")
    st.markdown("---")
    col_s, col_e = st.columns(2)
    with col_s:
        st.markdown(f"#### {icon('hourglass_top')} Before Clock", unsafe_allow_html=True)
        f_start = st.file_uploader("Start clock image", type=["jpg","png","jpeg"], key="cmp_start")
        sp = st.selectbox("AM/PM (optional)", ["— unknown —","AM","PM"], key="sp_sel")
    with col_e:
        st.markdown(f"#### {icon('hourglass_bottom')} After Clock", unsafe_allow_html=True)
        f_end = st.file_uploader("End clock image", type=["jpg","png","jpeg"], key="cmp_end")
        ep = st.selectbox("AM/PM (optional)", ["— unknown —","AM","PM"], key="ep_sel")
    if f_start and f_end and st.button("Calculate Elapsed Time", type="primary"):
        with st.spinner("Reading both clocks..."):
            try:
                sp_val = None if sp == "— unknown —" else sp
                ep_val = None if ep == "— unknown —" else ep
                files_payload = [
                    ("file_start", (f_start.name, f_start.getvalue(), "image/jpeg")),
                    ("file_end",   (f_end.name,   f_end.getvalue(),   "image/jpeg")),
                ]
                form_data = {}
                if sp_val: form_data["start_period"] = sp_val
                if ep_val: form_data["end_period"]   = ep_val
                res = requests.post(f"{API_URL}/compare-clocks", files=files_payload, data=form_data)
                if res.status_code == 200:
                    d = res.json()
                    if "error" in d:
                        st.error(d["error"])
                    else:
                        mp = d["elapsed"]["most_probable"]
                        st.success(
                            f"**Start:** {d['start_clock_reading']}  →  "
                            f"**End:** {d['end_clock_reading']}  →  "
                            f"**Elapsed:** {mp['elapsed_display']} ({mp['direction']})"
                        )
                        st.markdown("---")
                        st.markdown(f"#### {icon('list')} All Possible Spans", unsafe_allow_html=True)
                        spans = d["elapsed"]["all_spans"]
                        sdf = pd.DataFrame([{
                            "From": s["from"], "To": s["to"],
                            "Elapsed": s["elapsed_display"], "Minutes": s["elapsed_minutes"],
                            "Direction": s["direction"], "Plausibility": s["plausibility"],
                        } for s in spans])
                        st.dataframe(sdf, use_container_width=True, hide_index=True)
                        st.caption(d["elapsed"]["notes"])
                else:
                    st.error(f"Server error: {res.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")
    elif not (f_start and f_end):
        st.warning("Please upload both a Before and After clock image.")

elif st.session_state.page == "accuracy":
    st.markdown(f"## {icon('verified')} Clock Accuracy Checker", unsafe_allow_html=True)
    st.info("Upload a clock photo. C4 reads the time, compares it to your device's current real time, and reports the drift.")
    st.markdown("---")
    col_up, col_cfg = st.columns([2, 1])
    with col_up:
        acc_file = st.file_uploader("Clock image", type=["jpg","png","jpeg"], key="acc_file")
    with col_cfg:
        period_hint = st.selectbox("AM/PM hint", ["— unknown —","AM","PM"], key="acc_period")
        tz_offset   = st.number_input("UTC offset (hours)", value=0.0, step=0.5, min_value=-12.0, max_value=14.0, key="acc_tz")
    if acc_file and st.button("Check Accuracy", type="primary"):
        with st.spinner("Analysing clock..."):
            try:
                pv      = None if period_hint == "— unknown —" else period_hint
                files_p = {"file": (acc_file.name, acc_file.getvalue(), "image/jpeg")}
                form_p  = {"tz_offset_hours": str(tz_offset)}
                if pv: form_p["period"] = pv
                res = requests.post(f"{API_URL}/analyze-with-accuracy", files=files_p, data=form_p)
                if res.status_code == 200:
                    d       = res.json()
                    if "error" in d:
                        st.error(d["error"])
                    else:
                        acc     = d["accuracy"]
                        verdict = acc["verdict"]
                        v_color = {"Accurate": "green", "Slightly Off": "orange", "Needs Adjustment": "red"}.get(verdict, "grey")
                        v_icon  = {"Accurate": "check_circle", "Slightly Off": "schedule", "Needs Adjustment": "error"}.get(verdict, "info")
                        st.markdown(f"### {icon(v_icon, color=v_color)} {verdict}", unsafe_allow_html=True)
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Detected Time", d["detected_time"])
                        m2.metric("Real Time Now",  acc["reference_time"])
                        m3.metric("Drift",          acc["drift_class"])
                        drift_label = f"{'+' if acc['offset_minutes']>0 else ''}{acc['offset_minutes']} min"
                        m4.metric("Offset",          drift_label)
                        st.markdown("---")
                        st.info(acc["suggestion"])
                        ampm = d.get("ampm", {})
                        if ampm:
                            st.markdown(f"**Period Inference:** {ampm.get('period','?')} ({ampm.get('confidence',0):.0f}% confidence) — {ampm.get('reason','')}")
                else:
                    st.error(f"Server error: {res.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")

elif st.session_state.page == "dashboard":
    st.markdown(f"## {icon('monitoring')} Analytics Dashboard", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 4])
    if col_a.button("Refresh Data"):
        st.rerun()
    if col_b.button("Clear Database"):
        requests.post(f"{API_URL}/metrics/clear")
        st.rerun()
    try:
        metrics = requests.get(f"{API_URL}/metrics").json()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Scans",   metrics["total_analyses"])
        k2.metric("Success Rate",  f"{metrics['success_rate']:.1f}%")
        k3.metric("Avg Latency",   f"{metrics['avg_processing_time']:.3f}s")
        k4.metric("Failures",      metrics["failure_count"])
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### {icon('alt_route')} Logic Path Distribution", unsafe_allow_html=True)
            df_method = pd.DataFrame(list(metrics["method_usage"].items()), columns=["Method", "Count"])
            if not df_method.empty:
                st.plotly_chart(px.bar(df_method, x="Method", y="Count", color="Method"), use_container_width=True)
            else:
                st.info("No data yet.")
        with c2:
            st.markdown(f"#### {icon('memory')} Component Utilization", unsafe_allow_html=True)
            df_comp = pd.DataFrame(list(metrics["component_usage"].items()), columns=["Component", "Count"])
            if not df_comp.empty:
                st.plotly_chart(px.pie(df_comp, names="Component", values="Count", hole=0.4), use_container_width=True)
            else:
                st.info("No data yet.")
        st.markdown("---")
        st.markdown(f"#### {icon('history')} Reading History", unsafe_allow_html=True)
        hcol1, hcol2, hcol3 = st.columns(3)
        with hcol1:
            hist_limit = st.number_input("Max rows", value=50, min_value=5, max_value=500, step=5)
        with hcol2:
            hist_conf = st.selectbox("Confidence filter", ["All", "High", "Refined", "Low"])
        with hcol3:
            hist_hours = st.number_input("Last N hours (0 = all)", value=0.0, min_value=0.0, step=1.0)
        hist_params = {"limit": int(hist_limit)}
        if hist_conf != "All":            hist_params["confidence"]  = hist_conf
        if hist_hours and hist_hours > 0: hist_params["since_hours"] = hist_hours
        hist_res = requests.get(f"{API_URL}/readings/history", params=hist_params)
        if hist_res.status_code == 200:
            hdata = hist_res.json()
            if hdata["readings"]:
                hdf = pd.DataFrame(hdata["readings"])
                st.caption(f"Showing {hdata['count']} reading(s)")
                st.dataframe(hdf, use_container_width=True, hide_index=True)
                csv_res = requests.get(f"{API_URL}/metrics/export")
                if csv_res.status_code == 200:
                    st.download_button(
                        label=f"{icon('download')} Export All as CSV",
                        data=csv_res.text,
                        file_name="clock_readings.csv",
                        mime="text/csv",
                    )
            else:
                st.info("No reading history matches the selected filters.")
    except Exception as e:
        st.error(f"Dashboard Error: {e}")


# ============================================================
# GAUGE PAGES  (new)
# ============================================================

# ── Gauge helper: sidebar config ─────────────────────────────────────────────
def _gauge_type_selector(key_prefix: str):
    """
    Renders a gauge-type selector widget.
    Returns (gauge_type_override_or_none, ocr_text).
    """
    try:
        types_res = requests.get(f"{API_URL}/gauge/types", timeout=2)
        gauge_types = ["— auto-detect —"] + types_res.json().get("gauge_types", []) if types_res.ok else ["— auto-detect —"]
    except Exception:
        gauge_types = ["— auto-detect —"]

    col_gt, col_ocr = st.columns([2, 2])
    with col_gt:
        selected = st.selectbox("Gauge type", gauge_types, key=f"{key_prefix}_gt",
                                help="'auto-detect' uses C1 metadata + OCR text to determine the gauge type.")
    with col_ocr:
        ocr_text = st.text_input("OCR text from gauge face (optional)", key=f"{key_prefix}_ocr",
                                 placeholder="e.g. 0  50  100 PSI")

    override = None if selected == "— auto-detect —" else selected
    return override, ocr_text


# ── PAGE: Gauge Image Analysis ────────────────────────────────────────────────
if st.session_state.page == "gauge_image":
    st.markdown(f"## {icon('image_search')} Gauge Image Analysis", unsafe_allow_html=True)
    st.info(
        "Upload a gauge image. **C1** detects and crops the gauge face, "
        "**C2** finds the needle skeleton, **C3** measures the needle angle, "
        "and **C4** (this component) converts that angle into a real-world value."
    )
    st.markdown("---")

    col_up, col_cfg = st.columns([2, 2])
    with col_up:
        gauge_file = st.file_uploader("Gauge image", type=["jpg","png","jpeg"], key="gauge_img_file")
        if gauge_file:
            st.image(gauge_file, caption="Uploaded gauge image", use_container_width=True)

    with col_cfg:
        st.markdown(f"#### {icon('tune')} Configuration", unsafe_allow_html=True)
        gauge_type_override, ocr_text = _gauge_type_selector("gi")

        st.markdown("---")
        st.markdown("**Simulate C3 output** *(for testing without running the full pipeline)*")
        test_angle = st.slider("Needle angle (°)", min_value=-180, max_value=180, value=45, step=1, key="gi_angle",
                               help="In production this comes from C3 automatically.")
        test_conf  = st.slider("C3 confidence",    min_value=0.0,  max_value=1.0,  value=0.9, step=0.01, key="gi_conf")

        c1_class = st.text_input("C1 detected class (optional)", placeholder="e.g. pressure_gauge", key="gi_c1class")

    if gauge_file and st.button("Analyze Gauge", type="primary"):
        with st.spinner("Running C4 gauge logic..."):
            try:
                payload = {
                    "c3_output":  {"angle": float(test_angle), "confidence": float(test_conf)},
                    "c1_metadata": {"class": c1_class} if c1_class else None,
                    "gauge_type_override": gauge_type_override,
                    "ocr_text": ocr_text or None,
                }
                res = requests.post(f"{API_URL}/gauge/analyze", json=payload)
                if res.status_code == 200:
                    reading = res.json()
                    display_gauge_reading(reading)

                    st.markdown("---")
                    with st.expander("Raw API response"):
                        st.json(reading)
                else:
                    st.error(f"API error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
    elif not gauge_file:
        st.warning("Please upload a gauge image.")


# ── PAGE: Gauge Live Stream ───────────────────────────────────────────────────
elif st.session_state.page == "gauge_stream":
    st.markdown(f"## {icon('sensors')} Gauge Live Stream Monitor", unsafe_allow_html=True)
    st.info(
        "Simulates a live CCTV stream feeding into C4. "
        "Each 'frame' sends the current needle angle to /gauge/stream-frame with temporal smoothing."
    )
    st.markdown("---")

    col_cfg, col_live = st.columns([1, 2])

    with col_cfg:
        st.markdown(f"#### {icon('tune')} Stream Configuration", unsafe_allow_html=True)
        stream_id       = st.text_input("Stream / camera ID", value="cctv_cam1", key="gs_stream_id")
        gauge_type_override, ocr_text = _gauge_type_selector("gs")
        smooth          = st.checkbox("Enable temporal smoothing", value=True, key="gs_smooth")
        sim_angle       = st.slider("Live needle angle (°)", -180, 180, 0, 1, key="gs_angle")
        sim_conf        = st.slider("C3 confidence", 0.0, 1.0, 0.9, 0.01, key="gs_conf")

        col_r, col_rst = st.columns(2)
        send_frame      = col_r.button("Send Frame", type="primary")
        reset_stream    = col_rst.button("Reset Stream", type="secondary")

        if reset_stream:
            try:
                r = requests.post(f"{API_URL}/gauge/reset-stream", json={"stream_id": stream_id})
                if r.ok:
                    st.success(f"Stream '{stream_id}' history cleared.")
                    if "stream_history" in st.session_state:
                        st.session_state.stream_history = []
            except Exception as e:
                st.error(f"Reset failed: {e}")

    with col_live:
        st.markdown(f"#### {icon('show_chart')} Live Reading", unsafe_allow_html=True)

        if "stream_history" not in st.session_state:
            st.session_state.stream_history = []
        if "stream_frame_id" not in st.session_state:
            st.session_state.stream_frame_id = 0

        if send_frame:
            try:
                st.session_state.stream_frame_id += 1
                payload = {
                    "c3_output":           {"angle": float(sim_angle), "confidence": float(sim_conf)},
                    "gauge_type_override": gauge_type_override,
                    "ocr_text":            ocr_text or None,
                    "frame_id":            st.session_state.stream_frame_id,
                    "stream_id":           stream_id,
                    "smooth":              smooth,
                }
                res = requests.post(f"{API_URL}/gauge/stream-frame", json=payload)
                if res.status_code == 200:
                    reading = res.json()
                    display_gauge_reading(reading, show_title=False)
                    # Append to history for chart
                    st.session_state.stream_history.append({
                        "frame":  st.session_state.stream_frame_id,
                        "angle":  sim_angle,
                        "value":  reading["value"],
                        "status": reading["status"],
                        "unit":   reading["unit"],
                    })
                else:
                    st.error(f"API error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

        # ── Live chart ────────────────────────────────────────────────────────
        if st.session_state.stream_history:
            st.markdown("---")
            st.markdown(f"#### {icon('timeline')} Reading History (this session)", unsafe_allow_html=True)
            hdf = pd.DataFrame(st.session_state.stream_history)
            unit_label = hdf["unit"].iloc[-1] if not hdf.empty else ""

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hdf["frame"], y=hdf["value"],
                mode="lines+markers",
                name=f"Reading ({unit_label})",
                line=dict(color="#38bdf8", width=2),
                marker=dict(size=6),
            ))
            fig.update_layout(
                xaxis_title="Frame", yaxis_title=unit_label,
                height=260, margin=dict(l=10, r=10, t=20, b=40),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.8)",
                font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(hdf.tail(10).iloc[::-1], use_container_width=True, hide_index=True)


# ── PAGE: Multi-Gauge Scene ───────────────────────────────────────────────────
elif st.session_state.page == "gauge_multi":
    st.markdown(f"## {icon('dashboard')} Multi-Gauge Scene Analysis", unsafe_allow_html=True)
    st.info(
        "Analyse multiple gauges detected in the **same image or CCTV frame**. "
        "Add one row per gauge detected by C1/C2/C3."
    )
    st.markdown("---")

    # Fetch gauge types for dropdowns
    try:
        types_res   = requests.get(f"{API_URL}/gauge/types", timeout=2)
        gauge_types = types_res.json().get("gauge_types", []) if types_res.ok else []
    except Exception:
        gauge_types = []

    # ── Dynamic gauge input rows ──────────────────────────────────────────────
    if "gauge_rows" not in st.session_state:
        st.session_state.gauge_rows = [{"angle": 45.0, "conf": 0.9, "type": "pressure_psi"}]

    def add_gauge_row():
        st.session_state.gauge_rows.append({"angle": 0.0, "conf": 0.9, "type": gauge_types[0] if gauge_types else ""})

    def remove_gauge_row(idx):
        if len(st.session_state.gauge_rows) > 1:
            st.session_state.gauge_rows.pop(idx)

    st.markdown(f"#### {icon('list')} Gauges in Scene", unsafe_allow_html=True)

    for i, row in enumerate(st.session_state.gauge_rows):
        with st.container():
            c_n, c_a, c_c, c_t, c_del = st.columns([0.4, 1.2, 1.2, 2, 0.6])
            c_n.markdown(f"**#{i+1}**")
            row["angle"] = c_a.number_input("Angle (°)", value=row["angle"], step=1.0, key=f"mg_angle_{i}")
            row["conf"]  = c_c.number_input("Conf",      value=row["conf"],  step=0.01, min_value=0.0, max_value=1.0, key=f"mg_conf_{i}")
            type_opts    = gauge_types if gauge_types else ["generic_0_100"]
            default_idx  = type_opts.index(row["type"]) if row["type"] in type_opts else 0
            row["type"]  = c_t.selectbox("Gauge type", type_opts, index=default_idx, key=f"mg_type_{i}")
            if c_del.button("✕", key=f"mg_del_{i}"):
                remove_gauge_row(i)
                st.rerun()

    st.button("+ Add Gauge", on_click=add_gauge_row)
    source = st.radio("Source", ["image", "cctv_stream"], horizontal=True, key="mg_source")

    st.markdown("---")
    if st.button("Analyze All Gauges", type="primary"):
        with st.spinner("Processing all gauges..."):
            try:
                gauge_list = [
                    {
                        "c3_output":           {"angle": row["angle"], "confidence": row["conf"]},
                        "gauge_type_override": row["type"],
                    }
                    for row in st.session_state.gauge_rows
                ]
                payload = {"gauge_list": gauge_list, "source": source}
                res = requests.post(f"{API_URL}/gauge/analyze-multiple", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    readings = data.get("readings", [])
                    st.markdown(f"#### {icon('check_circle', color='green')} {len(readings)} Gauge(s) Processed", unsafe_allow_html=True)
                    cols = st.columns(min(len(readings), 3))
                    for i, reading in enumerate(readings):
                        with cols[i % 3]:
                            st.markdown(f"**Gauge #{i+1}**")
                            display_gauge_reading(reading, show_title=False)
                            st.markdown("---")
                    # Summary table
                    st.markdown(f"#### {icon('table_chart')} Summary", unsafe_allow_html=True)
                    summary_df = pd.DataFrame([{
                        "Gauge":      f"#{i+1}",
                        "Type":       r["gauge_type"],
                        "Reading":    r["display"],
                        "Status":     r["status"],
                        "Confidence": r["confidence"],
                        "% of Scale": r["percentage"],
                    } for i, r in enumerate(readings)])
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                else:
                    st.error(f"API error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")


# ── PAGE: Register Custom Gauge ───────────────────────────────────────────────
elif st.session_state.page == "gauge_register":
    st.markdown(f"## {icon('add_circle')} Register Custom Gauge Type", unsafe_allow_html=True)
    st.info(
        "Register a new gauge type at runtime. Once registered, use its key in any gauge analysis page. "
        "The registration persists until the server restarts."
    )
    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(f"#### {icon('settings')} Gauge Scale Definition", unsafe_allow_html=True)
        reg_key   = st.text_input("Gauge type key (unique, no spaces)", placeholder="e.g. my_boiler_pressure", key="reg_key")
        reg_unit  = st.text_input("Unit",        placeholder="e.g. bar, PSI, °C, RPM",  key="reg_unit")
        reg_desc  = st.text_input("Description", placeholder="Optional label",           key="reg_desc")

        c1, c2 = st.columns(2)
        reg_min  = c1.number_input("Min value",    value=0.0,    key="reg_min")
        reg_max  = c2.number_input("Max value",    value=100.0,  key="reg_max")
        reg_sa   = c1.number_input("Start angle (°, at min value)", value=-135.0, key="reg_sa",
                                   help="Angle where needle = min_value. Negative = CCW from 12 o'clock.")
        reg_ea   = c2.number_input("End angle (°, at max value)",   value= 135.0, key="reg_ea",
                                   help="Angle where needle = max_value.")
        reg_dp   = st.number_input("Decimal places", value=1, min_value=0, max_value=4, key="reg_dp")

    with col_r:
        st.markdown(f"#### {icon('warning')} Warning / Danger Thresholds *(optional)*", unsafe_allow_html=True)
        tw1, tw2 = st.columns(2)
        reg_wl = tw1.number_input("Warning LOW",  value=0.0,  key="reg_wl")
        reg_wh = tw2.number_input("Warning HIGH", value=0.0,  key="reg_wh")
        td1, td2 = st.columns(2)
        reg_dl = td1.number_input("Danger LOW",   value=0.0,  key="reg_dl")
        reg_dh = td2.number_input("Danger HIGH",  value=0.0,  key="reg_dh")

        use_wl = st.checkbox("Use Warning LOW",  key="use_wl")
        use_wh = st.checkbox("Use Warning HIGH", key="use_wh")
        use_dl = st.checkbox("Use Danger LOW",   key="use_dl")
        use_dh = st.checkbox("Use Danger HIGH",  key="use_dh")

        st.markdown("---")
        st.markdown(f"#### {icon('preview')} Scale Preview", unsafe_allow_html=True)
        if reg_key and reg_unit:
            st.markdown(
                f"Key: `{reg_key}` · Unit: `{reg_unit}` · Range: `{reg_min}` → `{reg_max}` · "
                f"Sweep: `{reg_sa}°` → `{reg_ea}°`"
            )

    st.markdown("---")
    if st.button("Register Gauge Type", type="primary"):
        if not reg_key.strip():
            st.error("Gauge type key is required.")
        elif reg_max <= reg_min:
            st.error("Max value must be greater than min value.")
        elif not reg_unit.strip():
            st.error("Unit is required.")
        else:
            try:
                payload = {
                    "gauge_type":    reg_key.strip().replace(" ", "_"),
                    "start_angle":   reg_sa,
                    "end_angle":     reg_ea,
                    "min_value":     reg_min,
                    "max_value":     reg_max,
                    "unit":          reg_unit.strip(),
                    "description":   reg_desc.strip(),
                    "decimal_places": int(reg_dp),
                    "warning_low":   reg_wl if use_wl else None,
                    "warning_high":  reg_wh if use_wh else None,
                    "danger_low":    reg_dl if use_dl else None,
                    "danger_high":   reg_dh if use_dh else None,
                }
                res = requests.post(f"{API_URL}/gauge/register", json=payload)
                if res.status_code == 200:
                    d = res.json()
                    st.success(f"Gauge type **{d['gauge_type']}** registered successfully!")
                    st.markdown("You can now select it from the **Gauge type** dropdown on the analysis pages.")
                else:
                    st.error(f"Registration failed ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    # ── Existing registered types ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### {icon('list')} All Registered Gauge Types", unsafe_allow_html=True)
    try:
        types_res = requests.get(f"{API_URL}/gauge/types", timeout=2)
        if types_res.ok:
            types_list = types_res.json().get("gauge_types", [])
            st.markdown(
                "  ".join([f"`{t}`" for t in types_list]) or "*(none)*",
                unsafe_allow_html=True,
            )
    except Exception:
        st.warning("Could not reach the API to list gauge types.")