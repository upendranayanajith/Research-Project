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
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Configuration
API_URL = "http://localhost:8000"
st.set_page_config(page_title="Chronos Vision", layout="wide", page_icon="static/favicon.ico")

# --- GOOGLE MATERIAL SYMBOLS SETUP ---
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">', unsafe_allow_html=True)

def icon(name, size=24, color="inherit", vertical_align="middle"):
    """Helper to generate Material Symbol HTML."""
    return f'<span class="material-symbols-outlined" style="font-size:{size}px; color:{color}; vertical-align:{vertical_align};">{name}</span>'

# ==========================================
# [C1 & C2] REAL-TIME PROCESSOR
# ==========================================
class ClockProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.force_expert = False 
        self.last_result = None
        
        from app.core.engine import ClockEngine
        self.engine = ClockEngine(parent_dir)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        now = time.time()
        if now - self.last_time > 1:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = now

        if self.frame_count % 5 == 0:
            try:
                self.last_result = self.engine.analyze(img, force_expert=self.force_expert)
            except Exception as e:
                print(f"AI Error: {e}")
        
        if self.last_result:
            res = self.last_result
            cv2.putText(img, f"TIME: {res.get('time', '--:--')}", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
            method = res.get('method', 'Unknown')
            color = (0, 255, 0) if "Fast" in method else (0, 0, 255)
            cv2.putText(img, f"Mode: {method}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if "angles" in res:
                a1 = res["angles"]["hand1"]
                a2 = res["angles"]["hand2"]
                cv2.putText(img, f"H:{a1:.0f} M:{a2:.0f}", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(img, f"FPS: {self.fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# [Shared] HELPER FUNCTIONS
# ==========================================
def display_results(data):
    res = data["result"]
    viz = data.get("visualizations", {})
    ampm_data    = data.get("ampm", {})
    amb_data     = data.get("ambiguity", {})
    report_data  = data.get("report", {})

    if "error" in res:
        st.markdown(f"#### {icon('error', color='red')} Analysis Failed", unsafe_allow_html=True)
        st.error(res['error'])
        return

    st.markdown(f"#### {icon('check_circle', color='green')} Analysis Complete ({data['processing_time']:.3f}s)", unsafe_allow_html=True)

    is_fast = "Fast Path" in res["method"]
    method_icon = "bolt" if is_fast else "psychology"
    method_color = "green" if is_fast else "orange"

    st.markdown(f"**Method Used:** <span style='color:{method_color}'>{icon(method_icon, size=20)} {res['method']}</span>", unsafe_allow_html=True)

    stages = [
        ("C1 Localization", "crop_free", ["C1", "C2", "C4"]),
        ("C2 Structure",    "timeline",    ["C1", "C2", "C4"]),
        ("C3 Expert AI",    "model_training", ["Expert"]),
        ("C4 Physics",      "functions",   ["C1", "C2", "C4"])
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

        # ── Pull C1 quality block from response ──────────────────────────
        c1q = data.get("c1_quality") or {}
        det_conf   = c1q.get("confidence")
        quality    = c1q.get("quality") or {}
        hough      = c1q.get("hough_validation") or {}

        # ── Row 1: Detection image + confidence + Hough validation ────────
        col_img, col_stats = st.columns([1, 1])

        with col_img:
            if "c1_detection" in viz:
                st.image(base64.b64decode(viz["c1_detection"]), use_container_width=True)

        with col_stats:
            # --- Detection Confidence ---
            if det_conf is not None:
                pct = int(det_conf * 100)
                if pct >= 80:
                    color, tier = "#3782eb", "HIGH" 
                elif pct >= 60:
                    color, tier = "#39F91FE9", "MEDIUM" 
                else:
                    color, tier = "#ef4444", "LOW"
                st.markdown(
                    f"""Detection Confidence :
                    <span style='font-size:1rem; font-weight:700'>{pct}%</span> <br><br>
                    <span style='font-weight:400'> {tier}</span>
                    <div style='
                    background:#e0e0e0; 
                    border-radius:6px; 
                    height:8px; 
                    margin:4px 0 14px
                    '>
                      <div style='
                      background:{color}; 
                      width:{pct}%; 
                      height:8px; 
                      border-radius:6px
                      '></div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                # st.markdown(
                #     f"""<div style='background:#e5e7eb; border-radius:6px; height:8px; margin:4px 0 14px'>
                #       <div style='background:{color}; width:{pct}%; height:8px; border-radius:6px'></div>
                #     </div>""",
                #     unsafe_allow_html=True,
                # )

            # --- Hough Circle Validation (temporarily disabled) ---
            # h_ok = hough.get("validated", False)
            # h_lbl = "PASS" if h_ok else "FAIL"
            # st.markdown(
            #     f"Hough Circle Validation : **{h_lbl}**",
            #     unsafe_allow_html=True,
            # )
            # if h_ok:
            #     bc = hough.get("best_circle", {})
            #     c_score  = hough.get("center_score", 0)
            #     cov      = hough.get("coverage_score", 0)
            #     n_circ   = hough.get("circle_count", 0)
            #     st.caption(
            #         f"Circles found: **{n_circ}** | "
            #         f"Radius: **{bc.get('radius', '?')}px** | "
            #         f"Centre align: **{c_score:.0f}%** | "
            #         f"Coverage: **{cov:.0f}%**"
            #     )
            # else:
            #     st.caption("No circular face detected — image may be obscured or angled.")

        # ── Row 2: Hough overlay image (temporarily disabled) ─────────────
        # if "c1_hough" in viz and viz["c1_hough"]:
        #     st.markdown(f"Hough Circle Overlay", unsafe_allow_html=True)
        #     st.image(base64.b64decode(viz["c1_hough"]), caption="Green = detected circle, Red = centre", width=280)

        st.markdown("---")

        # ── Row 3: Image Quality Analysis ─────────────────────────────────
        if quality:
            overall = quality.get("overall_quality", 0)
            if overall >= 75:
                q_color = "#3782eb"
            elif overall >= 45:
                q_color = "#39F91FE9"
            else:
                q_color = "#ef4444"

            st.markdown(
                f"""Image Quality Score : 
                 <span style='font-weight:500; '><b>{overall:.0f} / 100</b></span>""",
                unsafe_allow_html=True,
            )

            # Per-metric bars
            metrics_info = [
                ("Sharpness (Blur)",  quality.get("blur_score", 0),       quality.get("blur_raw", 0),       "Laplacian variance",    "blur_on"),
                ("Brightness",        quality.get("brightness_score", 0), quality.get("brightness_raw", 0), "Mean pixel value",      "wb_sunny"),
                ("Contrast",          quality.get("contrast_score", 0),   quality.get("contrast_raw", 0),   "Pixel std deviation",   "contrast"),
            ]
            cols_m = st.columns(3)
            for col_m, (label, score, raw, raw_label, icn) in zip(cols_m, metrics_info):
                if score >= 75:
                    bar_c = "#3782eb"
                elif score >= 45:
                    bar_c = "#39F91FE9"
                else:
                    bar_c = "#ef4444"
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
            # Fallback: show basic skeleton only
            if "c2_skeleton" in viz:
                st.image(base64.b64decode(viz["c2_skeleton"]), width=350)
            st.info("Enhanced C2 data not available.")
        else:
            # ── 6 Sub-Tabs ──────────────────────────────────────────────
            s1, s2, s3, s4, s5, s6 = st.tabs([
                "🦴 Skeleton", "🔭 Scale Analysis", "🧊 3D Reconstruction",
                "🌍 Manifold", "⏱ Temporal", "📊 Impact Summary"
            ])

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
                st.caption("Selects optimal Gaussian scale σ* where graph structure best matches image content.")
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
                st.caption("Probabilistic graphical model: P(G|I) = P(I|G) × P(G) / P(I)")
                if c2v.get("confidence_gauge"):
                    st.image(base64.b64decode(c2v["confidence_gauge"]), caption="Confidence Gauge + Occlusion Risk", width=350)
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
                st.caption("Riemannian metric → geodesic vs Euclidean distance analysis.")
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
                st.caption("Betti numbers measure topological features: β₀ = connected components, β₁ = loops.")
                if c2v.get("betti_badge"):
                    st.image(base64.b64decode(c2v["betti_badge"]), caption="Topology Status", width=300)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("β₀ (Components)", tmp.get("beta0", "—"))
                col_b.metric("β₁ (Loops)", tmp.get("beta1", "—"))
                col_c.metric("Status", tmp.get("topology_status", "—"))

            with s6:
                st.markdown("**Research Impact Summary**")
                if c2v.get("comparison"):
                    st.image(base64.b64decode(c2v["comparison"]), caption="Basic YOLO vs Enhanced Research Output", use_container_width=True)
                if c2v.get("impact_kpis"):
                    st.image(base64.b64decode(c2v["impact_kpis"]), caption="Key Performance Indicators", use_container_width=True)
                st.markdown("---")
                st.markdown("""
                **What C2 Research Adds:**
                - ✅ **3D Uncertainty Quantification** — confidence score + occlusion risk classification
                - ✅ **Multi-Scale LVM Oracle** — selects optimal detection scale σ* automatically
                - ✅ **Riemannian Manifold Analysis** — geodesic distance on curved clock surfaces
                - ✅ **Persistent Homology Tracking** — topological consistency across video frames
                - ✅ **LVM Temporal Smoothing** — rejects jittery detections using embedding distance
                """)

    with tab3:
        st.markdown(f"{icon('psychology')} **Angle Predictions**", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if "c3_angles" in viz: st.image(base64.b64decode(viz["c3_angles"]), caption="Angle Visual", width=300)
        with col_b:
            if "angles" in res:
                st.markdown(f"**H:** {res['angles']['hand1']:.1f}°")
                st.markdown(f"**M:** {res['angles']['hand2']:.1f}°")
        if "c3_crops" in viz and viz["c3_crops"]:
            st.markdown("---")
            st.markdown(f"**{icon('image')} ResNet Inputs**", unsafe_allow_html=True)
            c_cols = st.columns(len(viz["c3_crops"]))
            for idx, (col, crop) in enumerate(zip(c_cols, viz["c3_crops"])):
                col.image(base64.b64decode(crop), width=100)
            if data.get("heatmap_b64"):
                st.markdown(f"**{icon('opacity')} Attention Map (Grad-CAM)**", unsafe_allow_html=True)
                st.image(base64.b64decode(data["heatmap_b64"]), width=300)
        else: st.info("Fast Path Used — Expert AI skipped.")

    with tab4:
        # ── Primary reading ───────────────────────────────────────────
        st.markdown(f"# {icon('schedule')} {res['time']}", unsafe_allow_html=True)

        # ── C2 Research enrichments ──
        unc = res.get('uncertainty', '')
        c2_conf = res.get('c2_confidence', 0)
        c2_occ = res.get('c2_occlusion_risk', 'UNKNOWN')
        c2_ha = res.get('c2_hand_assignment', {})

        if unc:
            st.markdown(f"**Uncertainty:** `{res['time']} {unc}` (C2 Bayesian estimation)", unsafe_allow_html=True)

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("C2 Confidence", f"{c2_conf:.2f}" if c2_conf else "—")
        occ_icon = "✅" if c2_occ == "LOW" else ("⚠️" if c2_occ == "MEDIUM" else "🔴")
        col_r2.metric("Occlusion Risk", f"{occ_icon} {c2_occ}")
        col_r3.metric("Hour Hand", c2_ha.get('hour', '—'))
        col_r4.metric("Minute Hand", c2_ha.get('minute', '—'))

        if c2_occ == "HIGH":
            st.warning("⚠️ High occlusion risk — result may be less reliable. Hands may overlap.")

        st.markdown("---")
        st.markdown(f"**Reasoning:** `{res.get('reasoning', 'N/A')}`")
        st.markdown("---")

        col_l, col_r = st.columns(2)

        # AM/PM inference
        with col_l:
            st.markdown(f"#### {icon('wb_sunny')} AM/PM Inference", unsafe_allow_html=True)
            if ampm_data:
                period    = ampm_data.get("period", "Unknown")
                p_conf    = ampm_data.get("confidence", 0)
                p_reason  = ampm_data.get("reason", "")
                p_color   = "orange" if period == "Unknown" else "green"
                st.markdown(f"**Period:** <span style='color:{p_color};font-size:1.3em'>{period}</span>  ({p_conf:.0f}% confidence)", unsafe_allow_html=True)
                st.caption(p_reason)
            else:
                st.info("Not available.")

        # Ambiguity resolver
        with col_r:
            st.markdown(f"#### {icon('help_outline')} Ambiguity Analysis", unsafe_allow_html=True)
            if amb_data:
                is_amb = amb_data.get("is_ambiguous", False)
                amb_icon  = "warning" if is_amb else "check_circle"
                amb_color = "orange"  if is_amb else "green"
                st.markdown(f"{icon(amb_icon, color=amb_color)} {amb_data.get('ambiguity_reason', '')}", unsafe_allow_html=True)
                candidates = amb_data.get("top_candidates", [])
                if candidates:
                    cdf = pd.DataFrame([{
                        "Time": c["time"],
                        "Error (°)": c["angular_error"],
                        "Confidence %": c["confidence_pct"],
                        "Fit": c["fit_quality"],
                    } for c in candidates])
                    st.dataframe(cdf, use_container_width=True, hide_index=True)
            else:
                st.info("Not available.")

    with tab5:
        # ── Full narrative report ─────────────────────────────────────
        if report_data:
            st.markdown(f"## {icon('description')} {report_data.get('title','')}", unsafe_allow_html=True)
            st.caption(f"Generated at {report_data.get('generated_at','')} — {report_data.get('one_liner','')}")
            st.markdown("---")
            for section in report_data.get("sections", []):
                st.markdown(f"### {section['heading']}")
                st.markdown(section["body"])
                st.markdown("")
        else:
            st.info("Report not available for this analysis.")

# ==========================================
# CUSTOM NAVIGATION LOGIC
# ==========================================
if "page" not in st.session_state:
    st.session_state.page = "analysis"
if "img_rotation" not in st.session_state:
    st.session_state.img_rotation = 0

def nav_button(page_key, label, icon_name):
    """Creates a navigation button with an icon."""
    c1, c2 = st.sidebar.columns([1, 4])
    with c1:
        st.markdown(f"<div style='text-align: center; padding-top: 5px;'>{icon(icon_name)}</div>", unsafe_allow_html=True)
    with c2:
        # If selected, use 'primary' style (red), else 'secondary'
        btn_type = "primary" if st.session_state.page == page_key else "secondary"
        if st.button(label, key=f"nav_{page_key}", type=btn_type, use_container_width=True):
            st.session_state.page = page_key
            st.rerun()

# --- SIDEBAR UI ---
logo_path = os.path.join(current_dir, "..", "assets", "images", "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.markdown(f"### {icon('menu')} Navigation", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Render Navigation Buttons
nav_button("analysis",    "File Analysis",     "cloud_upload")
nav_button("webcam",      "Live Webcam",       "videocam")
nav_button("batch",       "Batch Processing",  "perm_media")
nav_button("comparator",  "Clock Comparator",  "compare_arrows")
nav_button("accuracy",    "Accuracy Checker",  "verified")
nav_button("dashboard",   "Analytics",         "monitoring")

st.sidebar.markdown("---")

# ==========================================
# PAGE ROUTING
# ==========================================

# --- PAGE 1: UPLOAD ---
if st.session_state.page == "analysis":
    st.markdown(f"## {icon('cloud_upload')} File Analysis", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    # --- Image Preview + Rotation Controls ---
    if uploaded_file:
        # Reset rotation when a new file is selected
        if st.session_state.get("_analysis_last_file") != uploaded_file.name:
            st.session_state.img_rotation = 0
            st.session_state._analysis_last_file = uploaded_file.name

        preview_img = Image.open(uploaded_file)
        if st.session_state.img_rotation != 0:
            preview_img = preview_img.rotate(-st.session_state.img_rotation, expand=True)

        col_prev, col_ctrl = st.columns([2, 1])
        with col_prev:
            st.image(preview_img, caption=f"{st.session_state.img_rotation}° rotation applied", width=400)
        with col_ctrl:
            st.markdown(f"#### {icon('rotate_right')} Rotate Image", unsafe_allow_html=True)
            st.caption("Use if the clock appears sideways or upside-down. Each click turns the image 90°.")
            r1, r2 = st.columns(2)
            if r1.button("↺ Left", use_container_width=True, key="rot_ccw", help="Rotate 90° to the left (counter-clockwise)"):
                st.session_state.img_rotation = (st.session_state.img_rotation - 90) % 360
                st.rerun()
            if r2.button("↻ Right", use_container_width=True, key="rot_cw", help="Rotate 90° to the right (clockwise)"):
                st.session_state.img_rotation = (st.session_state.img_rotation + 90) % 360
                st.rerun()
            if st.session_state.img_rotation != 0:
                if st.button("Reset to 0°", use_container_width=True, key="rot_reset"):
                    st.session_state.img_rotation = 0
                    st.rerun()
            st.markdown(
                f"<div style='text-align:center; font-size:1.1rem; font-weight:600; color:#3b82f6'>"
                f"{st.session_state.img_rotation}°</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(f"#### {icon('settings')} Configuration", unsafe_allow_html=True)
    force_expert = st.checkbox("Force Expert Path (Activate C3 + XAI)", value=False)

    if uploaded_file and st.button("Run Analysis", type="primary"):
        with st.spinner("Processing..."):
            try:
                rotation = st.session_state.get("img_rotation", 0)
                if rotation != 0:
                    image = Image.open(uploaded_file)
                    image = image.rotate(-rotation, expand=True)
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format="JPEG")
                    img_bytes = img_byte_arr.getvalue()
                else:
                    img_bytes = uploaded_file.getvalue()
                files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
                data_form = {"force_expert": str(force_expert)}
                response = requests.post(f"{API_URL}/analyze", files=files, data=data_form)
                if response.status_code == 200: display_results(response.json())
                else: st.error(f"Server Error: {response.status_code}")
            except Exception as e: st.error(f"Connection Failed: {e}")

# --- PAGE 2: WEBCAM ---
elif st.session_state.page == "webcam":
    st.markdown(f"## {icon('videocam')} Real-Time Analysis", unsafe_allow_html=True)
    st.info("Running C1 (Localization) + C2 (Pose) locally. C4 runs on every 5th frame.")
    
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    col1, col2 = st.columns([3, 1])
    with col1:
        ctx = webrtc_streamer(key="clock-ai", video_processor_factory=ClockProcessor, rtc_configuration=rtc_configuration, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
    with col2:
        st.markdown(f"### {icon('tune')} Controls", unsafe_allow_html=True)
        if ctx.video_processor:
            st.markdown(f"{icon('military_tech')} **Force Expert Mode**", unsafe_allow_html=True)
            ctx.video_processor.force_expert = st.checkbox("", value=False)
        st.markdown("---")
        if st.button("Reset Connection"): st.cache_resource.clear(); st.rerun()

# --- PAGE 3: BATCH ---
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
                else: st.error("Batch failed.")
            except Exception as e: st.error(f"Error: {e}")


# --- PAGE 4: CLOCK COMPARATOR ---
elif st.session_state.page == "comparator":
    st.markdown(f"## {icon('compare_arrows')} Clock Comparator — Elapsed Time", unsafe_allow_html=True)
    st.info(
        "Upload two clock images (a **Before** and an **After** shot). "
        "C4 will read both clocks and compute how much time has elapsed between them."
    )
    st.markdown("---")
    col_s, col_e = st.columns(2)
    with col_s:
        st.markdown(f"#### {icon('hourglass_top')} Before Clock", unsafe_allow_html=True)
        f_start = st.file_uploader("Start clock image", type=["jpg","png","jpeg"], key="cmp_start")
        sp = st.selectbox("AM/PM (optional)", ["— unknown —","AM","PM"], key="sp_sel")
    with col_e:
        st.markdown(f"#### {icon('hourglass_bottom')} After Clock", unsafe_allow_html=True)
        f_end   = st.file_uploader("End clock image",   type=["jpg","png","jpeg"], key="cmp_end")
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
                        st.markdown(f"**Processing time:** {d['processing_time']:.3f}s")

                        # All spans
                        st.markdown("---")
                        st.markdown(f"#### {icon('list')} All Possible Spans", unsafe_allow_html=True)
                        spans = d["elapsed"]["all_spans"]
                        sdf = pd.DataFrame([{
                            "From":     s["from"],
                            "To":       s["to"],
                            "Elapsed":  s["elapsed_display"],
                            "Minutes":  s["elapsed_minutes"],
                            "Direction":s["direction"],
                            "Plausibility": s["plausibility"],
                        } for s in spans])
                        st.dataframe(sdf, use_container_width=True, hide_index=True)
                        st.caption(d["elapsed"]["notes"])
                else:
                    st.error(f"Server error: {res.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")
    elif not (f_start and f_end):
        st.warning("Please upload both a Before and After clock image.")


# --- PAGE 5: ACCURACY CHECKER ---
elif st.session_state.page == "accuracy":
    st.markdown(f"## {icon('verified')} Clock Accuracy Checker", unsafe_allow_html=True)
    st.info(
        "Upload a clock photo. C4 reads the time, compares it to your **device's "
        "current real time**, and tells you if the clock is fast, slow, or accurate."
    )
    st.markdown("---")

    col_up, col_cfg = st.columns([2, 1])
    with col_up:
        acc_file = st.file_uploader("Clock image", type=["jpg","png","jpeg"], key="acc_file")
    with col_cfg:
        period_hint = st.selectbox("AM/PM hint", ["— unknown —","AM","PM"], key="acc_period")
        tz_offset   = st.number_input("UTC offset (hours)", value=0.0, step=0.5,
                                      min_value=-12.0, max_value=14.0, key="acc_tz",
                                      help="e.g. 5.5 for UTC+5:30 (Sri Lanka / India)")

    if acc_file and st.button("Check Accuracy", type="primary"):
        with st.spinner("Analysing clock..."):
            try:
                pv = None if period_hint == "— unknown —" else period_hint
                files_p = {"file": (acc_file.name, acc_file.getvalue(), "image/jpeg")}
                form_p  = {"tz_offset_hours": str(tz_offset)}
                if pv: form_p["period"] = pv

                res = requests.post(f"{API_URL}/analyze-with-accuracy", files=files_p, data=form_p)
                if res.status_code == 200:
                    d = res.json()
                    if "error" in d:
                        st.error(d["error"])
                    else:
                        acc = d["accuracy"]
                        verdict = acc["verdict"]
                        v_color = {"Accurate": "green", "Slightly Off": "orange", "Needs Adjustment": "red"}.get(verdict, "grey")
                        v_icon  = {"Accurate": "check_circle", "Slightly Off": "schedule", "Needs Adjustment": "error"}.get(verdict, "info")

                        st.markdown(f"### {icon(v_icon, color=v_color)} {verdict}", unsafe_allow_html=True)

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Detected Time",  d["detected_time"])
                        m2.metric("Real Time Now",  acc["reference_time"])
                        m3.metric("Drift",          acc["drift_class"])
                        drift_label = f"{'+' if acc['offset_minutes']>0 else ''}{acc['offset_minutes']} min"
                        m4.metric("Offset",         drift_label)

                        st.markdown("---")
                        st.info(acc["suggestion"])

                        # AM/PM
                        ampm = d.get("ampm", {})
                        if ampm:
                            st.markdown(f"**Period Inference:** {ampm.get('period','?')} ({ampm.get('confidence',0):.0f}% confidence) — {ampm.get('reason','')}")
                else:
                    st.error(f"Server error: {res.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")


# --- PAGE 6: DASHBOARD ---
elif st.session_state.page == "dashboard":
    st.markdown(f"## {icon('monitoring')} Analytics Dashboard", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 4])
    if col_a.button("Refresh Data"): st.rerun()
    if col_b.button("Clear Database"): 
        requests.post(f"{API_URL}/metrics/clear")
        st.rerun()

    try:
        metrics = requests.get(f"{API_URL}/metrics").json()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Scans", metrics["total_analyses"])
        k2.metric("Success Rate", f"{metrics['success_rate']:.1f}%")
        k3.metric("Avg Latency", f"{metrics['avg_processing_time']:.3f}s")
        k4.metric("Failures", metrics["failure_count"])
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### {icon('alt_route')} Logic Path Distribution", unsafe_allow_html=True)
            df_method = pd.DataFrame(list(metrics["method_usage"].items()), columns=["Method", "Count"])
            if not df_method.empty: st.plotly_chart(px.bar(df_method, x="Method", y="Count", color="Method"), use_container_width=True)
            else: st.info("No data yet.")
        with c2:
            st.markdown(f"#### {icon('memory')} Component Utilization", unsafe_allow_html=True)
            df_comp = pd.DataFrame(list(metrics["component_usage"].items()), columns=["Component", "Count"])
            if not df_comp.empty: st.plotly_chart(px.pie(df_comp, names="Component", values="Count", hole=0.4), use_container_width=True)
            else: st.info("No data yet.")

        # --- Reading History with Filters ---
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
        if hist_conf != "All":          hist_params["confidence"] = hist_conf
        if hist_hours and hist_hours > 0: hist_params["since_hours"] = hist_hours

        try:
            hist_res = requests.get(f"{API_URL}/readings/history", params=hist_params)
            if hist_res.status_code == 200:
                hdata = hist_res.json()
                if hdata["readings"]:
                    hdf = pd.DataFrame(hdata["readings"])
                    st.caption(f"Showing {hdata['count']} reading(s)")
                    st.dataframe(hdf, use_container_width=True, hide_index=True)
                    # Export CSV button
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
        except Exception as he:
            st.warning(f"Could not load history: {he}")

    except Exception as e: st.error(f"Dashboard Error: {e}")