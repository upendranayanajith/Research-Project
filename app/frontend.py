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
        ("C2 Structure", "timeline", ["C1", "C2", "C4"]),
        ("C3 Expert AI", "model_training", ["Expert"]),
        ("C4 Physics", "functions", ["C1", "C2", "C4"])
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
    tab1, tab2, tab3, tab4 = st.tabs(["Localization", "Structure", "Expert AI", "Result"])
    
    with tab1:
        st.markdown(f"{icon('crop_free')} **YOLO Localization**", unsafe_allow_html=True)
        if "c1_detection" in viz: st.image(base64.b64decode(viz["c1_detection"]), width=300)
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
        else: st.info("Fast Path Used - Expert AI skipped.")
    with tab4:
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
        st.caption("Time reading enhanced by C2 Bayesian 3D reconstruction, multi-scale LVM analysis, and persistent homology tracking.")

# ==========================================
# CUSTOM NAVIGATION LOGIC
# ==========================================
if "page" not in st.session_state:
    st.session_state.page = "analysis"

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
nav_button("analysis", "File Analysis", "cloud_upload")
nav_button("webcam", "Live Webcam", "videocam")
nav_button("batch", "Batch Processing", "perm_media")
nav_button("c2_research", "C2 Research", "science")
nav_button("dashboard", "Analytics", "monitoring")

st.sidebar.markdown("---")

# ==========================================
# PAGE ROUTING
# ==========================================

# --- PAGE 1: UPLOAD ---
if st.session_state.page == "analysis":
    st.markdown(f"## {icon('cloud_upload')} File Analysis", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    st.markdown(f"#### {icon('settings')} Configuration", unsafe_allow_html=True)
    force_expert = st.checkbox("Force Expert Path (Activate C3 + XAI)", value=False)

    if uploaded_file and st.button("Run Analysis", type="primary"):
        with st.spinner("Processing..."):
            try:
                image = Image.open(uploaded_file)
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                files = {"file": ("image.jpg", img_byte_arr.getvalue(), "image/jpeg")}
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

# --- PAGE 4: DEDICATED C2 RESEARCH PAGE ---
elif st.session_state.page == "c2_research":
    st.markdown(f"## {icon('science')} C2 — Skeleton Structure Research", unsafe_allow_html=True)
    st.markdown("""
    **Component Owner:** C2 — Hand Skeleton Extraction & Research  
    **Role in Pipeline:** C1 (Localize) → **C2 (Skeleton + Research)** → C3 (Refine) → C4 (Time)
    """)
    st.markdown("---")

    # ── Architecture ──
    st.markdown(f"### {icon('account_tree')} How C2 Research Improves Time Reading", unsafe_allow_html=True)
    st.markdown("""
    | Research Module | Impact on Final Result |
    |---|---|
    | **Bayesian 3D Reconstruction** (GAP 1) | Confidence score → ±uncertainty in Result |
    | **Persistent Homology** (GAP 2) | Topology tracking → occlusion detection |
    | **Multi-Scale LVM Oracle** (GAP 3) | Optimal scale σ* → better keypoint detection |
    | **Riemannian Manifold** (GAP 4) | Geodesic distances → curved surface correction |
    | **LVM Temporal Smoothing** (GAP 5) | Frame gating → rejects noisy detections |
    """)
    st.markdown("---")

    # ── Live Demo ──
    st.markdown(f"### {icon('play_circle')} Live C2 Demo", unsafe_allow_html=True)
    C2_URL = "http://localhost:8002"
    uploaded_c2 = st.file_uploader("Upload Clock Image", type=["jpg", "png", "jpeg"], key="c2_demo")

    if uploaded_c2:
        if st.button("Run C2 Enhanced Analysis", type="primary", key="btn_c2_run"):
            with st.spinner("Running all 5 research algorithms..."):
                try:
                    files = {"file": ("img.jpg", uploaded_c2.getvalue(), "image/jpeg")}
                    r = requests.post(f"{C2_URL}/extract-skeleton-enhanced", files=files, timeout=60)
                    if r.status_code == 200:
                        d = r.json()
                        enh = d.get("enhanced", {})
                        rv = d.get("research_visuals", {})

                        # KPI strip
                        if rv.get("impact_kpis"):
                            st.image(base64.b64decode(rv["impact_kpis"]), caption="Research KPIs", use_container_width=True)

                        # Metrics row
                        r3d = enh.get("reconstruction_3d", {})
                        sa = enh.get("scale_analysis", {})
                        mf = enh.get("manifold", {})
                        tmp = enh.get("temporal", {})
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Confidence", f"{r3d.get('confidence', 0):.3f}")
                        col2.metric("Best Scale σ*", sa.get("best_sigma", "—"))
                        col3.metric("Surface", mf.get("surface_classification", "—"))
                        col4.metric("β₀", tmp.get("beta0", "—"))
                        st.markdown("---")

                        # Visual tabs
                        vt1, vt2, vt3, vt4 = st.tabs(["🔭 Scale Pyramid", "🧊 Confidence", "🌍 Curvature", "📊 Before vs After"])
                        with vt1:
                            if rv.get("scale_pyramid"):
                                st.image(base64.b64decode(rv["scale_pyramid"]), caption="Multi-Scale LVM — GAP 3", use_container_width=True)
                            st.markdown(f"**σ\\* = {sa.get('best_sigma')}** — {sa.get('interpretation', '')}")
                        with vt2:
                            if rv.get("confidence_gauge"):
                                st.image(base64.b64decode(rv["confidence_gauge"]), caption="Bayesian Confidence — GAP 1", width=350)
                            ha = r3d.get("hand_assignment", {})
                            if ha:
                                st.markdown(f"Hour = `{ha.get('hour')}`, Minute = `{ha.get('minute')}`")
                        with vt3:
                            if rv.get("curvature_heatmap"):
                                st.image(base64.b64decode(rv["curvature_heatmap"]), caption="Riemannian Manifold — GAP 4", use_container_width=True)
                            if mf.get("recommendation"):
                                st.info(mf["recommendation"])
                        with vt4:
                            if rv.get("comparison"):
                                st.image(base64.b64decode(rv["comparison"]), caption="Basic YOLO vs Enhanced", use_container_width=True)
                            if rv.get("betti_badge"):
                                st.image(base64.b64decode(rv["betti_badge"]), caption="Topology — GAP 2", width=300)
                    else:
                        st.error(f"C2 error: {r.status_code}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
    else:
        st.info("Upload a clock image to see all research outputs.")

    st.markdown("---")
    st.markdown(f"### {icon('checklist')} Research Contributions", unsafe_allow_html=True)
    st.markdown("""
    - ✅ **GAP 1 — Probabilistic 3D:** Bayesian P(G|I) with K=10 hypotheses → confidence & hand assignment
    - ✅ **GAP 2 — Persistent Homology:** Betti numbers β₀, β₁ → occlusion detection
    - ✅ **GAP 3 — Multi-Scale LVM:** Scale-space + HOG matching → optimal σ*
    - ✅ **GAP 4 — Riemannian Manifold:** Metric tensor + Dijkstra → curvature-aware distances
    - ✅ **GAP 5 — LVM Temporal:** Embedding gating → ACCEPT/INTERPOLATE/REJECT frames

    **All integrated into the main pipeline** — confidence, uncertainty, and occlusion risk appear in the Result tab.
    """)







# --- PAGE 5: DASHBOARD ---
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
    except Exception as e: st.error(f"Dashboard Error: {e}")