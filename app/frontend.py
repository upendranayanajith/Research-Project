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
        st.markdown(f"{icon('timeline')} **Hand Keypoints**", unsafe_allow_html=True)
        if "c2_skeleton" in viz: st.image(base64.b64decode(viz["c2_skeleton"]), width=300)
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
        st.markdown(f"**Reasoning:** `{res.get('reasoning', 'N/A')}`")

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

# --- PAGE 4: DASHBOARD ---
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