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
logo_path = os.path.join(current_dir, "..", "assets", "images", "logo.png")
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Configuration
API_URL = "http://localhost:8000"
st.set_page_config(page_title="Clock AI Research", layout="wide", page_icon="🕰️")

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
            
            # Note: We display angles on Live Cam too for debugging, matching C3 logic
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
        st.error(f"Analysis Failed: {res['error']}")
        return

    st.success(f"✅ Analysis Complete in {data['processing_time']:.3f}s")
    
    method_color = "green" if "Fast Path" in res["method"] else "orange"
    st.markdown(f"**Method Used:** :{method_color}[{res['method']}]")
    
    stages = ["C1 Localization", "C2 Hand Pose", "C3 Refinement", "C4 Physics"]
    active_stages = ["C1", "C2", "C4"]
    if "Expert" in res["method"]: active_stages.append("C3")
    
    cols = st.columns(4)
    for c, stage in zip(cols, stages):
        code = stage.split()[0]
        if code in active_stages: c.success(f"✅ {stage}")
        else: c.markdown(f"⚪ {stage}")
    
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["C1: Localization", "C2: Structure", "C3: Expert AI", "C4: Result"])
    
    with tab1:
        if "c1_detection" in viz: 
            st.image(base64.b64decode(viz["c1_detection"]), caption="YOLO Localization", width=500)
    
    with tab2:
        # [UPDATED] Only shows structure (detection), NO ANGLE DATA here.
        if "c2_skeleton" in viz: 
            st.image(base64.b64decode(viz["c2_skeleton"]), caption="Hand Keypoints (Structure)", width=500)
    
    with tab3:
        # [UPDATED] Shows the image with Marked Angles + Metric Cards
        st.markdown("### Angle Predictions")
        if "c3_angles" in viz:
            st.image(base64.b64decode(viz["c3_angles"]), caption="Angle Estimation (Visual)", width=500)
        
        # Move Metric Cards to C3 Tab
        if "angles" in res:
            c1, c2 = st.columns(2)
            c1.metric("Hand 1 Angle", f"{res['angles']['hand1']:.1f}°")
            c2.metric("Hand 2 Angle", f"{res['angles']['hand2']:.1f}°")

        st.markdown("---")
        if "c3_crops" in viz and viz["c3_crops"]:
            st.markdown("**Hand Crops (ResNet Input)**")
            c_cols = st.columns(len(viz["c3_crops"]))
            for idx, (col, crop) in enumerate(zip(c_cols, viz["c3_crops"])):
                col.image(base64.b64decode(crop), caption=f"Hand {idx+1}")
            if data.get("heatmap_b64"):
                st.markdown("**XAI Attention Map (Grad-CAM)**")
                st.image(base64.b64decode(data["heatmap_b64"]), caption="Model Focus Area", width=500)
        else: 
            st.info("Fast Path Used - Deep Learning (C3) was skipped for efficiency.")
            
    with tab4:
        st.markdown(f"# Time: {res['time']}")
        st.code(res.get("reasoning", "No reasoning provided"))

# ==========================================
# [Shared] UI LAYOUT
# ==========================================
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)
else:
    st.sidebar.warning("Logo not found")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Select Page", ["Analysis (Upload)", "Live Webcam", "Batch Processing", "Performance Dashboard"])

if page == "Analysis (Upload)":
    st.title("File Analysis")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
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

elif page == "Live Webcam":
    st.title("Real-Time Analysis")
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    col1, col2 = st.columns([3, 1])
    with col1:
        ctx = webrtc_streamer(key="clock-ai", video_processor_factory=ClockProcessor, rtc_configuration=rtc_configuration, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
    with col2:
        if ctx.video_processor:
            ctx.video_processor.force_expert = st.checkbox("Force Expert Mode", value=False)
        if st.button("🔄 Reset Connection"): st.cache_resource.clear(); st.rerun()

elif page == "Batch Processing":
    st.title("Batch Processing")
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)
    if uploaded_files and st.button("Process All"):
        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
        with st.spinner("Processing..."):
            res = requests.post(f"{API_URL}/analyze_batch", files=files)
            if res.status_code == 200:
                data = res.json()
                st.success(f"Processed {data['total_images']} images!")
                st.dataframe(pd.DataFrame(data["results"]), use_container_width=True)

elif page == "Performance Dashboard":
    st.title("Analytics Dashboard (C4 Metrics)")
    if st.button("Refresh"): st.rerun()
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
            df_method = pd.DataFrame(list(metrics["method_usage"].items()), columns=["Method", "Count"])
            if not df_method.empty: st.plotly_chart(px.bar(df_method, x="Method", y="Count", color="Method"), use_container_width=True)
        with c2:
            df_comp = pd.DataFrame(list(metrics["component_usage"].items()), columns=["Component", "Count"])
            if not df_comp.empty: st.plotly_chart(px.pie(df_comp, names="Component", values="Count", hole=0.4), use_container_width=True)
    except Exception as e:
        st.error(f"Dashboard Error: {e}")