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
import threading

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Configuration
API_URL = "http://localhost:8000"
st.set_page_config(page_title="HARP Vision", layout="wide", page_icon="static/favicon.ico")

# --- MULTITHREADED CAMERA STREAMER ---
class CameraStream:
    """Reads frames from a camera in a background thread to prevent GUI lagging."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

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
        self.manual_min_val = ""
        self.manual_max_val = ""
        self.last_result = None
        
        from app.core.engine import HARPEngine
        self.engine = HARPEngine(parent_dir)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        now = time.time()
        
        # FPS Calculation
        if now - self.last_time > 1:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = now


        # Process every 5th frame to save CPU
        if self.frame_count % 5 == 0:
            try:
                self.last_result = self.engine.analyze(
                    img,
                    force_expert=self.force_expert,
                    manual_min_val=self.manual_min_val,
                    manual_max_val=self.manual_max_val,
                    enable_temporal=True,   # [Tier 1.4] Kalman smoothing in live mode
                )
            except Exception as e:
                print(f"AI Error: {e}")

        # Draw overlays
        if self.last_result:
            res = self.last_result
            
            # Show Time or Gauge %
            display_val = res.get('time', '--')
            cv2.putText(img, f"READING: {display_val}", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
            
            method = res.get('method', 'Unknown')
            color = (0, 255, 0) if "Fast" in method or "Gauge" in method else (0, 0, 255)
            cv2.putText(img, f"Mode: {method}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Only show angles if it's a clock
            if "angles" in res and res["angles"].get("hand1", 0) != 0.0:
                a1 = res["angles"].get("hand1", 0)
                a2 = res["angles"].get("hand2", 0)
                cv2.putText(img, f"H:{a1:.0f} M:{a2:.0f}", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(img, f"FPS: {self.fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # [Tier 1.4] Temporal stability indicator
        if self.last_result:
            tx = self.last_result.get("temporal_xai")
            if tx and tx.get("status") == "Active":
                trend = tx.get("trend", "")
                stab = tx.get("stability_score", 0)
                t_color = (0, 200, 0) if stab > 75 else (0, 200, 200) if stab > 40 else (0, 0, 255)
                cv2.putText(img, f"Kalman: {trend} ({stab:.0f}%)", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, t_color, 1)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# [Shared] HELPER FUNCTIONS
# ==========================================
def display_results(data):
    res = data["result"]
    viz = data.get("visualizations", {})
    
    if "error" in res and res["error"]:
        st.markdown(f"#### {icon('error', color='red')} Analysis Failed", unsafe_allow_html=True)
        st.error(res['error'])
        return

    st.markdown(f"#### {icon('check_circle', color='green')} Analysis Complete ({data['processing_time']:.3f}s)", unsafe_allow_html=True)
    
    is_fast = "Fast" in res["method"] or "Gauge" in res["method"]
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
        elif name.split()[0] in active_list and ("Fast" in res["method"] or "Gauge" in res["method"]) and "Expert" not in name:
            is_active = True
        color = "green" if is_active else "grey"
        col.markdown(f"{icon(icn, color=color)} {name}", unsafe_allow_html=True)
    
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["Localization", "Structure", "Expert AI", "Result"])
    
    with tab1:
        st.markdown(f"{icon('crop_free')} **YOLO Localization**", unsafe_allow_html=True)
        if "c1_detection" in viz: st.image(base64.b64decode(viz["c1_detection"]), width=300)
    with tab2:
        st.markdown(f"{icon('timeline')} **Keypoints**", unsafe_allow_html=True)
        col2a, col2b = st.columns(2)
        with col2a:
            if "c2_skeleton" in viz: st.image(base64.b64decode(viz["c2_skeleton"]), width=300)
        with col2b:
            if "scale" in res and res["scale"]:
                st.markdown("### OCR Scale Check")
                st.info(f"**Min Reading:** {res['scale'].get('min', 'Failed')}")
                st.error(f"**Max Reading:** {res['scale'].get('max', 'Failed')}")
    with tab3:
        st.markdown(f"{icon('psychology')} **Angle Predictions**", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if "c3_angles" in viz: st.image(base64.b64decode(viz["c3_angles"]), caption="Angle Visual", width=300)
        with col_b:
            if "angles" in res and res["angles"]:
                if "span" in res["angles"]:
                    st.markdown(f"**Total Scale Span:** {res['angles'].get('span', 0):.1f}°")
                    st.markdown(f"**Needle Pos:** {res['angles'].get('needle', 0):.1f}°")
                    upd = res['angles'].get('units_per_deg', 0.0)
                    if upd > 0:
                        st.markdown(f"**1° Angle =** {upd:.4f} scale units")
                else:
                    st.markdown(f"**H:** {res['angles'].get('hand1', 0):.1f}°")
                    st.markdown(f"**M:** {res['angles'].get('hand2', 0):.1f}°")
        if "c3_crops" in viz and viz["c3_crops"]:
            st.markdown("---")
            st.markdown(f"**{icon('image')} ResNet Inputs**", unsafe_allow_html=True)
            c_cols = st.columns(len(viz["c3_crops"]))
            for idx, (col, crop) in enumerate(zip(c_cols, viz["c3_crops"])):
                col.image(base64.b64decode(crop), width=100)
            if data.get("heatmap_b64"):
                st.markdown(f"**{icon('opacity')} Attention Map (GradCAM++)**", unsafe_allow_html=True)
                st.image(base64.b64decode(data["heatmap_b64"]), width=300)

            # --- AI Insight Explanations (Gemini or LocalExplainer) ---
            debug_lines = res.get("debug", [])
            insight_lines = [l for l in debug_lines if "AI Insight" in l]
            uncertainty_lines = [l for l in debug_lines if "uncertainty" in l.lower() or "alpha" in l.lower()]

            if insight_lines:
                st.markdown("---")
                st.markdown(f"**{icon('psychology')} AI Model Explanations**", unsafe_allow_html=True)
                for line in insight_lines:
                    # Strip the "AI Insight Hand X: " prefix for clean display
                    label, _, text = line.partition(": ")
                    hand_label = label.replace("AI Insight ", "")
                    is_local = "[Local XAI]" in text
                    is_gemini = "[Gemini]" in text
                    badge = "🔵 Local" if is_local else "✨ Gemini" if is_gemini else "ℹ️"
                    clean_text = text.replace("[Local XAI] ", "").replace("[Gemini] ", "")
                    st.info(f"**{badge} — {hand_label}:** {clean_text}")
            else:
                st.markdown("---")
                st.info("💡 AI Explanation: Enable **Force Expert Path** and re-run to generate model explanations.")

            # --- Uncertainty & Confidence ---
            if uncertainty_lines or res.get("uncertainty_deg"):
                st.markdown(f"**{icon('bar_chart')} C3 Uncertainty**", unsafe_allow_html=True)
                unc_val = res.get("uncertainty_deg", "N/A")
                if unc_val and unc_val != "N/A":
                    st.success(f"**MC Dropout Uncertainty:** {unc_val}")
                for line in uncertainty_lines:
                    st.caption(f"🔢 {line}")

            # --- Collapsible Debug Log ---
            with st.expander("🔍 Full Pipeline Debug Log", expanded=False):
                for line in debug_lines:
                    icon_char = "✅" if "Accepted" in line or "Gemini API" in line or "Manual" in line else \
                                "⚠️" if "Rejected" in line or "Failed" in line else \
                                "🔵" if "alpha" in line.lower() or "uncertainty" in line.lower() else "▪️"
                    st.markdown(f"{icon_char} `{line}`")

            # --- [Tier 1.4] Temporal Stability Panel ---
            temporal_xai = res.get("temporal_xai")
            if temporal_xai:
                st.markdown("---")
                st.markdown(
                    f"**{icon('bar_chart')} 📈 Temporal Stability (Kalman Filter)**",
                    unsafe_allow_html=True,
                )
                t_status = temporal_xai.get("status", "N/A")
                if t_status == "Initialising":
                    st.info(temporal_xai.get("message", "Kalman filter warming up..."))
                elif t_status == "Active":
                    t_cols = st.columns(4)
                    t_cols[0].metric('Stability', f"{temporal_xai.get('stability_score', 'N/A')}%")
                    t_cols[1].metric('Trend', temporal_xai.get('trend', 'N/A'))
                    t_cols[2].metric('Spikes Rejected', temporal_xai.get('total_spike_count', 0))
                    t_cols[3].metric('Avg Correction', f"{temporal_xai.get('mean_kalman_correction_deg', 0):.1f}°")
                    st.caption(f"🔢 {temporal_xai.get('message', '')}")
                    with st.expander("Variance Details"):
                        st.json({
                            "hand1_variance_deg": temporal_xai.get("hand1_variance_deg"),
                            "hand2_variance_deg": temporal_xai.get("hand2_variance_deg"),
                            "spike_rate_per_frame": temporal_xai.get("spike_rate_per_frame"),
                            "frames_seen": temporal_xai.get("frames_seen"),
                        })
            else:
                st.caption("📈 Temporal Stability: N/A (only active in Live Webcam / RTSP mode)")
        else:
            st.info("Expert AI skipped (Fast Path or Gauge Mode used). Enable 'Force Expert Path' to activate C3 + XAI.")

    with tab4:
        st.markdown(f"# {icon('schedule')} {res['time']}", unsafe_allow_html=True)
        st.markdown(f"**Reasoning:** `{res.get('reasoning', 'N/A')}`")
        if "ampm" in res:
            ampm_icon = "wb_sunny" if "AM" in res["ampm"] else "bedtime"
            st.markdown(f"**{icon(ampm_icon)} Time of Day:** {res['ampm']}", unsafe_allow_html=True)
        if "drift" in res:
            st.markdown(f"**{icon('compare_arrows')} Accuracy vs Real-time:** {res['drift']}", unsafe_allow_html=True)
        if "ambiguity" in res and res["ambiguity"]:
            st.warning(f"{res['ambiguity']}")

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
nav_button("comparator", "Time Comparator", "compare")
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
    
    st.markdown(f"#### {icon('edit')} Manual Gauge Scale Overrides", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    manual_min = colA.text_input("Min Value (Optional)", "")
    manual_max = colB.text_input("Max Value (Optional)", "")

    if uploaded_file and st.button("Run Analysis", type="primary"):
        with st.spinner("Processing..."):
            try:
                from datetime import datetime
                device_time_str = datetime.now().isoformat()
                
                image = Image.open(uploaded_file)
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                files = {"file": ("image.jpg", img_byte_arr.getvalue(), "image/jpeg")}
                data_form = {
                    "force_expert": str(force_expert),
                    "manual_min_val": manual_min if manual_min.strip() else "",
                    "manual_max_val": manual_max if manual_max.strip() else "",
                    "device_time_str": device_time_str
                }
                response = requests.post(f"{API_URL}/analyze", files=files, data=data_form)
                if response.status_code == 200: display_results(response.json())
                else: st.error(f"Server Error: {response.status_code}")
            except Exception as e: st.error(f"Connection Failed: {e}")

# --- PAGE 2: WEBCAM ---
elif st.session_state.page == "webcam":
    st.markdown(f"## {icon('videocam')} Real-Time Analysis", unsafe_allow_html=True)
    st.info("Running C1 (Localization) + C2 (Pose) locally. Processes every 5th frame.")
    
    # Camera Source Selection
    cam_source = st.radio("Select Camera Source", ["Local Webcam", "IP Camera (RTSP)"], horizontal=True)
    
    col1, col2 = st.columns([3, 1])
    
    # We will store the stream loop execution to the end of the block
    run_ip_cam_loop = False
    rtsp_url = ""
    
    with col1:
        if cam_source == "Local Webcam":
            rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            ctx = webrtc_streamer(
                key="harp-ai", 
                video_processor_factory=ClockProcessor, 
                rtc_configuration=rtc_configuration, 
                media_stream_constraints={"video": True, "audio": False}, 
                async_processing=True
            )
            expert_controls_ctx = ctx
        else:
            # IP Camera (RTSP)
            rtsp_url = st.text_input("RTSP Stream URL", placeholder="rtsp://username:password@ip_address:port/stream")
            start_ip_cam = st.checkbox("Start IP Camera Stream")
            
            expert_controls_ctx = "ip_cam"
            if start_ip_cam and rtsp_url:
                run_ip_cam_loop = True
                stframe = st.empty()
                
                if "ip_cam_engine" not in st.session_state:
                    from app.core.engine import HARPEngine
                    st.session_state.ip_cam_engine = HARPEngine(parent_dir)
                if "ip_cam_expert" not in st.session_state:
                    st.session_state.ip_cam_expert = False
                if "ip_cam_manual_min" not in st.session_state:
                    st.session_state.ip_cam_manual_min = ""
                if "ip_cam_manual_max" not in st.session_state:
                    st.session_state.ip_cam_manual_max = ""

    with col2:
        st.markdown(f"### {icon('tune')} Controls", unsafe_allow_html=True)
        
        # Expert Mode Toggle
        st.markdown(f"{icon('military_tech')} **Force Expert Mode**", unsafe_allow_html=True)
        if cam_source == "Local Webcam" and expert_controls_ctx and expert_controls_ctx.video_processor:
            expert_controls_ctx.video_processor.force_expert = st.checkbox("Enable C3/XAI", value=False, key="webcam_expert")
        elif cam_source == "IP Camera (RTSP)":
            expert_enabled = st.checkbox("Enable C3/XAI", value=st.session_state.get("ip_cam_expert", False), key="ip_expert")
            st.session_state.ip_cam_expert = expert_enabled
            
        st.markdown("---")
        
        # Manual Scale Overrides
        st.markdown(f"#### {icon('edit')} Manual Gauge Scale", unsafe_allow_html=True)
        colA, colB = st.columns(2)
        
        if cam_source == "Local Webcam" and expert_controls_ctx and expert_controls_ctx.video_processor:
            manual_min = colA.text_input("Min Value", "", key="webcam_min")
            manual_max = colB.text_input("Max Value", "", key="webcam_max")
            expert_controls_ctx.video_processor.manual_min_val = manual_min if manual_min.strip() else ""
            expert_controls_ctx.video_processor.manual_max_val = manual_max if manual_max.strip() else ""
        elif cam_source == "IP Camera (RTSP)":
            manual_min = colA.text_input("Min Value", st.session_state.get("ip_cam_manual_min", ""), key="ip_min")
            manual_max = colB.text_input("Max Value", st.session_state.get("ip_cam_manual_max", ""), key="ip_max")
            st.session_state.ip_cam_manual_min = manual_min if manual_min.strip() else ""
            st.session_state.ip_cam_manual_max = manual_max if manual_max.strip() else ""

        st.markdown("---")
        if st.button("Reset Connection"): 
            st.cache_resource.clear()
            st.rerun()

    # Execute IP Camera Loop AFTER all UI is rendered
    if run_ip_cam_loop:
        # Optimization: use cv2.CAP_FFMPEG explicitly with TCP for stable RTSP parsing if needed, 
        # but standard VideoCapture usually works best with minimal buffering settings.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp" # or tcp
        
        # Instantiate MULTITHREADED streamer
        if "cam_stream" not in st.session_state or st.session_state.cam_stream_url != rtsp_url:
            if "cam_stream" in st.session_state:
                st.session_state.cam_stream.stop()
            st.session_state.cam_stream = CameraStream(rtsp_url).start()
            st.session_state.cam_stream_url = rtsp_url
            
        stream = st.session_state.cam_stream
        
        # Give stream a moment to connect
        time.sleep(0.5) 
        
        # We check stream.stream.isOpened() to verify OpenCV connected properly at init
        if not stream.stream.isOpened():
            stframe.error("Failed to open RTSP stream. Check the URL and network connection. IP Cameras require authentication in the format rtsp://USER:PASS@IP:PORT/stream.")
            stream.stop()
            del st.session_state.cam_stream
        else:
            frame_count = 0
            last_time = time.time()
            last_ui_update_time = time.time()
            fps = 0
            last_result = None
            
            # Target ~15 FPS for the frontend UI to prevent Streamlit websocket lag
            UI_UPDATE_INTERVAL = 1.0 / 15.0 
            
            # Analyze interval (e.g. 5 means AI runs roughly ~5-6 times a second assuming UI loop runs at 30 FPS)
            ANALYZE_INTERVAL = 5
            
            while start_ip_cam:  # Use the checkbox state from UI to control the loop
                # Instantly grab latest frame from background thread (no blocking!)
                frame = stream.read()
                
                if frame is None:
                    # Thread might be reconnecting or stopped
                    time.sleep(0.1)
                    continue
                    
                frame_count += 1
                now = time.time()
                
                # FPS Calculation
                if now - last_time > 1:
                    fps = frame_count
                    frame_count = 0
                    last_time = now
                    
                # 1. AI Inference
                if frame_count % ANALYZE_INTERVAL == 0:
                    try:
                        # Optional: resize frame before inference to save CPU if it's huge (e.g. 4k)
                        # small_frame = cv2.resize(frame, (640, 480)) 
                        # We must copy the frame because Streamlit/AI shouldn't modify the thread's raw array directly
                        frame_for_ai = frame.copy() 
                        last_result = st.session_state.ip_cam_engine.analyze(
                            frame_for_ai,
                            force_expert=st.session_state.ip_cam_expert,
                            manual_min_val=st.session_state.ip_cam_manual_min,
                            manual_max_val=st.session_state.ip_cam_manual_max,
                            enable_temporal=True,   # [Tier 1.4]
                        )
                    except Exception as e:
                        print(f"AI Error: {e}")
                        
                # 2. Draw Overlays (We draw instantly on the copied frame before rendering)
                display_frame = frame.copy()
                if last_result:
                    res = last_result
                    display_val = res.get('time', '--')
                    cv2.putText(display_frame, f"READING: {display_val}", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
                    
                    method = res.get('method', 'Unknown')
                    color = (0, 255, 0) if "Fast" in method or "Gauge" in method else (0, 0, 255)
                    cv2.putText(display_frame, f"Mode: {method}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    if "angles" in res and res["angles"].get("hand1", 0) != 0.0:
                        a1 = res["angles"].get("hand1", 0)
                        a2 = res["angles"].get("hand2", 0)
                        cv2.putText(display_frame, f"H:{a1:.0f} M:{a2:.0f}", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.putText(display_frame, f"Pipeline FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # 3. Streamlit UI Update
                if now - last_ui_update_time >= UI_UPDATE_INTERVAL:
                    # Convert BGR to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                    last_ui_update_time = now
                    
                # Sleep briefly to free CPU for the background reading thread if needed
                time.sleep(0.01)

# --- PAGE 3: BATCH ---
elif st.session_state.page == "batch":
    st.markdown(f"## {icon('perm_media')} Batch Processing", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)
    if uploaded_files and st.button("Process All"):
        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
        data_form = {
            "force_expert": "False",
            "manual_min_val": "",
            "manual_max_val": ""
        }
        with st.spinner("Processing Batch..."):
            try:
                res = requests.post(f"{API_URL}/analyze_batch", files=files, data=data_form)
                if res.status_code == 200:
                    data = res.json()
                    st.markdown(f"#### {icon('check_circle')} Processed {data['total_images']} images", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(data["results"]), use_container_width=True)
                else: st.error("Batch failed.")
            except Exception as e: st.error(f"Error: {e}")

# --- PAGE COMP: TIME COMPARATOR ---
elif st.session_state.page == "comparator":
    st.markdown(f"## {icon('compare')} Time Comparator", unsafe_allow_html=True)
    st.markdown("Upload a 'Before' and 'After' picture of a clock to calculate elapsed time.")
    
    col1, col2 = st.columns(2)
    file_before = col1.file_uploader("Upload 'Before' Clock", type=["jpg", "png", "jpeg"], key="fb")
    file_after = col2.file_uploader("Upload 'After' Clock", type=["jpg", "png", "jpeg"], key="fa")
    
    if file_before and file_after:
        if st.button("Compare Times", type="primary"):
            with st.spinner("Analyzing both clocks..."):
                try:
                    img_b = io.BytesIO(file_before.getvalue())
                    img_a = io.BytesIO(file_after.getvalue())
                    files = {
                        "file_before": ("before.jpg", img_b, "image/jpeg"),
                        "file_after": ("after.jpg", img_a, "image/jpeg")
                    }
                    response = requests.post(f"{API_URL}/compare_times", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"**Elapsed Time:** {data['elapsed_text']}")
                        c1, c2 = st.columns(2)
                        c1.metric("Time Before", data['time_before'])
                        c2.metric("Time After", data['time_after'])
                    else:
                        st.error(f"Error: {response.json().get('error', 'Unknown Error')}")
                except Exception as e:
                    st.error(f"Failed to connect: {e}")

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