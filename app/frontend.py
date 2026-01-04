import streamlit as st
import requests
import base64
from PIL import Image
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Clock AI Research - Advanced Demo", layout="wide")

# ===== SIDEBAR NAVIGATION =====
st.sidebar.title("🕰️ Clock AI Research")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["📊 Analysis", "📦 Batch Processing", "📹 Live Webcam", "📈 Performance Dashboard"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Year 4 Research Project**\nComponents C1-C5")

# ===== PAGE 1: ANALYSIS (Original) =====
if page == "📊 Analysis":
    st.title("🕰️ Multi-Stage Clock Reasoning System")
    st.markdown("### Single Image Analysis - Component Visualization")
    
    force_expert = st.checkbox("Force Expert Path (Activate C3 + XAI)", value=False)
    uploaded_file = st.file_uploader("Upload Clock Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Processing through Multi-Stage Pipeline..."):
                try:
                    image = Image.open(uploaded_file)
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format)
                    files = {"file": img_byte_arr.getvalue()}
                    data_form = {"force_expert": str(force_expert)}

                    response = requests.post(f"{API_URL}/analyze", files=files, data=data_form)
                    
                    if response.status_code == 200:
                        data = response.json()
                        res = data["result"]
                        viz = data.get("visualizations", {})
                        proc_time = data.get("processing_time", 0)
                        
                        if "error" in res:
                            st.error(f"Analysis Failed: {res['error']}")
                        else:
                            # Show processing time
                            st.success(f"✅ Analysis Complete! Processing Time: **{proc_time:.2f}s**")
                            
                            # Pipeline Stage Indicator
                            st.markdown("---")
                            stages = ["C1 Localization", "C2 Hand Detection", "C3 Refinement", "C4 Physics"]
                            active_stages = []
                            
                            if "Fast Path" in res["method"]:
                                active_stages = ["C1", "C2", "C4"]
                            elif "Expert Path" in res["method"]:
                                active_stages = ["C1", "C2", "C3", "C4"]
                            
                            cols_indicator = st.columns(4)
                            for idx, (col, stage) in enumerate(zip(cols_indicator, stages)):
                                component = stage.split()[0]
                                if component in active_stages:
                                    col.markdown(f"### ✅ {stage}")
                                else:
                                    col.markdown(f"### ⚪ {stage}")
                            
                            st.markdown("---")
                            
                            # Stage Visualizations Tabs
                            st.subheader("📊 Pipeline Stage Visualizations")
                            
                            tab1, tab2, tab3, tab4 = st.tabs([
                                "🔍 Stage 1: C1 Localization",
                                "✋ Stage 2: C2 Hand Detection", 
                                "🎯 Stage 3: C3 Refinement",
                                "🧮 Stage 4: Final Result"
                            ])
                            
                            with tab1:
                                st.markdown("**Component C1: Clock Localization (YOLO)**")
                                if "c1_detection" in viz:
                                    c1_img = base64.b64decode(viz["c1_detection"])
                                    st.image(c1_img, caption="Yellow box: Detected clock region", use_container_width=True)
                                    st.info("✓ Clock successfully localized in the image")
                            
                            with tab2:
                                st.markdown("**Component C2: Hand Skeleton Detection (YOLO Pose)**")
                                if "c2_hands" in viz:
                                    c2_img = base64.b64decode(viz["c2_hands"])
                                    st.image(c2_img, caption="Blue=Center, Green=Hand 1, Red=Hand 2", use_container_width=True)
                                    
                                    if "angles" in res:
                                        col1, col2 = st.columns(2)
                                        col1.metric("Hand 1 Angle", f"{res['angles']['hand1']:.1f}°")
                                        col2.metric("Hand 2 Angle", f"{res['angles']['hand2']:.1f}°")
                                    
                                    st.success("✓ Clock hands detected and angles calculated")
                            
                            with tab3:
                                st.markdown("**Component C3: Angle Regression (ResNet18)**")
                                if force_expert and "c3_crops" in viz and viz["c3_crops"]:
                                    st.info("Expert Path activated - showing hand crops for angle refinement")
                                    cols = st.columns(len(viz["c3_crops"]))
                                    for idx, (col, crop_b64) in enumerate(zip(cols, viz["c3_crops"])):
                                        crop_img = base64.b64decode(crop_b64)
                                        col.image(crop_img, caption=f"Hand {idx+1} Crop", use_container_width=True)
                                    
                                    if data.get("heatmap_b64"):
                                        st.markdown("**Component C5: XAI Visual Explanation (Grad-CAM)**")
                                        heatmap_bytes = base64.b64decode(data["heatmap_b64"])
                                        st.image(heatmap_bytes, caption="Attention Map", use_container_width=True)
                                    
                                    st.success("✓ Angles refined using deep learning")
                                else:
                                    if force_expert:
                                        st.warning("C3 refinement attempted but no crops available")
                                    else:
                                        st.info("Fast Path used - C3 refinement skipped")
                            
                            with tab4:
                                st.markdown("**Component C4: Physics-Based Reasoning**")
                                st.markdown(f"## 🕐 Predicted Time: **{res['time']}**")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Method", res["method"])
                                    st.metric("Confidence", res.get("confidence", "N/A"))
                                
                                with col2:
                                    if "reasoning" in res:
                                        st.info(f"**Reasoning:** {res['reasoning']}")
                                
                                if "debug" in res and res["debug"]:
                                    with st.expander("🔧 Debug Information"):
                                        for info in res["debug"]:
                                            st.text(info)
                                
                                st.success("✓ Final time calculated using physics constraints")
                    else:
                        st.error(f"Backend Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")
    else:
        st.info("👈 Upload a clock image to analyze")

# ===== PAGE 2: BATCH PROCESSING =====
elif page == "📦 Batch Processing":
    st.title("📦 Batch Image Processing")
    st.markdown("### Process Multiple Clock Images Simultaneously")
    
    force_expert_batch = st.checkbox("Force Expert Path for all images", value=False, key="batch_expert")
    uploaded_files = st.file_uploader("Upload Multiple Clock Images", 
                                      type=["jpg", "png", "jpeg"], 
                                      accept_multiple_files=True)
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} images selected")
        
        if st.button("Process Batch", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing batch..."):
                try:
                    files_data = []
                    for file in uploaded_files:
                        files_data.append(("files", (file.name, file.getvalue(), file.type)))
                    
                    data_form = {"force_expert": str(force_expert_batch)}
                    
                    response = requests.post(f"{API_URL}/analyze_batch", 
                                           files=files_data, 
                                           data=data_form)
                    
                    progress_bar.progress(100)
                    
                    if response.status_code == 200:
                        batch_data = response.json()
                        results_list = batch_data["results"]
                        
                        st.success(f"✅ Batch Processing Complete! Total: {batch_data['total_images']} images")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        successful = sum(1 for r in results_list if r["success"])
                        failed = len(results_list) - successful
                        avg_time = sum(r["processing_time"] for r in results_list) / len(results_list) if results_list else 0
                        
                        col1.metric("Total Images", len(results_list))
                        col2.metric("Successful", successful, delta_color="normal")
                        col3.metric("Failed", failed, delta_color="inverse")
                        col4.metric("Avg Time", f"{avg_time:.2f}s")
                        
                        # Results Table
                        st.subheader("📋 Detailed Results")
                        df = pd.DataFrame(results_list)
                        st.dataframe(df, use_container_width=True)
                        
                        # Export Options
                        st.subheader("💾 Export Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv,
                                file_name="batch_results.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            json_data = df.to_json(orient="records", indent=2)
                            st.download_button(
                                label="📥 Download as JSON",
                                data=json_data,
                                file_name="batch_results.json",
                                mime="application/json"
                            )
                        
                        # Visualizations
                        st.subheader("📊 Batch Analysis Visualizations")
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            # Success/Failure Pie
                            fig_status = go.Figure(data=[go.Pie(
                                labels=["Success", "Failed"],
                                values=[successful, failed],
                                marker=dict(colors=["#00cc66", "#ff4444"])
                            )])
                            fig_status.update_layout(title="Success Rate")
                            st.plotly_chart(fig_status, use_container_width=True)
                        
                        with viz_col2:
                            # Processing Time Bar
                            fig_time = px.bar(df, x="filename", y="processing_time",
                                            title="Processing Time per Image",
                                            labels={"processing_time": "Time (s)", "filename": "Image"})
                            st.plotly_chart(fig_time, use_container_width=True)
                        
                    else:
                        st.error(f"Backend Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Batch Processing Failed: {e}")
    else:
        st.info("📤 Upload multiple images to begin batch processing")

# ===== PAGE 3: LIVE WEBCAM =====
elif page == "📹 Live Webcam":
    st.title("📹 Live Webcam Analysis")
    st.markdown("### Real-Time Clock Detection")
    
    st.info("⚠️ This feature requires webcam access. Make sure to allow access when prompted.")
    
    # Note: Simplified webcam implementation
    st.warning("🚧 Live webcam feature coming soon! For now, please use the Analysis or Batch pages.")
    st.markdown("""
    **Planned Features:**
    - Real-time frame capture from webcam
    - Automatic clock detection
    - Live time display overlay
    - Recording capability
    
    **Alternative:** Use your phone camera to take a photo and upload it to the Analysis page!
    """)

# ===== PAGE 4: PERFORMANCE DASHBOARD =====
elif page == "📈 Performance Dashboard":
    st.title("📈 Performance Dashboard")
    st.markdown("### Real-Time System Metrics & Analytics")
    
    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 Refresh Metrics"):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Metrics"):
            requests.post(f"{API_URL}/metrics/clear")
            st.success("Metrics cleared!")
            st.rerun()
    
    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            metrics = response.json()
            
            # Key Metrics
            st.subheader("📊 Key Performance Indicators")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric("Total Analyses", metrics["total_analyses"])
            col2.metric("Success Rate", f"{metrics['success_rate']:.1f}%")
            col3.metric("Avg Processing Time", f"{metrics['avg_processing_time']:.2f}s")
            col4.metric("Successful", metrics["success_count"])
            col5.metric("Failed", metrics["failure_count"])
            
            if metrics["total_analyses"] > 0:
                st.markdown("---")
                
                # Charts
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Component Usage Pie Chart
                    st.subheader("⚙️ Component Usage")
                    comp_usage = metrics["component_usage"]
                    fig_comp = go.Figure(data=[go.Pie(
                        labels=list(comp_usage.keys()),
                        values=list(comp_usage.values()),
                        marker=dict(colors=["#FFD700", "#00CED1", "#FF6347", "#9370DB"])
                    )])
                    fig_comp.update_layout(height=350)
                    st.plotly_chart(fig_comp, use_container_width=True)
                
                with chart_col2:
                    # Method Usage Bar Chart
                    st.subheader("🛤️ Method Distribution")
                    method_usage = metrics["method_usage"]
                    fig_method = px.bar(
                        x=list(method_usage.keys()),
                        y=list(method_usage.values()),
                        labels={"x": "Method", "y": "Count"},
                        color=list(method_usage.values()),
                        color_continuous_scale="viridis"
                    )
                    fig_method.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig_method, use_container_width=True)
                
                # Processing Time Chart
                st.subheader("⏱️ Processing Time Analysis")
                time_data = metrics["time_breakdown"]
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Min Time", f"{time_data['min']:.2f}s")
                col_b.metric("Max Time", f"{time_data['max']:.2f}s")
                col_c.metric("Avg Time", f"{time_data['avg']:.2f}s")
                
                if time_data["all_times"]:
                    fig_times = px.line(
                        y=time_data["all_times"],
                        labels={"y": "Time (s)", "index": "Analysis #"},
                        title="Processing Time Trend (Last 20)"
                    )
                    st.plotly_chart(fig_times, use_container_width=True)
                
                # Recent Analyses
                st.subheader("📝 Recent Analyses")
                if metrics["recent_analyses"]:
                    recent_df = pd.DataFrame(metrics["recent_analyses"])
                    display_cols = ["timestamp", "image_name", "processing_time", "success", "method", "time_detected"]
                    available_cols = [col for col in display_cols if col in recent_df.columns]
                    st.dataframe(recent_df[available_cols], use_container_width=True)
                
                # Export Metrics
                st.subheader("💾 Export Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📥 Download Full Metrics (CSV)"):
                        csv_response = requests.get(f"{API_URL}/metrics/export")
                        if csv_response.status_code == 200:
                            st.download_button(
                                label="Download CSV",
                                data=csv_response.text,
                                file_name="performance_metrics.csv",
                                mime="text/csv"
                            )
                
                with col2:
                    st.info(f"⏰ System Uptime: {metrics['uptime']:.2f} hours")
            
            else:
                st.info("📭 No analyses performed yet. Run some analyses to see metrics here!")
        
        else:
            st.error("Failed to retrieve metrics from backend")
            
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        st.info("Make sure the backend server is running at http://localhost:8000")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Clock AI Research © 2026")
