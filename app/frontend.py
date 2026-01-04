import streamlit as st
import requests
import base64
from PIL import Image
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_URL = "http://localhost:8000"
st.set_page_config(page_title="Clock AI Research", layout="wide", page_icon="🕰️")

# ===== SIDEBAR NAVIGATION =====
st.sidebar.markdown("### Multi-Model Ensemble Architecture")
st.sidebar.markdown("#### for Analog Clock Reading")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    ["📊 Analysis", "📦 Batch Processing", "📈 Performance Dashboard"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("Year 4 Research Project\nComponents C1-C5")

# ===== PAGE 1: ANALYSIS =====
if page == "📊 Analysis":
    st.title("🕰️ Multi-Stage Clock Reasoning")
    st.markdown("### Single Image Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Clock Image", type=["jpg", "png", "jpeg"])
    with col2:
        force_expert = st.checkbox("Force Expert Path (Activate C3 + XAI)", value=False)
        st.caption("Forces the Logic Engine to trigger the heavy ResNet model.")

    if uploaded_file:
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Processing through Hybrid Cascade Architecture..."):
                try:
                    # Prepare Request
                    image = Image.open(uploaded_file)
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format)
                    files = {"file": ("image.jpg", img_byte_arr.getvalue(), "image/jpeg")}
                    data_form = {"force_expert": str(force_expert)}

                    # Call Backend
                    response = requests.post(f"{API_URL}/analyze", files=files, data=data_form)
                    
                    if response.status_code == 200:
                        data = response.json()
                        res = data["result"]
                        viz = data.get("visualizations", {})
                        
                        if "error" in res:
                            st.error(f"✗ Analysis Failed: {res['error']}")
                        else:
                            # Success Banner
                            st.success(f"✅ Analysis Complete in {data['processing_time']:.3f}s")
                            
                            # Method Indicator
                            method_color = "green" if "Fast Path" in res["method"] else "orange"
                            st.markdown(f"**Method Used:** :{method_color}[{res['method']}]")
                            
                            # 4-Column Stage Indicator
                            stages = ["C1 Localization", "C2 Hand Pose", "C3 Refinement", "C4 Physics"]
                            active_stages = ["C1", "C2", "C4"]
                            if "Expert" in res["method"]: active_stages.append("C3")
                            
                            cols = st.columns(4)
                            for c, stage in zip(cols, stages):
                                code = stage.split()[0]
                                if code in active_stages:
                                    c.success(f"✅ {stage}")
                                else:
                                    c.markdown(f"⚪ {stage}")
                            
                            st.markdown("---")

                            # TABS FOR VISUALIZATION
                            tab1, tab2, tab3, tab4 = st.tabs(["🔍 C1: Localization", "✋ C2: Structure", "🎯 C3: Expert AI", "🧮 C4: Result"])
                            
                            with tab1:
                                if "c1_detection" in viz:
                                    st.image(base64.b64decode(viz["c1_detection"]), caption="YOLO Localization", use_container_width=True)
                            
                            with tab2:
                                if "c2_hands" in viz:
                                    st.image(base64.b64decode(viz["c2_hands"]), caption="Skeleton Keypoints", use_container_width=True)
                                if "angles" in res:
                                    c1, c2 = st.columns(2)
                                    c1.metric("Hand 1 Angle", f"{res['angles']['hand1']:.1f}°")
                                    c2.metric("Hand 2 Angle", f"{res['angles']['hand2']:.1f}°")

                            with tab3:
                                if "c3_crops" in viz and viz["c3_crops"]:
                                    st.markdown("**Hand Crops (ResNet Input)**")
                                    c_cols = st.columns(len(viz["c3_crops"]))
                                    for idx, (col, crop) in enumerate(zip(c_cols, viz["c3_crops"])):
                                        col.image(base64.b64decode(crop), caption=f"Hand {idx+1}")
                                    
                                    if data.get("heatmap_b64"):
                                        st.markdown("**XAI Attention Map (Grad-CAM)**")
                                        st.image(base64.b64decode(data["heatmap_b64"]), caption="Model Focus Area", use_container_width=True)
                                else:
                                    st.info("Fast Path Used - Deep Learning (C3) was skipped for efficiency.")

                            with tab4:
                                st.markdown(f"# 🕒 Time: {res['time']}")
                                st.code(res.get("reasoning", "No reasoning provided"))
                                if res.get("debug"):
                                    with st.expander("Debug Logs"):
                                        st.write(res["debug"])

                    else:
                        st.error(f"Server Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")

# ===== PAGE 2: BATCH =====
elif page == "📦 Batch Processing":
    st.title("📦 Batch Processing")
    uploaded_files = st.file_uploader("Upload Multiple Images", accept_multiple_files=True)
    
    if uploaded_files and st.button("Process All"):
        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
        with st.spinner("Processing Batch..."):
            try:
                res = requests.post(f"{API_URL}/analyze_batch", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Processed {data['total_images']} images!")
                    
                    df = pd.DataFrame(data["results"])
                    st.dataframe(df, use_container_width=True)
                    
                    # Color Chart for Batch
                    if not df.empty:
                        fig = px.pie(df, names="method", title="Batch Method Distribution", 
                                     color_discrete_sequence=px.colors.sequential.RdBu)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Batch failed.")
            except Exception as e:
                st.error(f"Error: {e}")

# ===== PAGE 3: DASHBOARD (COLORFUL VERSION) =====
elif page == "📈 Performance Dashboard":
    st.title("📈 Analytics Dashboard")
    st.markdown("### Real-Time System Metrics")
    
    # Refresh & Wipe
    c1, c2 = st.columns([1, 4])
    if c1.button("🔄 Refresh"): st.rerun()
    if c2.button("🗑️ Reset DB"): 
        requests.post(f"{API_URL}/metrics/clear")
        st.rerun()

    try:
        # Get Data
        metrics = requests.get(f"{API_URL}/metrics").json()
        
        # 1. KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Scans", metrics["total_analyses"], delta="Count")
        k2.metric("Success Rate", f"{metrics['success_rate']:.1f}%", delta="Reliability")
        k3.metric("Avg Latency", f"{metrics['avg_processing_time']:.3f}s", delta="Speed", delta_color="inverse")
        k4.metric("Failures", metrics["failure_count"], delta_color="inverse")
        
        st.markdown("---")

        # 2. COLORFUL CHARTS
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("### 🛤️ Logic Path Distribution")
            # Convert dict to DataFrame for Plotly
            df_method = pd.DataFrame(list(metrics["method_usage"].items()), columns=["Method", "Count"])
            if not df_method.empty:
                fig_m = px.bar(
                    df_method, x="Method", y="Count", color="Method", 
                    text="Count", title="Fast Path vs Expert Path",
                    color_discrete_map={
                        "Fast Path (C1+C2+C4)": "#00CC96",  # Bright Green
                        "Expert Path (C1+C2+C3+C4)": "#EF553B" # Red/Orange
                    }
                )
                st.plotly_chart(fig_m, use_container_width=True)
            else:
                st.info("No data recorded yet.")

        with c2:
            st.markdown("### ⚙️ Component Activation")
            df_comp = pd.DataFrame(list(metrics["component_usage"].items()), columns=["Component", "Count"])
            if not df_comp.empty:
                # FIXED: Changed 'donut' to 'hole'
                fig_c = px.pie(
                    df_comp, names="Component", values="Count", hole=0.4,
                    title="Component Utilization",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                st.plotly_chart(fig_c, use_container_width=True)
            else:
                st.info("No data recorded yet.")

        # 3. History Table
        st.subheader("📝 Live Transaction Log")
        hist = requests.get(f"{API_URL}/metrics/history").json()
        if hist:
            df_hist = pd.DataFrame(hist)
            # Reorder cols
            if "id" in df_hist.columns:
                available_cols = [c for c in ["id", "timestamp", "image_name", "method", "processing_time", "success"] if c in df_hist.columns]
                df_hist = df_hist[available_cols]
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.caption("Table is empty.")

    except Exception as e:
        st.error(f"Dashboard Error: {e}")
        st.info("Ensure the Backend is running on Port 8000.")