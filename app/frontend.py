import streamlit as st
import requests
from PIL import Image
import io

# CONFIG
API_URL = "http://localhost:8000/analyze"

st.set_page_config(page_title="Clock Reader Research", layout="wide", page_icon="🕰️")

# CSS for Logic Trace box
st.markdown("""
    <style>
    .trace-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        font-family: monospace;
        white-space: pre-wrap;
        border-left: 5px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🕰️ Analog Clock Reasoning System")
st.write("**Year 4 Research Project** | Multi-Stage AI Architecture (C1-C4)")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Input Data")
    uploaded_file = st.file_uploader("Upload Clock Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Clock", use_container_width=True)
        
        if st.button("Analyze Clock", type="primary"):
            with st.spinner("Processing C1 -> C2 -> C3 -> C4..."):
                try:
                    # Convert to bytes for API
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format)
                    img_byte_arr = img_byte_arr.getvalue()

                    files = {"file": ("clock.jpg", img_byte_arr, "image/jpeg")}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['result'] = result
                    else:
                        st.error(f"Error: {response.json().get('error')}")
                except Exception as e:
                    st.error(f"Connection Error: Is the backend running? ({e})")

with col2:
    st.header("2. AI Reasoning Output")
    
    if 'result' in st.session_state:
        res = st.session_state['result']
        
        # Big Time Display
        st.metric(label="PREDICTED TIME", value=res['time'], delta=f"Confidence: {res['confidence']}")
        
        st.subheader("Component 4: Reasoning Trace")
        st.markdown(f'<div class="trace-box">{res["trace"]}</div>', unsafe_allow_html=True)
        
        st.subheader("Technical Telemetry")
        st.json({
            "Detected Angle A": f"{res.get('hour_angle', 0):.2f}°",
            "Detected Angle B": f"{res.get('minute_angle', 0):.2f}°",
            "Model": "YOLOv8-Pose + ResNet18-Regression"
        })
    else:
        st.info("Upload an image to see the reasoning engine in action.")