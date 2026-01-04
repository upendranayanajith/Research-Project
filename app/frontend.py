import streamlit as st
import requests
import base64
from PIL import Image
import io

API_URL = "http://localhost:8000/analyze"

st.set_page_config(page_title="Clock AI Research", layout="wide")

st.title("🕰️ Multi-Stage Clock Reasoning System")
st.markdown("### Year 4 Research Project | C1 - C5 Architecture")

# Sidebar
st.sidebar.header("Controls")
force_expert = st.sidebar.checkbox("Force Expert Path (Activate C3 + XAI)", value=False)
uploaded_file = st.sidebar.file_uploader("Upload Clock Image", type=["jpg", "png", "jpeg"])

col1, col2 = st.columns(2)

if uploaded_file:
    # Display Input
    image = Image.open(uploaded_file)
    with col1:
        # FIX 1: Updated parameter to fix deprecation warning
        st.image(image, caption="Input (Component 1)", width="stretch")
    
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Processing through Hybrid Cascade..."):
            try:
                # Prepare Request
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                files = {"file": img_byte_arr.getvalue()}
                data = {"force_expert": str(force_expert)}

                # Call API
                response = requests.post(API_URL, files=files, data=data)
                
                if response.status_code == 200:
                    data = response.json()
                    res = data["result"]
                    
                    # FIX 2: Check for "error" key before accessing "time"
                    if "error" in res:
                        st.error(f"Analysis Failed: {res['error']}")
                        st.warning("Try uploading a clearer image of a clock.")
                    else:
                        # Display Result
                        with col2:
                            st.subheader("Inference Result")
                            st.metric("Predicted Time", res["time"])
                            st.info(f"**Method:** {res['method']}")
                            
                            # Display XAI Heatmap (Component 5)
                            if data.get("heatmap_b64"):
                                st.subheader("Component 5: XAI Visual Explanation")
                                heatmap_bytes = base64.b64decode(data["heatmap_b64"])
                                st.image(heatmap_bytes, caption="Grad-CAM Attention Map (C3)", width="stretch")
                            elif force_expert:
                                st.warning("Expert path ran, but no heatmap generated.")
                            else:
                                st.markdown("*Heatmap not generated (Fast Path used).*")
                else:
                    st.error(f"Backend Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")