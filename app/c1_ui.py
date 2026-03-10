import streamlit as st
import base64

def render_c1_localization(viz, res):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if "c1_detection" in viz:
            st.image(base64.b64decode(viz["c1_detection"]), caption="Detection Region", width=350)
            
        if "c1_quality" in res:
            q = res["c1_quality"]
            st.markdown("##### Image Quality Score")
            overall = int(q.get('overall', 0))
            
            raw_blur = q.get('blur', 0)
            raw_bright = q.get('brightness', 0)
            raw_cont = q.get('contrast', 0)
            
            blur_perc = min(raw_blur / 500.0, 1.0)
            bright_perc = max(0.0, 1.0 - (abs(raw_bright - 127) / 127.0))
            cont_perc = min(raw_cont / 60.0, 1.0)
            
            st.progress(overall / 100.0, text=f"Overall Tracking Quality: {overall}/100")
            
            st.markdown("<br>", unsafe_allow_html=True) # Added spacing
            
            col_sharp, col_bright, col_cont = st.columns(3)
            with col_sharp:
                st.markdown(f"<span style='color:white; font-size:18px; font-weight:500;'>Sharpness</span> <span style='color:gray; font-size:12px;'>(Raw: {raw_blur:.1f})</span>", unsafe_allow_html=True)
                st.progress(blur_perc, text=f"{int(blur_perc*100)}% {'Clear' if blur_perc > 0.5 else 'Blurry'}")
            
            with col_bright:
                st.markdown(f"<span style='color:white; font-size:18px; font-weight:500;'>Brightness</span> <span style='color:gray; font-size:12px;'>(Raw: {raw_bright:.1f})</span>", unsafe_allow_html=True)
                st.progress(bright_perc, text=f"{int(bright_perc*100)}% {'Optimal' if bright_perc > 0.5 else 'Dark'}")
            
            with col_cont:
                st.markdown(f"<span style='color:white; font-size:18px; font-weight:500;'>Contrast</span> <span style='color:gray; font-size:12px;'>(Raw: {raw_cont:.1f})</span>", unsafe_allow_html=True)
                st.progress(cont_perc, text=f"{int(cont_perc*100)}% {'High' if cont_perc > 0.5 else 'Low'}")
            
    with col2:
        st.markdown("### Detection Confidence")
        
        type_str = "Gauge" if "Gauge" in res.get("method", "") else "Clock"
        st.markdown(f"**Identified Class:** `{type_str}`")
        
        if "c1_conf" in res:
            conf_perc = res.get("c1_conf", 0.0)
            conf_display = conf_perc * 100
            
            st.markdown(f"**Detection Probability:** {conf_display:.1f}%")
            if conf_display >= 80:
                st.progress(conf_perc, text=f"High Confidence ")
                st.success("Target clearly identified")
            elif conf_display >= 50:
                st.progress(conf_perc, text=f"Medium Confidence ")
                st.warning("Target found, but the model is somewhat unsure")
            else:
                st.progress(max(0.0, conf_perc), text=f"Low Confidence ")
                st.error("Target identification is poor")
        else:
            st.info("No confidence data available.")
