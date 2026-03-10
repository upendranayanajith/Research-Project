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
            
            col_sharp, col_bright, col_cont = st.columns(3)
            with col_sharp:
                st.caption(f"Sharpness (Raw: {raw_blur:.1f})")
                st.progress(blur_perc, text=f"{int(blur_perc*100)}% {'Clear' if blur_perc > 0.5 else 'Blurry'}")
            
            with col_bright:
                st.caption(f"Brightness (Raw: {raw_bright:.1f})")
                st.progress(bright_perc, text=f"{int(bright_perc*100)}% {'Optimal' if bright_perc > 0.5 else 'Dark'}")
            
            with col_cont:
                st.caption(f"Contrast (Raw: {raw_cont:.1f})")
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
                st.success("Target clearly identified. The model is highly confident this is a valid reading target.")
            elif conf_display >= 50:
                st.progress(conf_perc, text=f"Medium Confidence ")
                st.warning("Target found, but the model is somewhat unsure. Check for distortion, glare, or partial occlusion.")
            else:
                st.progress(max(0.0, conf_perc), text=f"Low Confidence ")
                st.error("Target identification is poor. The image may not contain a valid gauge or clock face.")
        else:
            st.info("No confidence data available.")
