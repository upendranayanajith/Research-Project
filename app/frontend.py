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

from app.c1_ui import render_c1_localization

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
@st.dialog("Cognitive Reasoning Report", width="large")
def show_reasoning_report(data):
    res = data["result"]
    
    st.markdown(f"### {icon('description')} Official Diagnostic Report")
    st.markdown(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"#### {icon('summarize')} Executive Summary")
        st.write(f"**Final Reading:** :green[{res.get('time', 'N/A')}]")
        st.write(f"**Detection Method:** {res.get('method', 'N/A')}")
        st.write(f"**Confidence:** {res.get('confidence', 'N/A')}")
        st.write(f"**AM/PM Inference:** {res.get('ampm', 'N/A')}")
    
    with col2:
        st.markdown(f"#### {icon('checklist')} Pipeline Status")
        stages = ["C1 Detection", "C2 Keypoints", "C3 Expert AI", "C4 Physics"]
        for s in stages:
            is_done = True if "Expert" in res["method"] else (s != "C3 Expert AI")
            st.markdown(f"{icon('check_circle' if is_done else 'cancel', color='green' if is_done else 'red')} {s}")

    st.markdown("---")
    st.markdown(f"#### {icon('psychology')} AI Logical Trace")
    if "debug" in res:
        for trace in res["debug"]:
            if "C4 Telemetry Trace" in trace:
                st.info(trace)
            elif "Heuristics" in trace:
                st.success(trace)
            else:
                st.markdown(f"{icon('chevron_right', size=18)} {trace}")
    else:
        st.warning("No trace logs available for this session.")

    st.markdown("---")
    st.markdown(f"#### {icon('analytics')} Physics Validation")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Angular Shift (H):** {res.get('angles', {}).get('hand1', 0):.1f}°")
        st.write(f"**Angular Shift (M):** {res.get('angles', {}).get('hand2', 0):.1f}°")
    with col_b:
        st.write(f"**Ambiguity:** {res.get('ambiguity', 'None detected')}")
        st.write(f"**Accuracy:** {res.get('drift', 'N/A')}")

    if st.button("Close Report", type="primary"):
        st.rerun()

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
    
    type_identified = "Gauge" if "Gauge" in res["method"] else "Clock"
    
    st.markdown(f"**Identified Type:** <span style='color:white; font-size:24px'> {type_identified}</span>", unsafe_allow_html=True)
    st.markdown(f"**Method Used:** <span style='color:{method_color}; font-size:22px'>{icon(method_icon, size=20)} {res['method']}</span>", unsafe_allow_html=True)
    
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
        render_c1_localization(viz, res)
    with tab2:
        st.markdown(f"### {icon('timeline')} C2 — Skeleton Structure Analysis", unsafe_allow_html=True)
        c2r = data.get("c2_research")
        
        sub1, sub2, sub3, sub4, sub5, sub6, sub7 = st.tabs([
            "✏️ Skeleton", "🔬 Scale Analysis", "🔮 3D Reconstruction",
            "🟢 Manifold", "⏱ Temporal", "👁️ Shadow Filter", "📊 Impact Summary"
        ])
        
        # ── SUB-TAB 1: Skeleton ──
        with sub1:
            if c2r and "skeleton" in c2r and c2r["skeleton"].get("image"):
                st.image(base64.b64decode(c2r["skeleton"]["image"]), caption="YOLO-Pose Skeleton", width=400)
                st.caption(f"Detected Type: **{c2r['skeleton'].get('detected_type', 'N/A')}** | Keypoints: {c2r['skeleton'].get('num_keypoints', 'N/A')} | Avg Confidence: {c2r['skeleton'].get('avg_confidence', 0):.2f}")
            elif "c2_skeleton" in viz:
                st.image(base64.b64decode(viz["c2_skeleton"]), caption="YOLO-Pose Skeleton", width=400)
            else:
                st.info("No skeleton data available.")
        
        # ── SUB-TAB 2: Scale Analysis (GAP 3) ──
        with sub2:
            if c2r and "scale_analysis" in c2r:
                sa = c2r["scale_analysis"]
                st.markdown(f"**GAP 3 — Multi-Scale LVM Oracle**")
                
                score_method = sa.get("score_method", "Unknown")
                method_color = "#a78bfa" if "LVM" in score_method else "#38bdf8"
                st.markdown(f"""
                <span style="background:{method_color}22;color:{method_color};padding:3px 12px;border-radius:4px;font-size:12px;font-weight:700;border:1px solid {method_color}44;">{score_method}</span>
                """, unsafe_allow_html=True)
                
                pyramid_imgs = sa.get("pyramid_images", [])
                scales = sa.get("scales", [])
                lvm_scores = sa.get("lvm_scores", [])
                best_idx = sa.get("best_index", -1)
                
                if pyramid_imgs and len(pyramid_imgs) > 0:
                    cols_py = st.columns(len(pyramid_imgs))
                    for i, (col, img_b64) in enumerate(zip(cols_py, pyramid_imgs)):
                        with col:
                            label = f"σ={scales[i]}" if i < len(scales) else f"σ=?"
                            if img_b64:
                                st.image(base64.b64decode(img_b64), caption=label, use_container_width=True)
                            score = lvm_scores[i] if i < len(lvm_scores) else 0
                            bar_color = "🟨" if i == best_idx else "🟦"
                            st.markdown(f"{bar_color} **{score:.4f}**")
                    
                    st.caption("Scale Pyramid + Coherence Scores")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Optimal Scale σ*", f"{sa.get('optimal_sigma', 'N/A')}")
                col_b.metric("Confidence Margin", f"{sa.get('confidence_margin', 0):.3f}")
                col_c.metric("Scoring Method", score_method.split("(")[0].strip())
                
                summary = sa.get("summary", "")
                if summary:
                    st.info(summary)
            else:
                st.info("Scale analysis data not available.")
        
        # ── SUB-TAB 3: 3D Reconstruction (GAP 1) ──
        with sub3:
            if c2r and "reconstruction_3d" in c2r:
                r3d = c2r["reconstruction_3d"]
                st.markdown(f"**GAP 1 — Bayesian 3D Reconstruction**")
                
                conf_val = r3d.get("confidence", 0)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=conf_val,
                    number={'suffix': '', 'font': {'size': 36, 'color': 'white'}},
                    title={'text': 'Confidence', 'font': {'size': 14, 'color': '#aaa'}},
                    gauge={
                        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': '#555'},
                        'bar': {'color': '#ff4444'},
                        'bgcolor': '#333',
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 0.3], 'color': '#442222'},
                            {'range': [0.3, 0.7], 'color': '#444422'},
                            {'range': [0.7, 1.0], 'color': '#224422'}
                        ],
                    }
                ))
                occ_text = r3d.get("occlusion_risk", "N/A")
                fig.add_annotation(
                    text=f"Occlusion: {occ_text}",
                    x=0.5, y=0.35, showarrow=False,
                    font=dict(size=12, color='#ff6666' if occ_text == 'HIGH' else '#66ff66')
                )
                hour_hand_label = r3d.get("hour_hand", "N/A")
                fig.add_annotation(
                    text=f"hour: {hour_hand_label}",
                    x=0.15, y=0.55, showarrow=False,
                    font=dict(size=10, color='#aaa')
                )
                fig.update_layout(
                    paper_bgcolor='#1a1a2e',
                    plot_bgcolor='#1a1a2e',
                    height=250,
                    margin=dict(t=40, b=10, l=40, r=40),
                )
                st.plotly_chart(fig, use_container_width=False)
                st.caption("Confidence Gauge")
                
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Confidence", f"{conf_val:.3f}")
                mc2.metric("Occlusion Risk", occ_text)
                mc3.metric("Hour Hand", hour_hand_label)
                
                depth = r3d.get("depth_estimates", {})
                if depth:
                    with st.expander("Depth Estimates"):
                        for key, val in depth.items():
                            st.markdown(f"**{key}**: Distance = {val['distance_px']}px, Depth = {val['estimated_depth']}")
            else:
                st.info("3D reconstruction data not available.")
        
        # ── SUB-TAB 4: Manifold (GAP 4) ──
        with sub4:
            if c2r and "manifold" in c2r:
                mf = c2r["manifold"]
                st.markdown(f"**GAP 4 — Non-Euclidean Manifold Skeleton**")
                
                if mf.get("manifold_image"):
                    st.image(base64.b64decode(mf["manifold_image"]), use_container_width=True)
                
                curvature = mf.get("curvature", {})
                if curvature:
                    st.markdown("#### Curvature Analysis")
                    for key, val in curvature.items():
                        col_e, col_g, col_r = st.columns(3)
                        col_e.markdown(f"**{key}**")
                        col_g.markdown(f"Euclid: **{int(val['euclid_px'])}px** → Geodesic: **{int(val['geodesic_px'])}px**")
                        ratio_color = "red" if val['ratio'] > 1.3 else "green"
                        col_r.markdown(f"Ratio: <span style='color:{ratio_color};font-weight:bold'>{val['ratio']}</span>", unsafe_allow_html=True)
            else:
                st.info("Manifold data not available.")
        
        # ── SUB-TAB 5: Temporal (GAP 2) ──
        with sub5:
            if c2r and "temporal" in c2r:
                tp = c2r["temporal"]
                st.markdown(f"**GAP 2 — Persistent Homology Tracking**")
                
                status = tp.get("status", "UNKNOWN")
                badge_color = "#00cc66" if status == "NOMINAL" else ("#ffaa00" if status == "OVERLAP" else "#ff4444")
                st.markdown(f"""
                <div style="background:#1a1a2e;padding:15px;border-radius:10px;display:inline-flex;align-items:center;gap:15px;margin-bottom:20px;">
                    <div style="text-align:center;padding:8px 15px;background:#222;border-radius:5px;">
                        <div style="color:#888;font-size:11px;">β₀</div>
                        <div style="color:white;font-size:28px;font-weight:bold;">{tp.get('beta_0', '?')}</div>
                    </div>
                    <div style="text-align:center;padding:8px 15px;background:#222;border-radius:5px;">
                        <div style="color:#888;font-size:11px;">β₁</div>
                        <div style="color:white;font-size:28px;font-weight:bold;">{tp.get('beta_1', '?')}</div>
                    </div>
                    <div style="background:{badge_color};color:white;padding:8px 20px;border-radius:5px;font-weight:bold;font-size:16px;">
                        {status}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Topology Status")
                
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("β₀ (Components)", tp.get("beta_0", "?"))
                tc2.metric("β₁ (Loops)", tp.get("beta_1", "?"))
                tc3.metric("Status", status)
            else:
                st.info("Temporal data not available.")
        
        # ── SUB-TAB 6: Shadow Filter (GAP 5) ──
        with sub6:
            if c2r and "shadow_filter" in c2r and c2r["shadow_filter"].get("available"):
                sf = c2r["shadow_filter"]
                st.markdown(f"**GAP 5 — Semantic Shadow Filter**")
                
                method = sf.get("method", "N/A")
                method_color = "#a78bfa" if "LVM" in method else "#38bdf8"
                st.markdown(f"""
                <div style="display:inline-flex;gap:10px;margin-bottom:16px;">
                    <span style="background:{method_color}22;color:{method_color};padding:3px 12px;border-radius:4px;font-size:12px;font-weight:700;border:1px solid {method_color}44;">{method}</span>
                    <span style="background:#22c55e22;color:#22c55e;padding:3px 12px;border-radius:4px;font-size:12px;font-weight:700;border:1px solid #22c55e44;">τ = {sf['thresholds']['accept']} / {sf['thresholds']['reject']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if "c2_shadow" in viz:
                    st.image(base64.b64decode(viz["c2_shadow"]), caption="Shadow Validation (Green=REAL, Red=SHADOW, Yellow=UNCERTAIN)", use_container_width=True)
                
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Accepted", f"{sf.get('accepted_count', 0)} / {sf.get('total', 0)}")
                sc2.metric("Rejected (Shadows)", sf.get('rejected_count', 0))
                shadow_conf = 1.0 / (1.0 + sf.get('rejected_count', 0))
                sc3.metric("C4 Shadow Confidence", f"{shadow_conf:.3f}")
                
                candidates = sf.get("candidates", [])
                if candidates:
                    st.markdown("---")
                    st.markdown("#### Per-Candidate Validation")
                    for i, cand in enumerate(candidates):
                        decision = cand.get('decision', 'UNKNOWN')
                        score = cand.get('weighted_score', 0)
                        
                        if decision == 'REAL':
                            dec_color = '#22c55e'
                            dec_icon = '✅'
                        elif decision == 'SHADOW':
                            dec_color = '#ef4444'
                            dec_icon = '❌'
                        else:
                            dec_color = '#f59e0b'
                            dec_icon = '⚠️'
                        
                        st.markdown(f"""
                        <div style="background:#0f1117;border:1px solid {dec_color}33;border-radius:8px;padding:12px 16px;margin-bottom:10px;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                                <span style="font-size:14px;font-weight:700;color:{dec_color};">{dec_icon} Candidate {i+1}: {decision}</span>
                                <span style="font-size:20px;font-weight:800;color:{dec_color};">{score:.4f}</span>
                            </div>
                            <div style="font-size:11px;color:#94a3b8;">{cand.get('reasoning', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        scores = cand.get('scores', {})
                        if scores:
                            score_cols = st.columns(4)
                            dims = [
                                ('origin_alignment', 'Origin', 0.35, '#f59e0b'),
                                ('geometry_coherence', 'Geometry', 0.30, '#22c55e'),
                                ('length_plausibility', 'Length', 0.20, '#38bdf8'),
                                ('shadow_offset_penalty', 'Shadow', 0.15, '#a78bfa'),
                            ]
                            for col, (key, label, weight, color) in zip(score_cols, dims):
                                val = scores.get(key, 0)
                                col.markdown(f"""
                                <div style="text-align:center;">
                                    <div style="font-size:10px;color:#64748b;">{label} (w={weight})</div>
                                    <div style="font-size:18px;font-weight:700;color:{color};">{val:.3f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("#### Decision Thresholds")
                thresh_cols = st.columns(3)
                thresh_cols[0].markdown("""
                <div style="background:#22c55e11;border:1px solid #22c55e33;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:18px;font-weight:800;color:#22c55e;">≥ 0.72</div>
                    <div style="font-size:11px;color:#22c55e;">REAL — Accept</div>
                </div>""", unsafe_allow_html=True)
                thresh_cols[1].markdown("""
                <div style="background:#f59e0b11;border:1px solid #f59e0b33;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:18px;font-weight:800;color:#f59e0b;">0.43–0.71</div>
                    <div style="font-size:11px;color:#f59e0b;">UNCERTAIN</div>
                </div>""", unsafe_allow_html=True)
                thresh_cols[2].markdown("""
                <div style="background:#ef444411;border:1px solid #ef444433;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:18px;font-weight:800;color:#ef4444;">≤ 0.42</div>
                    <div style="font-size:11px;color:#ef4444;">SHADOW — Reject</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Shadow filter data not available. Ensure the analysis pipeline ran successfully.")
        
        # ── SUB-TAB 7: Impact Summary ──
        with sub7:
            if c2r and "impact_summary" in c2r:
                imp = c2r["impact_summary"]
                st.markdown(f"**Research Impact Summary**")
                
                gaps = imp.get("gaps", [])
                if gaps:
                    gap_df = pd.DataFrame(gaps)
                    st.dataframe(gap_df, use_container_width=True, hide_index=True)
                
                ic1, ic2 = st.columns(2)
                ic1.metric("Active Research Gaps", f"{imp.get('num_active_gaps', 0)} / {len(gaps)}")
                ic2.metric("Confidence Boost", f"+{imp.get('overall_confidence_boost', 0):.3f}")
                
                st.success(f"All {imp.get('num_active_gaps', 0)} research components are actively enhancing the {imp.get('detected_type', 'instrument')} analysis pipeline.")
            else:
                st.info("Impact summary not available.")
        
        # ── SUB-TAB 1: Skeleton ──
        with sub1:
            if c2r and "skeleton" in c2r and c2r["skeleton"].get("image"):
                st.image(base64.b64decode(c2r["skeleton"]["image"]), caption="YOLO-Pose Skeleton", width=400)
                st.caption(f"Detected Type: **{c2r['skeleton'].get('detected_type', 'N/A')}** | Keypoints: {c2r['skeleton'].get('num_keypoints', 'N/A')} | Avg Confidence: {c2r['skeleton'].get('avg_confidence', 0):.2f}")
            elif "c2_skeleton" in viz:
                st.image(base64.b64decode(viz["c2_skeleton"]), caption="YOLO-Pose Skeleton", width=400)
            else:
                st.info("No skeleton data available.")
        
        # ── SUB-TAB 2: Scale Analysis (GAP 3) ──
        with sub2:
            if c2r and "scale_analysis" in c2r:
                sa = c2r["scale_analysis"]
                st.markdown(f"**GAP 3 — Multi-Scale LVM Oracle**")
                
                # Score method badge
                score_method = sa.get("score_method", "Unknown")
                method_color = "#a78bfa" if "LVM" in score_method else "#38bdf8"
                st.markdown(f"""
                <span style="background:{method_color}22;color:{method_color};padding:3px 12px;border-radius:4px;font-size:12px;font-weight:700;border:1px solid {method_color}44;">{score_method}</span>
                """, unsafe_allow_html=True)
                
                # Scale Pyramid Images
                pyramid_imgs = sa.get("pyramid_images", [])
                scales = sa.get("scales", [])
                lvm_scores = sa.get("lvm_scores", [])
                best_idx = sa.get("best_index", -1)
                
                if pyramid_imgs and len(pyramid_imgs) > 0:
                    # Dark container with scale images
                    cols_py = st.columns(len(pyramid_imgs))
                    for i, (col, img_b64) in enumerate(zip(cols_py, pyramid_imgs)):
                        with col:
                            label = f"σ={scales[i]}" if i < len(scales) else f"σ=?"
                            if img_b64:
                                st.image(base64.b64decode(img_b64), caption=label, use_container_width=True)
                            score = lvm_scores[i] if i < len(lvm_scores) else 0
                            bar_color = "🟨" if i == best_idx else "🟦"
                            st.markdown(f"{bar_color} **{score:.4f}**")
                    
                    st.caption("Scale Pyramid + Coherence Scores")
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Optimal Scale σ*", f"{sa.get('optimal_sigma', 'N/A')}")
                col_b.metric("Confidence Margin", f"{sa.get('confidence_margin', 0):.3f}")
                col_c.metric("Scoring Method", score_method.split("(")[0].strip())
                
                # Summary callout
                summary = sa.get("summary", "")
                if summary:
                    st.info(summary)
            else:
                st.info("Scale analysis data not available.")
        
        # ── SUB-TAB 3: 3D Reconstruction (GAP 1) ──
        with sub3:
            if c2r and "reconstruction_3d" in c2r:
                r3d = c2r["reconstruction_3d"]
                st.markdown(f"**GAP 1 — Bayesian 3D Reconstruction**")
                
                # Confidence Gauge using Plotly
                conf_val = r3d.get("confidence", 0)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=conf_val,
                    number={'suffix': '', 'font': {'size': 36, 'color': 'white'}},
                    title={'text': 'Confidence', 'font': {'size': 14, 'color': '#aaa'}},
                    gauge={
                        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': '#555'},
                        'bar': {'color': '#ff4444'},
                        'bgcolor': '#333',
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 0.3], 'color': '#442222'},
                            {'range': [0.3, 0.7], 'color': '#444422'},
                            {'range': [0.7, 1.0], 'color': '#224422'}
                        ],
                    }
                ))
                occ_text = r3d.get("occlusion_risk", "N/A")
                fig.add_annotation(
                    text=f"Occlusion: {occ_text}",
                    x=0.5, y=0.35, showarrow=False,
                    font=dict(size=12, color='#ff6666' if occ_text == 'HIGH' else '#66ff66')
                )
                hour_hand_label = r3d.get("hour_hand", "N/A")
                fig.add_annotation(
                    text=f"hour: {hour_hand_label}",
                    x=0.15, y=0.55, showarrow=False,
                    font=dict(size=10, color='#aaa')
                )
                fig.update_layout(
                    paper_bgcolor='#1a1a2e',
                    plot_bgcolor='#1a1a2e',
                    height=250,
                    margin=dict(t=40, b=10, l=40, r=40),
                )
                st.plotly_chart(fig, use_container_width=False)
                st.caption("Confidence Gauge")
                
                # Metrics row
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Confidence", f"{conf_val:.3f}")
                mc2.metric("Occlusion Risk", occ_text)
                mc3.metric("Hour Hand", hour_hand_label)
                
                # Depth estimates (collapsible)
                depth = r3d.get("depth_estimates", {})
                if depth:
                    with st.expander("Depth Estimates"):
                        for key, val in depth.items():
                            st.markdown(f"**{key}**: Distance = {val['distance_px']}px, Depth = {val['estimated_depth']}")
            else:
                st.info("3D reconstruction data not available.")
        
        # ── SUB-TAB 4: Manifold (GAP 4) ──
        with sub4:
            if c2r and "manifold" in c2r:
                mf = c2r["manifold"]
                st.markdown(f"**GAP 4 — Non-Euclidean Manifold Skeleton**")
                
                # Manifold visualization image
                if mf.get("manifold_image"):
                    st.image(base64.b64decode(mf["manifold_image"]), use_container_width=True)
                
                # Curvature data table
                curvature = mf.get("curvature", {})
                if curvature:
                    st.markdown("#### Curvature Analysis")
                    for key, val in curvature.items():
                        col_e, col_g, col_r = st.columns(3)
                        col_e.markdown(f"**{key}**")
                        col_g.markdown(f"Euclid: **{int(val['euclid_px'])}px** → Geodesic: **{int(val['geodesic_px'])}px**")
                        ratio_color = "red" if val['ratio'] > 1.3 else "green"
                        col_r.markdown(f"Ratio: <span style='color:{ratio_color};font-weight:bold'>{val['ratio']}</span>", unsafe_allow_html=True)
            else:
                st.info("Manifold data not available.")
        
        # ── SUB-TAB 5: Temporal (GAP 2) ──
        with sub5:
            if c2r and "temporal" in c2r:
                tp = c2r["temporal"]
                st.markdown(f"**GAP 2 — Persistent Homology Tracking**")
                
                # Topology status badge
                status = tp.get("status", "UNKNOWN")
                badge_color = "#00cc66" if status == "NOMINAL" else ("#ffaa00" if status == "OVERLAP" else "#ff4444")
                st.markdown(f"""
                <div style="background:#1a1a2e;padding:15px;border-radius:10px;display:inline-flex;align-items:center;gap:15px;margin-bottom:20px;">
                    <div style="text-align:center;padding:8px 15px;background:#222;border-radius:5px;">
                        <div style="color:#888;font-size:11px;">β₀</div>
                        <div style="color:white;font-size:28px;font-weight:bold;">{tp.get('beta_0', '?')}</div>
                    </div>
                    <div style="text-align:center;padding:8px 15px;background:#222;border-radius:5px;">
                        <div style="color:#888;font-size:11px;">β₁</div>
                        <div style="color:white;font-size:28px;font-weight:bold;">{tp.get('beta_1', '?')}</div>
                    </div>
                    <div style="background:{badge_color};color:white;padding:8px 20px;border-radius:5px;font-weight:bold;font-size:16px;">
                        {status}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Topology Status")
                
                # Metrics
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("β₀ (Components)", tp.get("beta_0", "?"))
                tc2.metric("β₁ (Loops)", tp.get("beta_1", "?"))
                tc3.metric("Status", status)
            else:
                st.info("Temporal data not available.")
        
        # ── SUB-TAB 6: Shadow Filter (GAP 5) ──
        with sub6:
            if c2r and "shadow_filter" in c2r and c2r["shadow_filter"].get("available"):
                sf = c2r["shadow_filter"]
                st.markdown(f"**GAP 5 — Semantic Shadow Filter**")
                
                # Method badge
                method = sf.get("method", "N/A")
                method_color = "#a78bfa" if "LVM" in method else "#38bdf8"
                st.markdown(f"""
                <div style="display:inline-flex;gap:10px;margin-bottom:16px;">
                    <span style="background:{method_color}22;color:{method_color};padding:3px 12px;border-radius:4px;font-size:12px;font-weight:700;border:1px solid {method_color}44;">{method}</span>
                    <span style="background:#22c55e22;color:#22c55e;padding:3px 12px;border-radius:4px;font-size:12px;font-weight:700;border:1px solid #22c55e44;">τ = {sf['thresholds']['accept']} / {sf['thresholds']['reject']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Validation image
                if "c2_shadow" in viz:
                    st.image(base64.b64decode(viz["c2_shadow"]), caption="Shadow Validation (Green=REAL, Red=SHADOW, Yellow=UNCERTAIN)", use_container_width=True)
                
                # Metrics row
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Accepted", f"{sf.get('accepted_count', 0)} / {sf.get('total', 0)}")
                sc2.metric("Rejected (Shadows)", sf.get('rejected_count', 0))
                shadow_conf = 1.0 / (1.0 + sf.get('rejected_count', 0))
                sc3.metric("C4 Shadow Confidence", f"{shadow_conf:.3f}")
                
                # Per-candidate results
                candidates = sf.get("candidates", [])
                if candidates:
                    st.markdown("---")
                    st.markdown("#### Per-Candidate Validation")
                    for i, cand in enumerate(candidates):
                        decision = cand.get('decision', 'UNKNOWN')
                        score = cand.get('weighted_score', 0)
                        
                        if decision == 'REAL':
                            dec_color = '#22c55e'
                            dec_icon = '✅'
                        elif decision == 'SHADOW':
                            dec_color = '#ef4444'
                            dec_icon = '❌'
                        else:
                            dec_color = '#f59e0b'
                            dec_icon = '⚠️'
                        
                        st.markdown(f"""
                        <div style="background:#0f1117;border:1px solid {dec_color}33;border-radius:8px;padding:12px 16px;margin-bottom:10px;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                                <span style="font-size:14px;font-weight:700;color:{dec_color};">{dec_icon} Candidate {i+1}: {decision}</span>
                                <span style="font-size:20px;font-weight:800;color:{dec_color};">{score:.4f}</span>
                            </div>
                            <div style="font-size:11px;color:#94a3b8;">{cand.get('reasoning', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Score breakdown
                        scores = cand.get('scores', {})
                        if scores:
                            score_cols = st.columns(4)
                            dims = [
                                ('origin_alignment', 'Origin', 0.35, '#f59e0b'),
                                ('geometry_coherence', 'Geometry', 0.30, '#22c55e'),
                                ('length_plausibility', 'Length', 0.20, '#38bdf8'),
                                ('shadow_offset_penalty', 'Shadow', 0.15, '#a78bfa'),
                            ]
                            for col, (key, label, weight, color) in zip(score_cols, dims):
                                val = scores.get(key, 0)
                                col.markdown(f"""
                                <div style="text-align:center;">
                                    <div style="font-size:10px;color:#64748b;">{label} (w={weight})</div>
                                    <div style="font-size:18px;font-weight:700;color:{color};">{val:.3f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Threshold legend
                st.markdown("---")
                st.markdown("#### Decision Thresholds")
                thresh_cols = st.columns(3)
                thresh_cols[0].markdown("""
                <div style="background:#22c55e11;border:1px solid #22c55e33;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:18px;font-weight:800;color:#22c55e;">≥ 0.72</div>
                    <div style="font-size:11px;color:#22c55e;">REAL — Accept</div>
                </div>""", unsafe_allow_html=True)
                thresh_cols[1].markdown("""
                <div style="background:#f59e0b11;border:1px solid #f59e0b33;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:18px;font-weight:800;color:#f59e0b;">0.43–0.71</div>
                    <div style="font-size:11px;color:#f59e0b;">UNCERTAIN</div>
                </div>""", unsafe_allow_html=True)
                thresh_cols[2].markdown("""
                <div style="background:#ef444411;border:1px solid #ef444433;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:18px;font-weight:800;color:#ef4444;">≤ 0.42</div>
                    <div style="font-size:11px;color:#ef4444;">SHADOW — Reject</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Shadow filter data not available. Ensure the analysis pipeline ran successfully.")
        
        # ── SUB-TAB 7: Impact Summary ──
        with sub7:
            if c2r and "impact_summary" in c2r:
                imp = c2r["impact_summary"]
                st.markdown(f"**Research Impact Summary**")
                
                # Gap table
                gaps = imp.get("gaps", [])
                if gaps:
                    gap_df = pd.DataFrame(gaps)
                    st.dataframe(gap_df, use_container_width=True, hide_index=True)
                
                # Overall metrics
                ic1, ic2 = st.columns(2)
                ic1.metric("Active Research Gaps", f"{imp.get('num_active_gaps', 0)} / {len(gaps)}")
                ic2.metric("Confidence Boost", f"+{imp.get('overall_confidence_boost', 0):.3f}")
                
                st.success(f"All {imp.get('num_active_gaps', 0)} research components are actively enhancing the {imp.get('detected_type', 'instrument')} analysis pipeline.")

    with tab3:
        debug_lines       = res.get("debug", [])
        insight_lines     = [l for l in debug_lines if "AI Insight" in l]
        routing_lines     = [l for l in debug_lines if "XAI Routing" in l]
        uncertainty_lines = [l for l in debug_lines if "uncertainty" in l.lower() or "alpha" in l.lower()]
        hand_type_lines   = [l for l in debug_lines if "Hand Type Heuristic" in l]
        temporal_xai      = res.get("temporal_xai")
        contrastive_xai   = res.get("contrastive_xai")
        has_c3            = "c3_crops" in viz and viz["c3_crops"]

        if has_c3 or insight_lines or data.get("heatmap_b64"):
            c3_sub = st.tabs([
                "\U0001f4d0 Angles",
                "\U0001f52c XAI Heatmaps",
                "\U0001f4c8 Temporal",
                "\U0001f50d Debug Log",
            ])

            # ── TAB 1: ANGLES ─────────────────────────────────────────────────
            with c3_sub[0]:
                st.markdown(f"#### {icon('psychology')} Predicted Angles", unsafe_allow_html=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    if "c3_angles" in viz:
                        st.image(base64.b64decode(viz["c3_angles"]), caption="C3 angle overlay", width=380)
                with col_b:
                    if "angles" in res and res["angles"]:
                        if "span" in res["angles"]:
                            st.metric("Scale Span", f"{res['angles'].get('span', 0):.1f}\u00b0")
                            st.metric("Needle Pos",  f"{res['angles'].get('needle', 0):.1f}\u00b0")
                            upd = res['angles'].get('units_per_deg', 0.0)
                            if upd > 0:
                                st.caption(f"1\u00b0 = {upd:.4f} scale units")
                        else:
                            st.metric("Hour hand",   f"{res['angles'].get('hand1', 0):.1f}\u00b0")
                            st.metric("Minute hand", f"{res['angles'].get('hand2', 0):.1f}\u00b0")

                if has_c3:
                    st.markdown("---")
                    st.markdown(f"**{icon('image')} ResNet-18 Input Crops**", unsafe_allow_html=True)
                    c_cols = st.columns(min(len(viz["c3_crops"]), 4))
                    for col, crop in zip(c_cols, viz["c3_crops"]):
                        col.image(base64.b64decode(crop), width=140)

                unc_val = res.get("uncertainty_deg", "")
                if unc_val and unc_val != "N/A":
                    st.markdown("---")
                    st.markdown(f"**{icon('bar_chart')} MC Dropout Uncertainty**", unsafe_allow_html=True)
                    st.success(str(unc_val))
                    for line in uncertainty_lines:
                        st.caption(f"\U0001f522 {line}")

                if hand_type_lines:
                    st.markdown("---")
                    st.markdown("**\U0001f91a Hand Type Detection**")
                    st.info(hand_type_lines[-1].replace("Hand Type Heuristic: ", ""))

            # ── TAB 2: XAI HEATMAPS ────────────────────────────────────────────
            with c3_sub[1]:
                if data.get("heatmap_b64"):
                    st.markdown(f"#### {icon('opacity')} Attention / Attribution Maps", unsafe_allow_html=True)
                    hm_tabs = st.tabs(["\U0001f525 GradCAM++", "\U0001f7e2 LIME", "\U0001f534 SHAP"])
                    with hm_tabs[0]:
                        st.image(base64.b64decode(data["heatmap_b64"]),
                                 caption="GradCAM++ \u2014 weighted fusion of ResNet-18 layer2+3+4",
                                 width=380)
                    with hm_tabs[1]:
                        if "lime_heatmap" in viz:
                            st.image(base64.b64decode(viz["lime_heatmap"]),
                                     caption="LIME \u2014 superpixel perturbation, top-5 regions highlighted",
                                     width=380)
                        else:
                            st.info("\U0001f7e2 LIME runs on the Expert Path \u2014 enable Force Expert and re-run.")
                    with hm_tabs[2]:
                        if "shap_heatmap" in viz:
                            st.image(base64.b64decode(viz["shap_heatmap"]),
                                     caption="SHAP DeepExplainer \u2014 pixel attribution (JET colormap: red = high)",
                                     width=380)
                        else:
                            st.info("\U0001f534 SHAP runs on the Expert Path \u2014 enable Force Expert and re-run.")
                else:
                    st.info("Heatmaps are generated on the Expert Path. Enable Force Expert Path and re-run.")

                st.markdown("---")
                st.markdown(f"#### {icon('psychology')} AI Explanations", unsafe_allow_html=True)
                if insight_lines:
                    for line in insight_lines:
                        label, _, text = line.partition(": ")
                        hand_label = label.replace("AI Insight ", "")
                        is_local  = "[Local XAI]" in text
                        is_gemini = "[Gemini]" in text
                        badge = "\U0001f535 Local XAI" if is_local else "\u2728 Gemini" if is_gemini else "\u2139\ufe0f"
                        clean_text = text.replace("[Local XAI] ", "").replace("[Gemini] ", "")
                        with st.container(border=True):
                            st.caption(f"{badge}  \u2014  {hand_label}")
                            st.markdown(clean_text)
                    if routing_lines:
                        for rl in routing_lines:
                            r_icon = "\u2728" if "Gemini escalated" in rl else "\U0001f535"
                            st.caption(f"{r_icon} {rl}")
                else:
                    st.info("\U0001f4a1 Enable **Force Expert Path** and re-run to generate explanations.")

                if contrastive_xai:
                    st.markdown("---")
                    st.markdown(f"#### {icon('compare_arrows')} \U0001f914 Contrastive XAI \u2014 Why not...?", unsafe_allow_html=True)
                    for line in contrastive_xai.split("\n"):
                        if not line.strip():
                            continue
                        if line.startswith("["):
                            st.markdown(line)
                        elif "\u2705" in line:
                            st.success(line.strip())
                        elif "\u274c" in line:
                            st.error(line.strip())
                        else:
                            st.markdown(line)

            # ── TAB 3: TEMPORAL STABILITY ───────────────────────────────────────
            with c3_sub[2]:
                st.markdown(f"#### {icon('bar_chart')} \U0001f4c8 Kalman Filter Temporal Stability", unsafe_allow_html=True)
                if temporal_xai:
                    t_status = temporal_xai.get("status", "N/A")
                    if t_status == "Initialising":
                        st.info(temporal_xai.get("message", "Kalman filter warming up \u2014 needs a few frames."))
                    elif t_status == "Active":
                        t_cols = st.columns(4)
                        t_cols[0].metric("Stability",       f"{temporal_xai.get('stability_score', 'N/A')}%")
                        t_cols[1].metric("Trend",           temporal_xai.get('trend', 'N/A'))
                        t_cols[2].metric("Spikes Rejected", temporal_xai.get('total_spike_count', 0))
                        t_cols[3].metric("Avg Correction",  f"{temporal_xai.get('mean_kalman_correction_deg', 0):.1f}\u00b0")
                        st.caption(f"\U0001f522 {temporal_xai.get('message', '')}")
                        with st.expander("\U0001f4ca Variance Details", expanded=False):
                            st.json({
                                "hand1_variance_deg":   temporal_xai.get("hand1_variance_deg"),
                                "hand2_variance_deg":   temporal_xai.get("hand2_variance_deg"),
                                "spike_rate_per_frame": temporal_xai.get("spike_rate_per_frame"),
                                "frames_seen":          temporal_xai.get("frames_seen"),
                            })
                    else:
                        st.warning(f"Unexpected status: {t_status}")
                else:
                    st.info("\U0001f4c8 Temporal stability is only active in **Live Webcam** or **RTSP** mode.")
                    st.caption("The Kalman filter needs multiple consecutive frames \u2014 single-image uploads do not activate it.")

            # ── TAB 4: DEBUG LOG ────────────────────────────────────────────────
            with c3_sub[3]:
                st.markdown(f"#### {icon('bug_report')} Full Pipeline Debug Log", unsafe_allow_html=True)
                st.caption(f"{len(debug_lines)} pipeline steps logged for this frame.")
                for line in debug_lines:
                    is_ok   = any(k in line for k in ["Accepted", "Gemini API", "Manual", "LIME:", "SHAP:"])
                    is_warn = "Rejected" in line or "Failed" in line
                    is_gem  = "Routing" in line and "Gemini escalated" in line
                    is_info = any(k in line.lower() for k in ["alpha", "uncertainty", "temporal", "routing", "contrastive", "hand type"])
                    icon_char = "\u2705" if is_ok else "\u26a0\ufe0f" if is_warn else "\U0001f7e2" if is_gem else "\U0001f535" if is_info else "\u25aa\ufe0f"
                    st.markdown(f"{icon_char} `{line}`")
        else:
            st.info("\U0001f4a1 Expert AI is only active when **Force Expert Path** is enabled, or the physics solver error > 20\u00b0.")
            st.caption("Fast Path and Gauge modes skip C3. Enable Force Expert Path in the sidebar and re-run.")

    with tab4:
        st.markdown(f"# {icon('schedule')} {res['time']}", unsafe_allow_html=True)
        st.markdown(f"**Reasoning:** `{res.get('reasoning', 'N/A')}`")
        if "ampm" in res:
            ampm_icon = "wb_sunny" if "AM" in res["ampm"] else "bedtime"
            st.markdown(f"**{icon(ampm_icon)} Time of Day:** {res['ampm']}", unsafe_allow_html=True)
        if "drift" in res:
            st.markdown(f"**{icon('compare_arrows')} Accuracy vs Real-time:** {res['drift']}", unsafe_allow_html=True)
        
        st.markdown("---")
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"#### {icon('wb_sunny')} AM/PM Inference", unsafe_allow_html=True)
            if "ampm" in res:
                st.info(f"Detected: **{res['ampm']}**")
            else:
                st.info("Not available.")
                
        with col_r:
            st.markdown(f"#### {icon('help_outline')} Ambiguity Analysis", unsafe_allow_html=True)
            amb_warning = res.get("ambiguity")
            candidates = res.get("ambiguity_candidates", [])
            
            if amb_warning or candidates:
                if amb_warning:
                    st.markdown(f"<p style='color:orange; font-weight:bold;'>{icon('warning', color='orange')} {amb_warning}</p>", unsafe_allow_html=True)
                
                if candidates:
                    def get_fit(err):
                        if err < 5.0: return "Excellent"
                        if err < 15.0: return "Good"
                        return "Marginal"
                        
                    cdf = pd.DataFrame([{
                        "Time": f"{c['hour']}:{c['minute']:02d}",
                        "Error (°)": round(c['error'], 2),
                        "Confidence %": round(c['confidence'], 1),
                        "Fit": get_fit(c['error']),
                    } for c in candidates])
                    st.dataframe(cdf, use_container_width=True, hide_index=True)
            else:
                st.info("No ambiguity detected.")

        st.markdown("---")
        if st.button("📄 Generate Cognitive Reasoning Report", use_container_width=True):
            show_reasoning_report(data)

# ==========================================
# CUSTOM NAVIGATION LOGIC
# ==========================================
if "page" not in st.session_state:
    st.session_state.page = "analysis"
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

def nav_button(page_key, label, icon_name):
    """Creates a navigation button with an icon."""
    c1, c2 = st.sidebar.columns([1, 4])
    with c1:
        st.markdown(f"<div style='text-align: center; padding-top: 5px;'>{icon(icon_name)}</div>", unsafe_allow_html=True)
    with c2:
        # If selected, use 'primary' style (red), else 'secondary'
        btn_type = "primary" if st.session_state.page == page_key else "secondary"
        if st.button(label, key=f"nav_{page_key}", type=btn_type, width=260):
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
    
    manual_min, manual_max = "", ""
    
    if uploaded_file:
        if "file_rotation" not in st.session_state:
            st.session_state.file_rotation = 0
            st.session_state.last_file_id = ""

        # Reset rotation and clear previous results if a new file is uploaded
        if st.session_state.last_file_id != uploaded_file.name:
            st.session_state.file_rotation = 0
            st.session_state.last_file_id = uploaded_file.name
            if "analysis_result" in st.session_state:
                del st.session_state.analysis_result

        # Main layout: Image (Left) | Controls (Right)
        col_img, col_ctrl = st.columns([2, 1])
        
        with col_img:
            try:
                image_preview = Image.open(uploaded_file)
                if st.session_state.file_rotation != 0:
                    image_preview = image_preview.rotate(-st.session_state.file_rotation * 90, expand=True)
                st.image(image_preview, caption=f"Oriented Preview ({st.session_state.file_rotation * 90}°)", width=320)
            except Exception as e:
                st.error(f"Preview Error: {e}")

        with col_ctrl:
            st.markdown(f"##### {icon('crop_rotate')} Rotate Image", unsafe_allow_html=True)
            
            # Left and Right buttons in one line
            sub_col1, sub_col2 = st.columns(2)
            if sub_col1.button("Left", key="btn_rot_l", use_container_width=True):
                st.session_state.file_rotation = (st.session_state.file_rotation - 1) % 4
                st.rerun()
            if sub_col2.button("Right", key="btn_rot_r", use_container_width=True):
                st.session_state.file_rotation = (st.session_state.file_rotation + 1) % 4
                st.rerun()
            
            # Reset button on a new line below them
            if st.button("Reset to 0°", key="btn_rot_reset", use_container_width=True):
                st.session_state.file_rotation = 0
                st.rerun()

    manual_min, manual_max = "", ""

    if uploaded_file:
        # Clear results if a new file is uploaded
        if uploaded_file.name != st.session_state.last_uploaded_file:
            st.session_state.analysis_result = None
            st.session_state.last_uploaded_file = uploaded_file.name

    if uploaded_file and st.button("Run Analysis", type="primary"):
        with st.spinner("Processing..."):
            try:
                from datetime import datetime
                device_time_str = datetime.now().isoformat()
                
                image = Image.open(uploaded_file)
                
                # Apply Component 1 Rotation if set
                if st.session_state.get("file_rotation", 0) != 0:
                    image = image.rotate(-st.session_state.file_rotation * 90, expand=True)
                
                # Convert to RGB if necessary for JPEG (handles RGBA/PNG transparency)
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                    
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="JPEG") # Consistency
                files = {"file": ("image.jpg", img_byte_arr.getvalue(), "image/jpeg")}
                data_form = {
                    "force_expert": str(force_expert),
                    "manual_min_val": manual_min if manual_min.strip() else "",
                    "manual_max_val": manual_max if manual_max.strip() else "",
                    "device_time_str": device_time_str
                }
                response = requests.post(f"{API_URL}/analyze", files=files, data=data_form)
                if response.status_code == 200: 
                    st.session_state.analysis_result = response.json()
                else: 
                    st.error(f"Server Error: {response.status_code}")
                    if "analysis_result" in st.session_state: del st.session_state.analysis_result
            except Exception as e: 
                st.error(f"Connection Failed: {e}")
                if "analysis_result" in st.session_state: del st.session_state.analysis_result

    # Display results if they exist in session state
    if st.session_state.get("analysis_result"):
        display_results(st.session_state.analysis_result)

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
                    stframe.image(frame_rgb, channels="RGB", width=260)
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
                    st.dataframe(pd.DataFrame(data["results"]), width=260)
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
            if not df_method.empty: st.plotly_chart(px.bar(df_method, x="Method", y="Count", color="Method"), width=260)
            else: st.info("No data yet.")
        with c2:
            st.markdown(f"#### {icon('memory')} Component Utilization", unsafe_allow_html=True)
            df_comp = pd.DataFrame(list(metrics["component_usage"].items()), columns=["Component", "Count"])
            if not df_comp.empty: st.plotly_chart(px.pie(df_comp, names="Component", values="Count", hole=0.4), width=260)
            else: st.info("No data yet.")
    except Exception as e: st.error(f"Dashboard Error: {e}")