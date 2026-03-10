import streamlit as st  # type: ignore
import requests  # type: ignore
import base64
from PIL import Image  # type: ignore
import io
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase  # type: ignore
import av  # type: ignore
import cv2  # type: ignore
import numpy as np  # type: ignore
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
        
        from app.core.engine import HARPEngine  # type: ignore
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
                    manual_max_val=self.manual_max_val
                )
            except Exception as e:
                print(f"AI Error: {e}")

        # Draw overlays
        if self.last_result and isinstance(self.last_result, dict):
            res: dict = self.last_result  # type: ignore[assignment]
            
            # Show Time or Gauge %
            display_val = res.get('time', '--')
            cv2.putText(img, f"READING: {display_val}", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
            
            method = res.get('method', 'Unknown')
            color = (0, 255, 0) if "Fast" in method or "Gauge" in method else (0, 0, 255)
            cv2.putText(img, f"Mode: {method}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Only show angles if it's a clock
            angles = res.get("angles") or {}
            if angles.get("hand1", 0) != 0.0:
                a1 = angles.get("hand1", 0)
                a2 = angles.get("hand2", 0)
                cv2.putText(img, f"H:{a1:.0f} M:{a2:.0f}", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(img, f"FPS: {self.fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# [Shared] HELPER FUNCTIONS
# ==========================================
@st.dialog("Cognitive Reasoning Report", width="large")
def show_reasoning_report(data):
    res = data["result"]
    
    st.markdown("### 📋 Official Diagnostic Report")
    st.caption(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### 📝 Executive Summary")
        st.write(f"**Final Reading:** :green[{res.get('time', 'N/A')}]")
        st.write(f"**Detection Method:** {res.get('method', 'N/A')}")
        
        c4_conf = res.get("c4_confidence")
        if c4_conf:
            tier_colors = {"CERTAIN": "green", "CONFIDENT": "blue", "UNCERTAIN": "orange", "UNRELIABLE": "red"}
            tier = c4_conf['tier']
            color = tier_colors.get(tier, "grey")
            st.markdown(f"**Confidence:** :{color}[{tier}] `{c4_conf['score']}/100`")
        else:
            st.write(f"**Confidence:** {res.get('confidence', 'N/A')}")
            
        st.write(f"**AM/PM Inference:** {res.get('ampm', 'N/A')}")
    
    with col2:
        st.markdown("#### ✅ Pipeline Status")
        stages = ["C1 Detection", "C2 Keypoints", "C3 Expert AI", "C4 Physics"]
        for s in stages:
            is_done = True if "Expert" in res["method"] else (s != "C3 Expert AI")
            icon_lbl = "✅" if is_done else "❌"
            st.write(f"{icon_lbl} {s}")

    st.divider()
    st.markdown("#### 🧠 AI Logical Trace")
    if "debug" in res:
        for trace in res["debug"]:
            if "C4 Telemetry Trace" in trace:
                st.info(trace)
            elif "Heuristics" in trace:
                st.success(trace)
            elif "C4 Confidence" in trace:
                st.warning(trace)
            else:
                st.write(f"▸ {trace}")
    else:
        st.warning("No trace logs available for this session.")

    st.divider()
    st.markdown("#### 📊 Physics Validation & Research Extension")
    
    c4_conf = res.get("c4_confidence")
    if c4_conf:
        tier = c4_conf['tier']
        tier_map = {"CERTAIN": "🟢", "CONFIDENT": "🔵", "UNCERTAIN": "🟠", "UNRELIABLE": "🔴"}
        st.info(f"{tier_map.get(tier, '⚪')} **C4+ Research Insight:** {c4_conf['reason']}")
        cA, cB, cC = st.columns(3)
        cA.metric("Confidence Score", f"{c4_conf['score']}%")
        cB.metric("Angular Gap", f"{c4_conf['angular_gap']}°")
        cC.metric("Tier", tier)
    else:
        st.warning("C4+ Confidence extension not available for this session.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Angular Shift (H):** {res.get('angles', {}).get('hand1', 0):.1f}°")
        st.write(f"**Angular Shift (M):** {res.get('angles', {}).get('hand2', 0):.1f}°")
    with col_b:
        st.write(f"**Ambiguity:** {res.get('ambiguity', 'None detected')}")
        st.write(f"**Accuracy:** {res.get('drift', 'N/A')}")

    st.divider()
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
            else:
                st.info("Impact summary not available.")
    with tab3:
        # ── Top: XAI method badge ──────────────────────────────────────────────
        xai_method = res.get("xai_method", "")
        if xai_method:
            st.markdown(
                f"<span style='background:#a78bfa22;color:#a78bfa;padding:3px 10px;"
                f"border-radius:4px;font-size:11px;font-weight:700;border:1px solid #a78bfa44;'>"
                f"🔬 {xai_method}</span>",
                unsafe_allow_html=True
            )

        c3_has_data = "c3_crops" in viz and viz["c3_crops"]

        c3s1, c3s2, c3s3, c3s4, c3s5 = st.tabs([
            "📐 Angle Regression",
            "🔬 XAI Heatmaps",
            "🧠 AI Explanations",
            "📊 Uncertainty",
            "🔍 Pipeline Debug",
        ])

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB 1 — Angle Regression
        # ══════════════════════════════════════════════════════════════════════
        with c3s1:
            col_a, col_b = st.columns(2)
            with col_a:
                if "c3_angles" in viz:
                    st.image(base64.b64decode(viz["c3_angles"]), caption="Angle Visual", width=300)
            with col_b:
                if "angles" in res and res["angles"]:
                    if "span" in res["angles"]:
                        st.metric("Scale Span", f"{res['angles'].get('span', 0):.1f}°")
                        st.metric("Needle Pos",  f"{res['angles'].get('needle', 0):.1f}°")
                        upd = res["angles"].get("units_per_deg", 0.0)
                        if upd > 0:
                            st.markdown(f"**1° =** `{upd:.4f}` scale units")
                    else:
                        h_ang = res["angles"].get("hand1", 0)
                        m_ang = res["angles"].get("hand2", 0)
                        st.metric("Hour Hand",   f"{h_ang:.1f}°")
                        st.metric("Minute Hand", f"{m_ang:.1f}°")

            if c3_has_data:
                st.markdown("---")
                st.markdown(
                    f"**{icon('image')} ResNet-18 Input Crops (MC Dropout active)**",
                    unsafe_allow_html=True
                )
                c_cols = st.columns(max(len(viz["c3_crops"]), 1))
                for idx, (col, crop) in enumerate(zip(c_cols, viz["c3_crops"])):
                    col.image(base64.b64decode(crop), caption=f"Hand {idx+1} crop (64×64)", width=120)

            if not c3_has_data:
                st.info("Expert AI skipped (Fast Path or Gauge Mode). Enable 'Force Expert Path' to activate.")

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB 2 — XAI Heatmaps
        # ══════════════════════════════════════════════════════════════════════
        with c3s2:
            if not c3_has_data:
                st.info("No XAI heatmaps — run with Force Expert Path.")
            else:
                xai_t1, xai_t2, xai_t3, xai_t4 = st.tabs(
                    ["🔥 GradCAM++", "🧮 Integrated Gradients", "🟩 LIME", "🔴 SHAP"]
                )

                with xai_t1:
                    if data.get("heatmap_b64"):
                        st.image(base64.b64decode(data["heatmap_b64"]),
                                 caption="GradCAM++ — fused L2×0.20 + L3×0.30 + L4×0.50 + quadrant annotation",
                                 width=400)
                        # AFS badges
                        afs_list = res.get("afs_scores", [])
                        if afs_list:
                            st.markdown("**Attribution Fidelity Score (ROAR — top-20% pixels blanked)**")
                            afs_cols = st.columns(len(afs_list))
                            for ac, afs in zip(afs_cols, afs_list):
                                afs_val   = afs.get("afs", 0.0)
                                afs_pct   = int(afs_val * 100)
                                afs_color = "#22c55e" if afs_val > 0.5 else "#f59e0b" if afs_val > 0.25 else "#ef4444"
                                causal_lbl = "Causal ✅" if afs_val > 0.5 else "Weak ⚠️" if afs_val > 0.25 else "Not causal ❌"
                                ac.markdown(
                                    f"<div style='text-align:center;padding:8px;background:#0f0f1a;"
                                    f"border:1px solid {afs_color}44;border-radius:8px;'>"
                                    f"<div style='color:#64748b;font-size:10px;'>maskedΔ={afs.get('delta_deg',0):.1f}°</div>"
                                    f"<div style='color:{afs_color};font-size:22px;font-weight:800;'>AFS {afs_pct}%</div>"
                                    f"<div style='color:#475569;font-size:10px;'>{causal_lbl}</div>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                    else:
                        st.info("GradCAM++ heatmap not available.")

                with xai_t2:
                    ig_found = False
                    for hi in range(1, 3):
                        key = f"xai_ig_h{hi}"
                        if key in viz and viz[key]:
                            st.image(base64.b64decode(viz[key]),
                                     caption=f"Integrated Gradients — Hand {hi} (HOT colourmap, 50 steps)",
                                     width=300)
                            ig_found = True
                    if not ig_found:
                        st.info("IG overlay requires **Force Expert Path**.")
                    st.caption(
                        "IG satisfies Completeness • Sensitivity • Implementation Invariance "
                        "(Sundararajan et al., ICML 2017) — theoretically grounded for regression."
                    )

                with xai_t3:
                    lime_found = False
                    for hi in range(1, 3):
                        key = f"xai_lime_h{hi}"
                        if key in viz and viz[key]:
                            st.image(base64.b64decode(viz[key]),
                                     caption=f"LIME — Hand {hi} (top-5 superpixels, 200 perturbations)", width=300)
                            lime_found = True
                    if not lime_found:
                        st.info("LIME requires `pip install lime` + Force Expert Path.")

                with xai_t4:
                    shap_found = False
                    for hi in range(1, 3):
                        key = f"xai_shap_h{hi}"
                        if key in viz and viz[key]:
                            st.image(base64.b64decode(viz[key]),
                                     caption=f"SHAP — Hand {hi} attribution (red=high positive influence)", width=300)
                            shap_found = True
                    if not shap_found:
                        st.info("SHAP requires `pip install shap` + Force Expert Path.")

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB 3 — AI Explanations
        # ══════════════════════════════════════════════════════════════════════
        with c3s3:
            # — Local / Gemini cards —
            xai_exps = res.get("xai_explanations", [])
            if xai_exps:
                st.markdown(f"#### {icon('psychology')} AI Explanation per Hand", unsafe_allow_html=True)
                for exp in xai_exps:
                    h_num   = exp.get("hand", "?")
                    source  = exp.get("source", "Local")
                    text    = exp.get("explanation", "")
                    entropy = exp.get("entropy", 0.0)
                    routing = exp.get("routing_reason", "")

                    if source == "Gemini":
                        bc, bi, bl = "#a78bfa", "✨", "Gemini Vision"
                    else:
                        bc, bi, bl = "#38bdf8", "🔵", "Local Heuristic"

                    st.markdown(f"""
                    <div style="background:#0f0f1a;border:1px solid {bc}33;
                                border-radius:10px;padding:14px 16px;margin-bottom:10px;">
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                            <span style="background:{bc}22;color:{bc};padding:2px 10px;
                                         border-radius:4px;font-size:11px;font-weight:700;
                                         border:1px solid {bc}44;">{bi} {bl}</span>
                            <span style="color:#94a3b8;font-size:12px;">Hand {h_num}</span>
                            <span style="color:#64748b;font-size:11px;margin-left:auto;">entropy={entropy:.3f}</span>
                        </div>
                        <div style="color:#e2e8f0;font-size:13px;line-height:1.5;">{text}</div>
                        <div style="color:#475569;font-size:10px;margin-top:6px;">Routing: {routing}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No AI explanations yet — run analysis to populate.")

            # — Contrastive XAI —
            contrastive = res.get("contrastive_xai", "")
            if contrastive:
                st.markdown("---")
                st.markdown(f"#### {icon('compare_arrows')} Contrastive XAI — Why this time?",
                            unsafe_allow_html=True)
                lines = contrastive.split("\n")
                if lines:
                    st.markdown(f"**{lines[0]}**")
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    color = "#22c55e" if "✅" in line else "#ef4444" if "❌" in line else "#94a3b8"
                    st.markdown(
                        f"<div style='padding:4px 10px;margin:2px 0;border-left:3px solid {color};"
                        f"background:{color}11;border-radius:0 4px 4px 0;font-size:13px;color:#e2e8f0;'>"
                        f"{line}</div>",
                        unsafe_allow_html=True
                    )

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB 4 — Uncertainty & Calibration
        # ══════════════════════════════════════════════════════════════════════
        with c3s4:
            # Structured per-hand cards
            per_hand = res.get("per_hand_xai", [])
            if per_hand:
                st.markdown(f"#### {icon('analytics')} MC Dropout — Per-Hand Uncertainty (20 passes)",
                            unsafe_allow_html=True)
                cols = st.columns(len(per_hand))
                for col, h in zip(cols, per_hand):
                    sig   = h.get("uncertainty_std", 0.0)
                    alpha = h.get("alpha", 1.0)
                    delta = h.get("delta", 0.0)
                    c3a   = h.get("c3_angle", 0.0)
                    raw_a = h.get("rough_angle", 0.0)
                    ent   = h.get("entropy", 0.0)
                    u_raw = h.get("uncertainty_raw", sig)
                    temp  = h.get("temperature", 1.0)

                    sc = "#22c55e" if sig < 5.0 else "#f59e0b" if sig < 15.0 else "#ef4444"
                    sg = "High Confidence" if sig < 5.0 else "Moderate" if sig < 15.0 else "Low — C2 preferred"

                    col.markdown(f"""
                    <div style="background:#0f0f1a;border:1px solid {sc}44;
                                border-radius:10px;padding:12px;text-align:center;">
                        <div style="color:#94a3b8;font-size:10px;">Hand {h.get('hand','?')}</div>
                        <div style="color:{sc};font-size:28px;font-weight:800;margin:4px 0;">
                            ±{sig:.1f}°
                        </div>
                        <div style="color:#64748b;font-size:10px;">{sg}</div>
                        <div style="margin:8px 0;background:#1e293b;border-radius:3px;height:5px;">
                            <div style="width:{int(alpha*100)}%;height:5px;
                                        background:{sc};border-radius:3px;"></div>
                        </div>
                        <div style="color:#64748b;font-size:10px;">α = {alpha:.2f}</div>
                        <div style="color:#475569;font-size:10px;margin-top:4px;">
                            C2={raw_a:.1f}° → C3={c3a:.1f}° (δ={delta:+.1f}°)
                        </div>
                        <div style="color:#334155;font-size:10px;">entropy={ent:.3f}</div>
                        <div style="color:#1e3a5f;font-size:9px;margin-top:3px;">
                            raw σ={u_raw:.1f}° × T={temp:.2f} → {sig:.1f}°
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Legacy summary string (always show if present)
            unc_str = res.get("uncertainty_deg", "")
            if unc_str and unc_str != "N/A":
                st.markdown("---")
                st.caption(f"**Summary:** {unc_str}")
                unc_parts = [p.strip() for p in unc_str.split(",") if p.strip()]
                unc_cols  = st.columns(len(unc_parts)) if unc_parts else []
                for col, part in zip(unc_cols, unc_parts):
                    try:
                        label, val_str = part.split("=")
                        sigma = float(val_str.replace("±", "").replace("°", "").strip())
                        alpha = max(0.0, min(1.0, 1.0 - sigma / 20.0))
                        color = "#22c55e" if sigma < 5.0 else "#f59e0b" if sigma < 15.0 else "#ef4444"
                        grade = "High" if sigma < 5.0 else "Moderate" if sigma < 15.0 else "Low"
                        col.markdown(
                            f"<div style='background:#0f0f1a;border:1px solid {color}44;"
                            f"border-radius:8px;padding:10px;text-align:center;'>"
                            f"<div style='color:#64748b;font-size:10px;'>{label.strip()}</div>"
                            f"<div style='color:{color};font-size:22px;font-weight:800;'>±{sigma:.1f}°</div>"
                            f"<div style='color:#475569;font-size:10px;'>{grade} · α={alpha:.2f}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    except Exception:
                        col.code(part)

            with st.expander("ℹ️ How MC Dropout + Kalman works"):
                st.markdown("""
                **MC Dropout** keeps `Dropout(0.3)` stochastic during inference.
                **20 forward passes** → circular σ (uncertainty):
                - σ < 5° → **High Confidence** — C3 dominates (α ≈ 1.0)
                - σ 5–15° → **Moderate** — blended C2 + C3
                - σ > 15° → **Low** — C2 geometry preferred (α → 0)

                **Temperature scaling:** `σ_cal = σ_raw × T`  (T=1.0 = uncalibrated)

                **C3 Kalman Smoother** (new): after blending, each hand's output is
                passed through a per-hand circular Kalman filter `[angle, velocity]`
                to eliminate frame-to-frame spikes during live video mode.
                """)

            if not per_hand and not unc_str:
                st.info("No uncertainty data — run with Expert Path active.")

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB 5 — Pipeline Debug
        # ══════════════════════════════════════════════════════════════════════
        with c3s5:
            debug_lines = res.get("debug", [])
            if debug_lines:
                def _icon_for(line: str) -> str:
                    l = line.lower()
                    if any(k in l for k in ["error", "failed", "rejected", "❌"]):
                        return "🔴"
                    if any(k in l for k in ["warning", "⚠", "slow", "ambiguity"]):
                        return "⚠️"
                    if any(k in l for k in ["xai", "gemini", "local xai", "entropy", "roar", "ig"]):
                        return "🔵"
                    if any(k in l for k in ["kalman", "smooth"]):
                        return "🟣"
                    if any(k in l for k in ["accepted", "✅", "ok", "success", "passed"]):
                        return "✅"
                    if any(k in l for k in ["c1:", "c2", "c3", "c4", "physics"]):
                        return "🟢"
                    return "▪️"

                # Group filter
                filter_opts = ["All", "🔴 Errors", "🔵 XAI", "🟢 Pipeline", "⚠️ Warnings", "🟣 Kalman"]
                f_col, _ = st.columns([1, 3])
                chosen = f_col.selectbox("Filter", filter_opts, label_visibility="collapsed")

                filtered = debug_lines
                if chosen == "🔴 Errors":
                    filtered = [l for l in debug_lines if any(k in l.lower() for k in ["error", "failed", "❌"])]
                elif chosen == "🔵 XAI":
                    filtered = [l for l in debug_lines if any(k in l.lower() for k in ["xai", "gemini", "entropy", "roar", "ig"])]
                elif chosen == "🟢 Pipeline":
                    filtered = [l for l in debug_lines if any(k in l.lower() for k in ["c1:", "c2", "c3", "c4", "physics"])]
                elif chosen == "⚠️ Warnings":
                    filtered = [l for l in debug_lines if any(k in l.lower() for k in ["warning", "⚠", "ambiguity"])]
                elif chosen == "🟣 Kalman":
                    filtered = [l for l in debug_lines if any(k in l.lower() for k in ["kalman", "smooth"])]

                st.caption(f"Showing {len(filtered)} / {len(debug_lines)} steps")
                for step_i, line in enumerate(filtered, 1):
                    badge = _icon_for(line)
                    st.markdown(
                        f"<div style='font-family:monospace;font-size:11px;padding:2px 8px;"
                        f"border-radius:3px;margin:1px 0;color:#94a3b8;background:#0a0a14;'>"
                        f"<span style='color:#475569;'>[{step_i:02d}]</span> {badge} {line}</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.info("No debug log — run with Expert Path to see pipeline steps.")

        # (All C3 features are now in the 5 sub-tabs above)




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
                    st.dataframe(cdf, width="stretch", hide_index=True)
            else:
                st.info("No ambiguity detected.")

        st.markdown("---")
        if st.button("📄 Generate Cognitive Reasoning Report", width="stretch"):
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
        if st.button(label, key=f"nav_{page_key}", type=btn_type, width="stretch"):
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
    if "analysis_result" in st.session_state and st.session_state.analysis_result:
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
                    from app.core.engine import HARPEngine  # type: ignore
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
        if cam_source == "Local Webcam" and expert_controls_ctx and expert_controls_ctx.video_processor:  # type: ignore[union-attr]
            expert_controls_ctx.video_processor.force_expert = st.checkbox("Enable C3/XAI", value=False, key="webcam_expert")  # type: ignore[union-attr]
        elif cam_source == "IP Camera (RTSP)":
            expert_enabled = st.checkbox("Enable C3/XAI", value=st.session_state.get("ip_cam_expert", False), key="ip_expert")
            st.session_state.ip_cam_expert = expert_enabled
            
        st.markdown("---")
        
        # Manual Scale Overrides
        st.markdown(f"#### {icon('edit')} Manual Gauge Scale", unsafe_allow_html=True)
        colA, colB = st.columns(2)
        
        if cam_source == "Local Webcam" and expert_controls_ctx and expert_controls_ctx.video_processor:  # type: ignore[union-attr]
            manual_min = colA.text_input("Min Value", "", key="webcam_min")
            manual_max = colB.text_input("Max Value", "", key="webcam_max")
            expert_controls_ctx.video_processor.manual_min_val = manual_min if manual_min.strip() else ""  # type: ignore[union-attr]
            expert_controls_ctx.video_processor.manual_max_val = manual_max if manual_max.strip() else ""  # type: ignore[union-attr]
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
                            manual_max_val=st.session_state.ip_cam_manual_max
                        )
                    except Exception as e:
                        print(f"AI Error: {e}")
                        
                # 2. Draw Overlays (We draw instantly on the copied frame before rendering)
                display_frame = frame.copy()
                if last_result and isinstance(last_result, dict):
                    res: dict = last_result  # type: ignore[assignment]
                    display_val = res.get('time', '--')
                    cv2.putText(display_frame, f"READING: {display_val}", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
                    
                    method = res.get('method', 'Unknown')
                    color = (0, 255, 0) if "Fast" in method or "Gauge" in method else (0, 0, 255)
                    cv2.putText(display_frame, f"Mode: {method}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    angles = res.get("angles") or {}
                    if angles.get("hand1", 0) != 0.0:
                        a1 = angles.get("hand1", 0)
                        a2 = angles.get("hand2", 0)
                        cv2.putText(display_frame, f"H:{a1:.0f} M:{a2:.0f}", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.putText(display_frame, f"Pipeline FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # 3. Streamlit UI Update
                if now - last_ui_update_time >= UI_UPDATE_INTERVAL:
                    # Convert BGR to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB", width="stretch")
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
                    st.dataframe(pd.DataFrame(data["results"]), width="stretch")
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
            if not df_method.empty: st.plotly_chart(px.bar(df_method, x="Method", y="Count", color="Method"), use_container_width=True)  # plotly still uses this
            else: st.info("No data yet.")
        with c2:
            st.markdown(f"#### {icon('memory')} Component Utilization", unsafe_allow_html=True)
            df_comp = pd.DataFrame(list(metrics["component_usage"].items()), columns=["Component", "Count"])
            if not df_comp.empty: st.plotly_chart(px.pie(df_comp, names="Component", values="Count", hole=0.4), use_container_width=True)  # plotly still uses this
            else: st.info("No data yet.")
    except Exception as e: st.error(f"Dashboard Error: {e}")