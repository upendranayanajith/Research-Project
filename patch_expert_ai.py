"""
Patch: reorganise Expert AI tab (tab3) into 4 clean sub-tabs:
  📐 Angles     — angle visual, hand values, ResNet crops, hand type, uncertainty
  🔬 XAI        — GradCAM++/LIME/SHAP tabs, AI explanations, contrastive XAI
  📈 Temporal   — Kalman stability panel
  🔍 Debug      — full pipeline debug log
"""
import pathlib, re

fe = pathlib.Path(r"d:\Y4S1\Research 4\Clock_Time_Research\Research-Project\app\frontend.py")
content = fe.read_text(encoding="utf-8")

OLD = '''    with tab3:
        st.markdown(f"{icon('psychology')} **Angle Predictions**", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if "c3_angles" in viz: st.image(base64.b64decode(viz["c3_angles"]), caption="Angle Visual", width=300)
        with col_b:
            if "angles" in res and res["angles"]:
                if "span" in res["angles"]:
                    st.markdown(f"**Total Scale Span:** {res['angles'].get('span', 0):.1f}\\xb0")
                    st.markdown(f"**Needle Pos:** {res['angles'].get('needle', 0):.1f}\\xb0")
                    upd = res['angles'].get('units_per_deg', 0.0)
                    if upd > 0:
                        st.markdown(f"**1\\xb0 Angle =** {upd:.4f} scale units")
                else:
                    st.markdown(f"**H:** {res['angles'].get('hand1', 0):.1f}\\xb0")
                    st.markdown(f"**M:** {res['angles'].get('hand2', 0):.1f}\\xb0")
        if "c3_crops" in viz and viz["c3_crops"]:
            st.markdown("---")
            st.markdown(f"**{icon('image')} ResNet Inputs**", unsafe_allow_html=True)
            c_cols = st.columns(len(viz["c3_crops"]))
            for idx, (col, crop) in enumerate(zip(c_cols, viz["c3_crops"])):
                col.image(base64.b64decode(crop), width=100)
            if data.get("heatmap_b64"):
                xai_tabs = st.tabs(["\U0001f525 GradCAM++", "\U0001f7e2 LIME", "\U0001f534 SHAP"])
                with xai_tabs[0]:
                    st.image(base64.b64decode(data["heatmap_b64"]), caption="GradCAM++ (multi-layer fused)", width=300)
                with xai_tabs[1]:
                    if "lime_heatmap" in viz:
                        st.image(base64.b64decode(viz["lime_heatmap"]), caption="LIME: Superpixel perturbation (top-5 regions)", width=300)
                    else:
                        st.info("LIME heatmap not yet available. Run in Expert Path to generate.")
                with xai_tabs[2]:
                    if "shap_heatmap" in viz:
                        st.image(base64.b64decode(viz["shap_heatmap"]), caption="SHAP: DeepExplainer pixel attribution (JET colormap)", width=300)
                    else:
                        st.info("SHAP heatmap not yet available. Run in Expert Path to generate.")

            # --- AI Insight Explanations (Gemini or LocalExplainer via AdaptiveRouter) ---
            debug_lines = res.get("debug", [])
            insight_lines = [l for l in debug_lines if "AI Insight" in l]
            routing_lines = [l for l in debug_lines if "XAI Routing" in l]
            uncertainty_lines = [l for l in debug_lines if "uncertainty" in l.lower() or "alpha" in l.lower()]

            if insight_lines:
                st.markdown("---")
                st.markdown(f"**{icon('psychology')} AI Model Explanations**", unsafe_allow_html=True)
                for line in insight_lines:
                    label, _, text = line.partition(": ")
                    hand_label = label.replace("AI Insight ", "")
                    is_local  = "[Local XAI]" in text
                    is_gemini = "[Gemini]" in text
                    badge = "\U0001f535 Local" if is_local else "\u2728 Gemini" if is_gemini else "\u2139\ufe0f"
                    clean_text = text.replace("[Local XAI] ", "").replace("[Gemini] ", "")
                    st.info(f"**{badge} \u2014 {hand_label}:** {clean_text}")
                if routing_lines:
                    for rl in routing_lines:
                        escalated = "Gemini escalated" in rl
                        r_icon = "\u2728" if escalated else "\U0001f535"
                        st.caption(f"{r_icon} {rl}")
            else:
                st.markdown("---")
                st.info("\U0001f4a1 AI Explanation: Enable **Force Expert Path** and re-run.")

            # --- [6.6] Contrastive XAI: Why not X:XX? ---
            contrastive_xai = res.get("contrastive_xai")
            if contrastive_xai:
                st.markdown("---")
                st.markdown(f"**{icon('compare_arrows')} \U0001f914 Contrastive XAI: Why not...?**", unsafe_allow_html=True)
                for line in contrastive_xai.split("\\n"):
                    if not line.strip():
                        continue
                    if line.startswith("["):
                        st.markdown(line)
                    elif "consistent \u2705" in line:
                        st.success(line.strip())
                    elif "inconsistent \u274c" in line:
                        st.error(line.strip())
                    else:
                        st.markdown(line)

            # --- Uncertainty & Confidence ---
            if uncertainty_lines or res.get("uncertainty_deg"):
                st.markdown(f"**{icon('bar_chart')} C3 Uncertainty (MC Dropout)**", unsafe_allow_html=True)
                unc_val = res.get("uncertainty_deg", "N/A")
                if unc_val and unc_val != "N/A":
                    st.success(f"**MC Dropout Uncertainty:** {unc_val}")
                for line in uncertainty_lines:
                    st.caption(f"\U0001f522 {line}")

            # --- [6.7] Hand Type heuristic ---
            hand_type_lines = [l for l in debug_lines if "Hand Type Heuristic" in l]
            if hand_type_lines:
                st.caption(f"\U0001f91a {hand_type_lines[-1].replace(chr(72)+chr(97)+chr(110)+chr(100)+chr(32)+chr(84)+chr(121)+chr(112)+chr(101)+chr(32)+chr(72)+chr(101)+chr(117)+chr(114)+chr(105)+chr(115)+chr(116)+chr(105)+chr(99)+chr(58)+chr(32), chr(32))}")

            # --- Collapsible Debug Log ---
            with st.expander("\U0001f50d Full Pipeline Debug Log", expanded=False):
                for line in debug_lines:
                    is_ok   = any(k in line for k in ["Accepted","Gemini API","Manual","LIME:","SHAP:"])
                    is_warn = "Rejected" in line or "Failed" in line
                    is_gem  = "Routing" in line and "Gemini escalated" in line
                    is_info = any(k in line.lower() for k in ["alpha","uncertainty","temporal","routing","contrastive","hand type"])
                    icon_char = "\u2705" if is_ok else "\u26a0\ufe0f" if is_warn else "\U0001f7e2" if is_gem else "\U0001f535" if is_info else "\u25aa\ufe0f"
                    st.markdown(f"{icon_char} `{line}`")

            # --- [Tier 1.4] Temporal Stability Panel ---
            temporal_xai = res.get("temporal_xai")
            if temporal_xai:
                st.markdown("---")
                st.markdown(
                    f"**{icon('bar_chart')} \U0001f4c8 Temporal Stability (Kalman Filter)**",
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
                    t_cols[3].metric('Avg Correction', f"{temporal_xai.get('mean_kalman_correction_deg', 0):.1f}\\xb0")
                    st.caption(f"\U0001f522 {temporal_xai.get('message', '')}")
                    with st.expander("Variance Details"):
                        st.json({
                            "hand1_variance_deg": temporal_xai.get("hand1_variance_deg"),
                            "hand2_variance_deg": temporal_xai.get("hand2_variance_deg"),
                            "spike_rate_per_frame": temporal_xai.get("spike_rate_per_frame"),
                            "frames_seen": temporal_xai.get("frames_seen"),
                        })
            else:
                st.caption("\U0001f4c8 Temporal Stability: N/A (only active in Live Webcam / RTSP mode)")
        else:
            st.info("Expert AI skipped (Fast Path or Gauge Mode used). Enable 'Force Expert Path' to activate C3 + XAI.")'''

NEW = '''    with tab3:
        # ── helper: collect debug lines once ──────────────────────────────────
        debug_lines       = res.get("debug", [])
        insight_lines     = [l for l in debug_lines if "AI Insight" in l]
        routing_lines     = [l for l in debug_lines if "XAI Routing" in l]
        uncertainty_lines = [l for l in debug_lines if "uncertainty" in l.lower() or "alpha" in l.lower()]
        hand_type_lines   = [l for l in debug_lines if "Hand Type Heuristic" in l]
        temporal_xai      = res.get("temporal_xai")
        contrastive_xai   = res.get("contrastive_xai")
        has_c3            = "c3_crops" in viz and viz["c3_crops"]

        if has_c3 or insight_lines:
            c3_sub = st.tabs([
                "\U0001f4d0 Angles",
                "\U0001f52c XAI Heatmaps",
                "\U0001f4c8 Temporal",
                "\U0001f50d Debug Log",
            ])

            # ──────────────────────────────────────────────────────────────────
            # TAB 1: ANGLES
            # ──────────────────────────────────────────────────────────────────
            with c3_sub[0]:
                st.markdown(f"#### {icon('psychology')} Predicted Angles", unsafe_allow_html=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    if "c3_angles" in viz:
                        st.image(base64.b64decode(viz["c3_angles"]), caption="C3 angle overlay", use_container_width=True)
                with col_b:
                    if "angles" in res and res["angles"]:
                        if "span" in res["angles"]:
                            st.metric("Scale Span", f"{res['angles'].get('span', 0):.1f}\\xb0")
                            st.metric("Needle Pos", f"{res['angles'].get('needle', 0):.1f}\\xb0")
                            upd = res['angles'].get('units_per_deg', 0.0)
                            if upd > 0:
                                st.caption(f"1\\xb0 = {upd:.4f} scale units")
                        else:
                            st.metric("Hour hand",   f"{res['angles'].get('hand1', 0):.1f}\\xb0")
                            st.metric("Minute hand", f"{res['angles'].get('hand2', 0):.1f}\\xb0")

                # ResNet crops
                if has_c3:
                    st.markdown("---")
                    st.markdown(f"**{icon('image')} ResNet-18 Input Crops**", unsafe_allow_html=True)
                    c_cols = st.columns(min(len(viz["c3_crops"]), 4))
                    for col, crop in zip(c_cols, viz["c3_crops"]):
                        col.image(base64.b64decode(crop), use_container_width=True)

                # Uncertainty
                unc_val = res.get("uncertainty_deg", "")
                if unc_val and unc_val != "N/A":
                    st.markdown("---")
                    st.markdown(f"**{icon('bar_chart')} MC Dropout Uncertainty**", unsafe_allow_html=True)
                    st.success(f"{unc_val}")
                    for line in uncertainty_lines:
                        st.caption(f"\U0001f522 {line}")

                # Hand type heuristic
                if hand_type_lines:
                    st.markdown("---")
                    raw = hand_type_lines[-1]
                    display = raw.replace("Hand Type Heuristic: ", "")
                    st.markdown(f"**\U0001f91a Hand Type Detection**")
                    st.info(display)

            # ──────────────────────────────────────────────────────────────────
            # TAB 2: XAI HEATMAPS
            # ──────────────────────────────────────────────────────────────────
            with c3_sub[1]:
                # Heatmap method selector
                if data.get("heatmap_b64"):
                    st.markdown(f"#### {icon('opacity')} Attention / Attribution Maps", unsafe_allow_html=True)
                    heatmap_tabs = st.tabs(["\U0001f525 GradCAM++", "\U0001f7e2 LIME", "\U0001f534 SHAP"])
                    with heatmap_tabs[0]:
                        st.image(base64.b64decode(data["heatmap_b64"]),
                                 caption="GradCAM++ \u2014 weighted fusion of ResNet18 layer2+3+4",
                                 use_container_width=True)
                    with heatmap_tabs[1]:
                        if "lime_heatmap" in viz:
                            st.image(base64.b64decode(viz["lime_heatmap"]),
                                     caption="LIME \u2014 superpixel perturbation (top-5 regions highlighted)",
                                     use_container_width=True)
                        else:
                            st.info("\U0001f7e2 LIME heatmap is generated on the Expert Path. Re-run with Force Expert enabled.")
                    with heatmap_tabs[2]:
                        if "shap_heatmap" in viz:
                            st.image(base64.b64decode(viz["shap_heatmap"]),
                                     caption="SHAP DeepExplainer \u2014 pixel attribution (JET colormap: red = high)",
                                     use_container_width=True)
                        else:
                            st.info("\U0001f534 SHAP heatmap is generated on the Expert Path. Re-run with Force Expert enabled.")
                else:
                    st.info("Heatmaps are only generated in Expert Path mode. Enable Force Expert Path and re-run.")

                # AI explanations
                st.markdown("---")
                st.markdown(f"#### {icon('psychology')} AI Model Explanations", unsafe_allow_html=True)
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
                            escalated = "Gemini escalated" in rl
                            r_icon = "\u2728" if escalated else "\U0001f535"
                            st.caption(f"{r_icon} {rl}")
                else:
                    st.info("\U0001f4a1 Enable **Force Expert Path** and re-run to generate AI explanations.")

                # Contrastive XAI
                if contrastive_xai:
                    st.markdown("---")
                    st.markdown(f"#### {icon('compare_arrows')} \U0001f914 Contrastive XAI \u2014 Why not...?", unsafe_allow_html=True)
                    for line in contrastive_xai.split("\\n"):
                        if not line.strip():
                            continue
                        if line.startswith("["):
                            st.markdown(line)
                        elif "consistent \u2705" in line:
                            st.success(line.strip())
                        elif "inconsistent \u274c" in line:
                            st.error(line.strip())
                        else:
                            st.markdown(line)

            # ──────────────────────────────────────────────────────────────────
            # TAB 3: TEMPORAL STABILITY
            # ──────────────────────────────────────────────────────────────────
            with c3_sub[2]:
                st.markdown(f"#### {icon('bar_chart')} \U0001f4c8 Kalman Filter Temporal Stability", unsafe_allow_html=True)
                if temporal_xai:
                    t_status = temporal_xai.get("status", "N/A")
                    if t_status == "Initialising":
                        st.info(temporal_xai.get("message", "Kalman filter warming up \u2014 needs a few frames to stabilise."))
                    elif t_status == "Active":
                        t_cols = st.columns(4)
                        t_cols[0].metric("Stability",       f"{temporal_xai.get('stability_score', 'N/A')}%")
                        t_cols[1].metric("Trend",           temporal_xai.get('trend', 'N/A'))
                        t_cols[2].metric("Spikes Rejected", temporal_xai.get('total_spike_count', 0))
                        t_cols[3].metric("Avg Correction",  f"{temporal_xai.get('mean_kalman_correction_deg', 0):.1f}\\xb0")
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
                    st.info("\U0001f4c8 Temporal stability tracking is only active in **Live Webcam** or **RTSP** mode.")
                    st.caption("The Kalman filter accumulates predictions over time \u2014 it cannot run on single-image uploads.")

            # ──────────────────────────────────────────────────────────────────
            # TAB 4: DEBUG LOG
            # ──────────────────────────────────────────────────────────────────
            with c3_sub[3]:
                st.markdown(f"#### {icon('bug_report')} Full Pipeline Debug Log", unsafe_allow_html=True)
                st.caption(f"{len(debug_lines)} pipeline steps recorded for this analysis.")
                for line in debug_lines:
                    is_ok   = any(k in line for k in ["Accepted","Gemini API","Manual","LIME:","SHAP:"])
                    is_warn = "Rejected" in line or "Failed" in line
                    is_gem  = "Routing" in line and "Gemini escalated" in line
                    is_info = any(k in line.lower() for k in ["alpha","uncertainty","temporal","routing","contrastive","hand type"])
                    icon_char = "\u2705" if is_ok else "\u26a0\ufe0f" if is_warn else "\U0001f7e2" if is_gem else "\U0001f535" if is_info else "\u25aa\ufe0f"
                    st.markdown(f"{icon_char} `{line}`")
        else:
            st.info("\U0001f4a1 Expert AI is only active when **Force Expert Path** is enabled or the physics error > 20\\xb0.")
            st.caption("Fast Path and Gauge modes skip C3 \u2014 enable Force Expert Path in the sidebar and re-run.")'''

if OLD in content:
    content = content.replace(OLD, NEW, 1)
    fe.write_text(content, encoding="utf-8")
    print("PATCH OK")
else:
    print("OLD string not found — looking for key anchor...")
    idx = content.find('with tab3:')
    print(f"'with tab3:' at char {idx}: {repr(content[idx:idx+60])}")

import ast
try:
    ast.parse(content)
    print("Syntax: OK")
except SyntaxError as e:
    print(f"Syntax ERROR: {e}")
