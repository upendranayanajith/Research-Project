"""
Tier 2 frontend.py patch:
- Replace single GradCAM++ image with 3-tab selector (GradCAM++ / LIME / SHAP)
- Add routing_lines extraction alongside insight_lines
- Add Contrastive XAI panel
- Add adaptive routing caption
- Add Hand Type heuristic caption
- Update debug log icon set
"""
import pathlib, re

fe = pathlib.Path(r"d:\Y4S1\Research 4\Clock_Time_Research\Research-Project\app\frontend.py")
content = fe.read_text(encoding="utf-8")

# ======================== PATCH 1: heatmap section -> 3-tab XAI selector ========================
OLD1 = (
    '            if data.get("heatmap_b64"):\n'
    '                st.markdown(f"**{icon(\'opacity\')} Attention Map (GradCAM++)**", unsafe_allow_html=True)\n'
    '                st.image(base64.b64decode(data["heatmap_b64"]), width=300)\n'
    '\n'
    '            # --- AI Insight Explanations (Gemini or LocalExplainer) ---\n'
    '            debug_lines = res.get("debug", [])\n'
    '            insight_lines = [l for l in debug_lines if "AI Insight" in l]\n'
    '            uncertainty_lines = [l for l in debug_lines if "uncertainty" in l.lower() or "alpha" in l.lower()]\n'
)

NEW1 = (
    '            if data.get("heatmap_b64"):\n'
    '                xai_tabs = st.tabs(["\U0001f525 GradCAM++", "\U0001f7e2 LIME", "\U0001f534 SHAP"])\n'
    '                with xai_tabs[0]:\n'
    '                    st.image(base64.b64decode(data["heatmap_b64"]), caption="GradCAM++ (multi-layer fused)", width=300)\n'
    '                with xai_tabs[1]:\n'
    '                    if "lime_heatmap" in viz:\n'
    '                        st.image(base64.b64decode(viz["lime_heatmap"]), caption="LIME: Superpixel perturbation (top-5 regions)", width=300)\n'
    '                    else:\n'
    '                        st.info("LIME heatmap not yet available. Run in Expert Path to generate.")\n'
    '                with xai_tabs[2]:\n'
    '                    if "shap_heatmap" in viz:\n'
    '                        st.image(base64.b64decode(viz["shap_heatmap"]), caption="SHAP: DeepExplainer pixel attribution (JET colormap)", width=300)\n'
    '                    else:\n'
    '                        st.info("SHAP heatmap not yet available. Run in Expert Path to generate.")\n'
    '\n'
    '            # --- AI Insight Explanations (Gemini or LocalExplainer via AdaptiveRouter) ---\n'
    '            debug_lines = res.get("debug", [])\n'
    '            insight_lines = [l for l in debug_lines if "AI Insight" in l]\n'
    '            routing_lines = [l for l in debug_lines if "XAI Routing" in l]\n'
    '            uncertainty_lines = [l for l in debug_lines if "uncertainty" in l.lower() or "alpha" in l.lower()]\n'
)

if OLD1 in content:
    content = content.replace(OLD1, NEW1, 1)
    print("PATCH 1 OK: 3-tab XAI selector")
else:
    print("PATCH 1 NOT FOUND")

# ======================== PATCH 2: insight_lines section -> add routing captions + Contrastive XAI ========================
OLD2 = (
    '            if insight_lines:\n'
    '                st.markdown("---")\n'
    '                st.markdown(f"**{icon(\'psychology\')} AI Model Explanations**", unsafe_allow_html=True)\n'
    '                for line in insight_lines:\n'
    '                    # Strip the "AI Insight Hand X: " prefix for clean display\n'
    '                    label, _, text = line.partition(": ")\n'
    '                    hand_label = label.replace("AI Insight ", "")\n'
    '                    is_local = "[Local XAI]" in text\n'
    '                    is_gemini = "[Gemini]" in text\n'
    '                    badge = "\U0001f535 Local" if is_local else "\u2728 Gemini" if is_gemini else "\u2139\ufe0f"\n'
    '                    clean_text = text.replace("[Local XAI] ", "").replace("[Gemini] ", "")\n'
    '                    st.info(f"**{badge} \u2014 {hand_label}:** {clean_text}")\n'
    '            else:\n'
    '                st.markdown("---")\n'
    '                st.info("\U0001f4a1 AI Explanation: Enable **Force Expert Path** and re-run to generate model explanations.")\n'
)

NEW2 = (
    '            if insight_lines:\n'
    '                st.markdown("---")\n'
    '                st.markdown(f"**{icon(\'psychology\')} AI Model Explanations**", unsafe_allow_html=True)\n'
    '                for line in insight_lines:\n'
    '                    label, _, text = line.partition(": ")\n'
    '                    hand_label = label.replace("AI Insight ", "")\n'
    '                    is_local  = "[Local XAI]" in text\n'
    '                    is_gemini = "[Gemini]" in text\n'
    '                    badge = "\U0001f535 Local" if is_local else "\u2728 Gemini" if is_gemini else "\u2139\ufe0f"\n'
    '                    clean_text = text.replace("[Local XAI] ", "").replace("[Gemini] ", "")\n'
    '                    st.info(f"**{badge} \u2014 {hand_label}:** {clean_text}")\n'
    '                if routing_lines:\n'
    '                    for rl in routing_lines:\n'
    '                        escalated = "Gemini escalated" in rl\n'
    '                        r_icon = "\u2728" if escalated else "\U0001f535"\n'
    '                        st.caption(f"{r_icon} {rl}")\n'
    '            else:\n'
    '                st.markdown("---")\n'
    '                st.info("\U0001f4a1 AI Explanation: Enable **Force Expert Path** and re-run.")\n'
    '\n'
    '            # --- [6.6] Contrastive XAI: Why not X:XX? ---\n'
    '            contrastive_xai = res.get("contrastive_xai")\n'
    '            if contrastive_xai:\n'
    '                st.markdown("---")\n'
    '                st.markdown(f"**{icon(\'compare_arrows\')} \U0001f914 Contrastive XAI: Why not...?**", unsafe_allow_html=True)\n'
    '                for line in contrastive_xai.split("\\n"):\n'
    '                    if not line.strip():\n'
    '                        continue\n'
    '                    if line.startswith("["):\n'
    '                        st.markdown(line)\n'
    '                    elif "consistent \u2705" in line:\n'
    '                        st.success(line.strip())\n'
    '                    elif "inconsistent \u274c" in line:\n'
    '                        st.error(line.strip())\n'
    '                    else:\n'
    '                        st.markdown(line)\n'
)

if OLD2 in content:
    content = content.replace(OLD2, NEW2, 1)
    print("PATCH 2 OK: routing captions + Contrastive XAI panel")
else:
    print("PATCH 2 NOT FOUND — checking badges...")
    idx = content.find('badge = "')
    if idx >= 0:
        print(repr(content[idx:idx+80]))
    else:
        print("badge not found")

# ======================== PATCH 3: Uncertainty section -> add hand type + update debug icons ========================
OLD3 = (
    '            # --- Uncertainty & Confidence ---\n'
    '            if uncertainty_lines or res.get("uncertainty_deg"):\n'
    '                st.markdown(f"**{icon(\'bar_chart\')} C3 Uncertainty**", unsafe_allow_html=True)\n'
    '                unc_val = res.get("uncertainty_deg", "N/A")\n'
    '                if unc_val and unc_val != "N/A":\n'
    '                    st.success(f"**MC Dropout Uncertainty:** {unc_val}")\n'
    '                for line in uncertainty_lines:\n'
    '                    st.caption(f"\U0001f522 {line}")\n'
    '\n'
    '            # --- Collapsible Debug Log ---\n'
    '            with st.expander("\U0001f50d Full Pipeline Debug Log", expanded=False):\n'
    '                for line in debug_lines:\n'
    '                    icon_char = "\u2705" if "Accepted" in line or "Gemini API" in line or "Manual" in line else \\\n'
    '                                "\u26a0\ufe0f" if "Rejected" in line or "Failed" in line else \\\n'
    '                                "\U0001f535" if "alpha" in line.lower() or "uncertainty" in line.lower() or "temporal" in line.lower() else "\u25aa\ufe0f"\n'
    '                    st.markdown(f"{icon_char} `{line}`")\n'
)

NEW3 = (
    '            # --- Uncertainty & Confidence ---\n'
    '            if uncertainty_lines or res.get("uncertainty_deg"):\n'
    '                st.markdown(f"**{icon(\'bar_chart\')} C3 Uncertainty (MC Dropout)**", unsafe_allow_html=True)\n'
    '                unc_val = res.get("uncertainty_deg", "N/A")\n'
    '                if unc_val and unc_val != "N/A":\n'
    '                    st.success(f"**MC Dropout Uncertainty:** {unc_val}")\n'
    '                for line in uncertainty_lines:\n'
    '                    st.caption(f"\U0001f522 {line}")\n'
    '\n'
    '            # --- [6.7] Hand Type heuristic ---\n'
    '            hand_type_lines = [l for l in debug_lines if "Hand Type Heuristic" in l]\n'
    '            if hand_type_lines:\n'
    '                st.caption(f"\U0001f91a {hand_type_lines[-1].replace(\'Hand Type Heuristic: \', \'\')}")\n'
    '\n'
    '            # --- Collapsible Debug Log ---\n'
    '            with st.expander("\U0001f50d Full Pipeline Debug Log", expanded=False):\n'
    '                for line in debug_lines:\n'
    '                    is_ok   = any(k in line for k in ["Accepted","Gemini API","Manual","LIME:","SHAP:"])\n'
    '                    is_warn = "Rejected" in line or "Failed" in line\n'
    '                    is_gem  = "Routing" in line and "Gemini escalated" in line\n'
    '                    is_info = any(k in line.lower() for k in ["alpha","uncertainty","temporal","routing","contrastive","hand type"])\n'
    '                    icon_char = "\u2705" if is_ok else "\u26a0\ufe0f" if is_warn else "\U0001f7e2" if is_gem else "\U0001f535" if is_info else "\u25aa\ufe0f"\n'
    '                    st.markdown(f"{icon_char} `{line}`")\n'
)

if OLD3 in content:
    content = content.replace(OLD3, NEW3, 1)
    print("PATCH 3 OK: Hand type heuristic + updated debug icons")
else:
    print("PATCH 3 NOT FOUND")
    idx = content.find("C3 Uncertainty")
    print(repr(content[max(0,idx-30):idx+60]))

fe.write_text(content, encoding="utf-8")
print("Done writing frontend.py")

# Final syntax check
import ast
try:
    ast.parse(content)
    print("Syntax: OK")
except SyntaxError as e:
    print(f"Syntax ERROR: {e}")
