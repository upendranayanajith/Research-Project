import streamlit as st
import base64
from collections import deque, Counter

def render_c1_localization(viz, res):
    col_img, col_gap, col_val = st.columns([1, 0.08, 3])

    # --- Left: Localization Visual ---
    with col_img:
        st.markdown(
            "<p style='margin:0 0 6px 0; font-size:11px; font-weight:600; "
            "color:#7f8c9a; text-transform:uppercase; letter-spacing:0.1em;'>C1 Detection</p>",
            unsafe_allow_html=True
        )
        if "c1_detection" in viz:
            st.image(
                base64.b64decode(viz["c1_detection"]),
                caption="Bounding Box Output",
                use_container_width=True
            )
        else:
            st.markdown(
                "<div style='height:160px; border:1px dashed #2a3a4a; border-radius:8px; "
                "display:flex; align-items:center; justify-content:center; color:#44576a; font-size:12px;'>"
                "No detection image</div>",
                unsafe_allow_html=True
            )

    # --- Right: Validation Panel ---
    with col_val:
        c1_conf = res.get("c1_conf", 0.0)
        conf_threshold = 0.45
        quality = res.get("c1_quality", {})
        overall_score = quality.get("overall", 0) if quality else 0
        q_threshold = 40.0

        conf_passed = c1_conf >= conf_threshold
        qual_passed = overall_score >= q_threshold

        conf_color  = "#00d97e" if conf_passed else "#ff4d6d"
        qual_color  = "#00d97e" if qual_passed else "#ff4d6d"
        conf_bg     = "rgba(0,217,126,0.07)" if conf_passed else "rgba(255,77,109,0.07)"
        qual_bg     = "rgba(0,217,126,0.07)" if qual_passed else "rgba(255,77,109,0.07)"
        conf_status = "PASSED" if conf_passed else "BLOCKED"
        qual_status = "PASSED" if qual_passed else "BLOCKED"

        conf_pct  = int(c1_conf * 100)
        conf_fill = int(c1_conf * 100)
        qual_fill = int(min(overall_score, 100))

        # ── Header row ──
        st.markdown(
            "<p style='margin:0 0 10px 0; font-size:24px; font-weight:600; "
            "color:#FFFFFF; '>Detection Validation</p>",
            unsafe_allow_html=True
        )

        # ── Full-width dual card + bar block ──
        st.markdown(
            f"""
            <style>
              .harp-bar-track {{
                width:100%; height:7px; background:#1e2d3d;
                border-radius:99px; overflow:hidden; margin-top:6px;
              }}
              .harp-bar-fill {{
                height:100%; border-radius:99px;
                transition: width 0.4s ease;
              }}
            </style>

            <div style='display:flex; gap:14px; width:100%;'>

              <!-- CONFIDENCE FLOOR -->
              <div style='flex:1; background:transparent; border:2px solid {conf_color}55;
                          border-radius:10px; padding:24px 20px;'>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
                  <span style='font-size:18px; font-weight:800; color:#ffffff;
                               text-transform:uppercase; letter-spacing:0.08em;'>Confidence Floor</span>
                  <span style='font-size:11px; font-weight:600; color:{conf_color};
                               border:1px solid {conf_color}55;
                               padding:2px 10px; border-radius:20px; letter-spacing:0.06em;'>{conf_status}</span>
                </div>
                <div style='display:flex; align-items:baseline; gap:6px; margin-bottom:4px;'>
                  <span style='font-size:32px; font-weight:800; color:{conf_color}; line-height:1;'>{c1_conf*100:.1f}%</span>
                  <span style='font-size:12px; color:#7f8c9a;'>/ min {conf_threshold*100:.0f}%</span>
                </div>
                <div style='font-size:11px; color:#9aaabb; margin-bottom:8px;'>C1 Confidence Score</div>
                <div class='harp-bar-track'>
                  <div class='harp-bar-fill' style='width:{conf_fill}%; background:linear-gradient(90deg,{conf_color}44,{conf_color}88);'></div>
                </div>
              </div>

              <!-- QUALITY GATE -->
              <div style='flex:1; background:transparent; border:2px solid {qual_color}55;
                          border-radius:10px; padding:24px 20px;'>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
                  <span style='font-size:18px; font-weight:800; color:#ffffff;
                               text-transform:uppercase; letter-spacing:0.08em;'>Quality Gate</span>
                  <span style='font-size:11px; font-weight:600; color:{qual_color};
                               border:1px solid {qual_color}55;
                               padding:2px 10px; border-radius:20px; letter-spacing:0.06em;'>{qual_status}</span>
                </div>
                <div style='display:flex; align-items:baseline; gap:6px; margin-bottom:4px;'>
                  <span style='font-size:32px; font-weight:800; color:{qual_color}; line-height:1;'>{overall_score:.0f}</span>
                  <span style='font-size:12px; color:#7f8c9a;'>/ 100 &nbsp;·&nbsp; min {q_threshold:.0f}</span>
                </div>
                <div style='font-size:11px; color:#9aaabb; margin-bottom:8px;'>Image Quality Score</div>
                <div class='harp-bar-track'>
                  <div class='harp-bar-fill' style='width:{qual_fill}%; background:linear-gradient(90deg,{qual_color}44,{qual_color}88);'></div>
                </div>
              </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ── Quality breakdown ──
        if quality:
            blur_val       = quality.get("blur", 0)
            brightness_val = quality.get("brightness", 127)
            contrast_val   = quality.get("contrast", 0)

            blur_perc   = min(blur_val / 500.0, 1.0)
            bright_perc = max(0.0, 1.0 - abs(brightness_val - 127) / 127.0)
            cont_perc   = min(contrast_val / 60.0, 1.0)

            def _metric_card(label, pct, good_lbl, bad_lbl):
                ok    = pct > 0.5
                color = "#00d97e" if ok else "#ff4d6d"
                badge = good_lbl if ok else bad_lbl
                badge_bg = "rgba(0,217,126,0.12)" if ok else "rgba(255,77,109,0.12)"
                fill = int(pct * 100)
                return f"""
                <div style='flex:1; background:#0e1620; border:1px solid #1e2d3d;
                            border-radius:10px; padding:12px 14px;'>
                  <div style='font-size:10px; color:#7f8c9a; text-transform:uppercase;
                              letter-spacing:0.1em; margin-bottom:6px;'>{label}</div>
                  <div style='font-size:26px; font-weight:800; color:#e0e6ed; line-height:1;
                              margin-bottom:4px;'>{fill}%</div>
                  <span style='font-size:10px; font-weight:600; color:{color};
                               background:{badge_bg}; padding:2px 8px;
                               border-radius:20px; letter-spacing:0.05em;'>{badge}</span>
                  <div class='harp-bar-track' style='margin-top:8px;'>
                    <div class='harp-bar-fill'
                         style='width:{fill}%; background:linear-gradient(90deg,{color}33,{color}66);'>
                    </div>
                  </div>
                </div>"""

            st.markdown(
                f"""
                <div style='display:flex; gap:12px; margin-top:14px; width:100%;'>
                  {_metric_card("Sharpness",  blur_perc,   "Clear",  "Blurry")}
                  {_metric_card("Brightness", bright_perc, "Good",   "Dark")}
                  {_metric_card("Contrast",   cont_perc,   "High",   "Low")}
                </div>
                """,
                unsafe_allow_html=True
            )


