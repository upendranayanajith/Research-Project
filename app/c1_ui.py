import streamlit as st
import base64

# ── Thresholds ──────────────────────────────────────────────────────────────
_CONF_FLOOR  = 45.0   # min C1 confidence %
_QUALITY_MIN = 40     # min quality gate score

# ── Pure-string helpers (no nested HTML-returning calls inside f-strings) ───

def _badge_html(passed: bool) -> str:
    if passed:
        return (
            "<span style='background:#00ff9022;color:#00e67a;"
            "padding:3px 14px;border-radius:4px;font-size:11px;"
            "font-weight:800;border:1px solid #00e67a55;letter-spacing:1px;'>"
            "PASSED</span>"
        )
    return (
        "<span style='background:#ff444422;color:#ff6666;"
        "padding:3px 14px;border-radius:4px;font-size:11px;"
        "font-weight:800;border:1px solid #ff666655;letter-spacing:1px;'>"
        "FAILED</span>"
    )


def _bar_html(frac: float, color: str = "#00e67a") -> str:
    pct = int(min(max(frac, 0.0), 1.0) * 100)
    return (
        f"<div style='background:#1e293b;border-radius:4px;height:6px;margin-top:10px;'>"
        f"<div style='width:{pct}%;height:6px;background:{color};border-radius:4px;'></div>"
        f"</div>"
    )


def _top_card(title: str, value_html: str, sub: str, bar_frac: float, badge: str) -> str:
    bar = _bar_html(bar_frac)
    return (
        "<div style='background:#111827;border:1px solid #1e293b;"
        "border-radius:12px;padding:20px 22px;'>"
        "<div style='display:flex;justify-content:space-between;"
        f"align-items:flex-start;margin-bottom:6px;'>"
        f"<div style='color:#9ca3af;font-size:11px;letter-spacing:1.5px;"
        f"text-transform:uppercase;font-weight:700;'>{title}</div>"
        f"{badge}"
        "</div>"
        f"<div style='margin-top:8px;'>{value_html}</div>"
        f"<div style='color:#6b7280;font-size:12px;margin-top:4px;'>{sub}</div>"
        f"{bar}"
        "</div>"
    )


def _metric_card(label: str, value: str, sub: str, bar_frac: float) -> str:
    bar = _bar_html(bar_frac)
    return (
        "<div style='background:#111827;border:1px solid #1e293b;"
        "border-radius:10px;padding:18px 20px;'>"
        f"<div style='color:#6b7280;font-size:11px;letter-spacing:1px;"
        f"text-transform:uppercase;margin-bottom:4px;'>{label}</div>"
        f"<div style='color:white;font-size:26px;font-weight:800;line-height:1;'>{value}</div>"
        f"<div style='color:#00e67a;font-size:12px;margin-top:4px;'>{sub}</div>"
        f"{bar}"
        "</div>"
    )


# ── Main render function ─────────────────────────────────────────────────────

def render_c1_localization(viz, res):
    col_img, col_val = st.columns([1, 2])

    # ── LEFT: Detection image ────────────────────────────────────────────────
    with col_img:
        st.markdown(
            "<div style='color:#6b7280;font-size:11px;letter-spacing:2px;"
            "text-transform:uppercase;margin-bottom:8px;'>C1 DETECTION</div>",
            unsafe_allow_html=True,
        )
        if "c1_detection" in viz:
            st.image(
                base64.b64decode(viz["c1_detection"]),
                caption="Bounding Box Output",
                use_container_width=True,
            )
        else:
            st.info("No detection image available.")

    # ── RIGHT: Detection Validation ──────────────────────────────────────────
    with col_val:
        st.markdown(
            "<h2 style='color:white;font-size:28px;font-weight:700;"
            "margin-bottom:20px;'>Detection Validation</h2>",
            unsafe_allow_html=True,
        )

        # ── Row 1: Confidence Floor  +  Quality Gate ─────────────────────────
        raw_conf  = res.get("c1_conf", 0.0)
        conf_perc = float(raw_conf) / 100.0 if raw_conf > 1.0 else float(raw_conf)
        conf_perc = max(0.0, min(1.0, conf_perc))
        conf_disp = conf_perc * 100.0
        conf_pass = conf_disp >= _CONF_FLOOR

        q         = res.get("c1_quality") or {}
        overall   = int(q.get("overall", 0))
        qual_pass = overall >= _QUALITY_MIN

        conf_value_html = (
            f"<span style='color:#00e67a;font-size:34px;font-weight:800;'>{conf_disp:.1f}%</span>"
            f"<span style='color:#6b7280;font-size:14px;margin-left:6px;'>/ min {int(_CONF_FLOOR)}%</span>"
        )
        qual_value_html = (
            f"<span style='color:#00e67a;font-size:34px;font-weight:800;'>{overall}</span>"
            f"<span style='color:#6b7280;font-size:14px;margin-left:6px;'>/ 100 · min {_QUALITY_MIN}</span>"
        )

        row1_a, row1_b = st.columns(2)
        with row1_a:
            st.markdown(
                _top_card("CONFIDENCE FLOOR", conf_value_html, "C1 Confidence Score",
                          conf_perc, _badge_html(conf_pass)),
                unsafe_allow_html=True,
            )
        with row1_b:
            st.markdown(
                _top_card("QUALITY GATE", qual_value_html, "Image Quality Score",
                          overall / 100.0, _badge_html(qual_pass)),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # ── Row 2: Sharpness  Brightness  Contrast ───────────────────────────
        if q:
            raw_blur   = q.get("blur", 0)
            raw_bright = q.get("brightness", 0)
            raw_cont   = q.get("contrast", 0)

            blur_frac   = min(raw_blur / 500.0, 1.0)
            bright_frac = max(0.0, 1.0 - (abs(raw_bright - 127) / 127.0))
            cont_frac   = min(raw_cont / 60.0, 1.0)

            sharp_lbl  = "Clear"  if blur_frac   > 0.5 else "Blurry"
            bright_lbl = "Good"   if bright_frac > 0.5 else "Dark"
            cont_lbl   = "High"   if cont_frac   > 0.5 else "Low"

            row2_a, row2_b, row2_c = st.columns(3)
            with row2_a:
                st.markdown(
                    _metric_card("SHARPNESS",  f"{int(blur_frac*100)}%",   sharp_lbl,  blur_frac),
                    unsafe_allow_html=True,
                )
            with row2_b:
                st.markdown(
                    _metric_card("BRIGHTNESS", f"{int(bright_frac*100)}%", bright_lbl, bright_frac),
                    unsafe_allow_html=True,
                )
            with row2_c:
                st.markdown(
                    _metric_card("CONTRAST",   f"{int(cont_frac*100)}%",   cont_lbl,   cont_frac),
                    unsafe_allow_html=True,
                )
        else:
            st.info("No quality metrics available.")
