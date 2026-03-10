import pathlib

fe = pathlib.Path(r"d:\Y4S1\Research 4\Clock_Time_Research\Research-Project\app\frontend.py")
content = fe.read_text(encoding="utf-8")

OLD = '                    st.markdown(f"{icon_char} `{line}`")\n        else:\n            st.info("Expert AI skipped'

PANEL = (
    '                    st.markdown(f"{icon_char} `{line}`")\n\n'
    '            # --- [Tier 1.4] Temporal Stability Panel ---\n'
    '            temporal_xai = res.get("temporal_xai")\n'
    '            if temporal_xai:\n'
    '                st.markdown("---")\n'
    '                st.markdown(\n'
    '                    f"**{icon(\'bar_chart\')} \U0001F4C8 Temporal Stability (Kalman Filter)**",\n'
    '                    unsafe_allow_html=True,\n'
    '                )\n'
    '                t_status = temporal_xai.get("status", "N/A")\n'
    '                if t_status == "Initialising":\n'
    '                    st.info(temporal_xai.get("message", "Kalman filter warming up..."))\n'
    '                elif t_status == "Active":\n'
    '                    t_cols = st.columns(4)\n'
    "                    t_cols[0].metric('Stability', f\"{temporal_xai.get('stability_score', 'N/A')}%\")\n"
    "                    t_cols[1].metric('Trend', temporal_xai.get('trend', 'N/A'))\n"
    "                    t_cols[2].metric('Spikes Rejected', temporal_xai.get('total_spike_count', 0))\n"
    "                    t_cols[3].metric('Avg Correction', f\"{temporal_xai.get('mean_kalman_correction_deg', 0):.1f}\u00b0\")\n"
    '                    st.caption(f"\U0001F522 {temporal_xai.get(\'message\', \'\')}")\n'
    '                    with st.expander("Variance Details"):\n'
    '                        st.json({\n'
    '                            "hand1_variance_deg": temporal_xai.get("hand1_variance_deg"),\n'
    '                            "hand2_variance_deg": temporal_xai.get("hand2_variance_deg"),\n'
    '                            "spike_rate_per_frame": temporal_xai.get("spike_rate_per_frame"),\n'
    '                            "frames_seen": temporal_xai.get("frames_seen"),\n'
    '                        })\n'
    '            else:\n'
    '                st.caption("\U0001F4C8 Temporal Stability: N/A (only active in Live Webcam / RTSP mode)")\n'
    '        else:\n'
    '            st.info("Expert AI skipped'
)

if OLD in content:
    content = content.replace(OLD, PANEL, 1)
    fe.write_text(content, encoding="utf-8")
    print("OK: Temporal Stability panel injected.")
else:
    print("NOT FOUND — checking nearby text:")
    idx = content.find('icon_char} `')
    print(repr(content[max(0,idx-5):idx+90]))
