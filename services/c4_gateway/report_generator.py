"""
C4 Clock Reading Report Generator (Day 5)
================================================
Generates a rich, plain-English narrative report for a clock analysis result.
The report covers:
  1. What the pipeline detected (method + path taken)
  2. Angle interpretation (what each hand position means geometrically)
  3. Time reading with AM/PM context
  4. Confidence interpretation
  5. Ambiguity notes (if applicable)
  6. Accuracy vs real time (if available)
  7. A summary verdict sentence

Owner: Member 4
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class ClockReport:
    title: str
    sections: List[Dict[str, str]]   # [{"heading": ..., "body": ...}]
    generated_at: str
    one_liner: str


class ReportGenerator:
    """Stateless — generates a report dict from analysis results."""

    # ------------------------------------------------------------------ #
    def generate(
        self,
        analysis_result: Dict[str, Any],
        ampm_result: Optional[Dict] = None,
        ambiguity_result: Optional[Dict] = None,
        accuracy_result: Optional[Dict] = None,
    ) -> ClockReport:
        """
        Parameters
        ----------
        analysis_result  : the dict returned by /analyze (result key)
        ampm_result      : output of ampm_engine.to_dict(...)
        ambiguity_result : output of ambiguity_resolver.to_dict(...)
        accuracy_result  : output of accuracy_checker.to_dict(...)
        """
        sections: List[Dict[str, str]] = []

        time_str    = analysis_result.get("time", "??:??")
        method      = analysis_result.get("method", "Unknown")
        confidence  = analysis_result.get("confidence", "Unknown")
        angles      = analysis_result.get("angles", {})
        reasoning   = analysis_result.get("reasoning", "")
        debug       = analysis_result.get("debug", [])

        h_angle = angles.get("hand1", 0.0)
        m_angle = angles.get("hand2", 0.0)

        # ---- Section 1: Pipeline Path -------------------------------- #
        is_expert = "Expert" in method
        path_body = (
            f"The image was processed using the **{method}** pipeline. "
        )
        if is_expert:
            path_body += (
                "All four components were activated: C1 located and cropped the "
                "clock face, C2 identified the hand keypoints, C3 refined the "
                "angles using a deep regression model, and C4 computed the final "
                "time using physics-based constraints."
            )
        else:
            path_body += (
                "C1 located and cropped the clock face, C2 extracted the hand "
                "skeleton, and C4 resolved the time directly from the raw angles. "
                "The C3 expert AI was not required because the initial fit was "
                "sufficiently confident."
            )
        if debug:
            path_body += f" Pipeline trace: {' → '.join(debug)}."
        sections.append({"heading": "Pipeline Path", "body": path_body})

        # ---- Section 2: Hand Geometry -------------------------------- #
        h_clock_pos = self._angle_to_clock_position(h_angle)
        m_clock_pos = self._angle_to_clock_position(m_angle)
        geom_body = (
            f"The first detected hand is at **{h_angle:.1f}°** from 12-o'clock "
            f"(approximately the {h_clock_pos} position on the dial). "
            f"The second hand is at **{m_angle:.1f}°** "
            f"(approximately the {m_clock_pos} position). "
            f"Physics assigns the hour role to the hand with the smaller angular "
            f"displacement per minute (0.5°/min) and the minute role to the faster "
            f"hand (6°/min)."
        )
        sections.append({"heading": "Hand Geometry", "body": geom_body})

        # ---- Section 3: Time Reading --------------------------------- #
        period_str = ""
        if ampm_result:
            p = ampm_result.get("period", "Unknown")
            pc = ampm_result.get("confidence", 50)
            period_str = f" The period is inferred as **{p}** (confidence: {pc:.0f}%)."
        time_body = (
            f"The clock reads **{time_str}**.{period_str} "
            f"{reasoning}"
        )
        sections.append({"heading": "Time Reading", "body": time_body})

        # ---- Section 4: Confidence & Fit ----------------------------- #
        conf_body = self._describe_confidence(confidence, ambiguity_result)
        sections.append({"heading": "Confidence Assessment", "body": conf_body})

        # ---- Section 5: Ambiguity (conditional) ---------------------- #
        if ambiguity_result:
            amb_body = self._describe_ambiguity(ambiguity_result, time_str)
            sections.append({"heading": "Ambiguity Analysis", "body": amb_body})

        # ---- Section 6: Accuracy vs Real Time (conditional) ---------- #
        if accuracy_result:
            acc_body = self._describe_accuracy(accuracy_result)
            sections.append({"heading": "Clock Accuracy", "body": acc_body})

        # ---- One-liner summary --------------------------------------- #
        one_liner = self._one_liner(
            time_str, confidence, is_expert, ampm_result, accuracy_result
        )

        return ClockReport(
            title=f"Clock Reading Report — {time_str}",
            sections=sections,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            one_liner=one_liner,
        )

    # ------------------------------------------------------------------ #
    def _angle_to_clock_position(self, angle: float) -> str:
        """Map 0-360° to a rough clock-face description."""
        angle = angle % 360
        positions = [
            (0,  15, "12"),  (15, 45,  "1"),  (45, 75, "2"),
            (75, 105, "3"),  (105,135, "4"),  (135,165,"5"),
            (165,195,"6"),   (195,225,"7"),   (225,255,"8"),
            (255,285,"9"),   (285,315,"10"),  (315,345,"11"),
            (345,360,"12"),
        ]
        for lo, hi, label in positions:
            if lo <= angle < hi:
                return f"{label} o'clock"
        return "12 o'clock"

    def _describe_confidence(self, confidence: str, ambiguity: Optional[dict]) -> str:
        base = {
            "High":     "The fast-path angular fit was tight, meaning the hand "
                        "positions map cleanly to a single time with minimal error.",
            "Refined":  "The expert AI (C3) was invoked and produced a refined angle "
                        "estimate, raising overall confidence.",
            "Low":      "C3 was unavailable or could not improve the fit. The result "
                        "is based on raw C2 angles and should be treated with caution.",
        }.get(confidence, f"Confidence level: {confidence}.")

        if ambiguity and ambiguity.get("is_ambiguous"):
            base += (
                " **However**, the ambiguity resolver flagged this reading: "
                + ambiguity.get("ambiguity_reason", "") + " Consider the top-N "
                "candidates listed in the Ambiguity Analysis section."
            )
        return base

    def _describe_ambiguity(self, amb: dict, best_time: str) -> str:
        candidates = amb.get("top_candidates", [])
        if not candidates:
            return "No candidate data available."
        lines = [
            f"The physics solver evaluated all 720 possible minute positions. "
            f"The top-{len(candidates)} candidates are:"
        ]
        for i, c in enumerate(candidates, 1):
            lines.append(
                f"  {i}. **{c['time']}** — angular error {c['angular_error']:.1f}°, "
                f"confidence {c['confidence_pct']:.1f}%, fit: {c['fit_quality']}."
            )
        if amb.get("is_ambiguous"):
            lines.append(
                f"\nThe reading **{best_time}** is returned as the primary result, "
                "but the alternatives above are plausible. If you have context about "
                "the approximate time, use it to select the correct candidate."
            )
        return "\n".join(lines)

    def _describe_accuracy(self, acc: dict) -> str:
        verdict  = acc.get("verdict", "")
        offset   = acc.get("offset_minutes", 0)
        drift    = acc.get("drift_class", "")
        suggest  = acc.get("suggestion", "")
        ref_time = acc.get("reference_time", "")
        tz       = acc.get("timezone_used", "UTC")
        return (
            f"The device's reference time ({tz}) shows **{ref_time}**. "
            f"This clock is **{drift}** by {abs(offset)} minute(s) → **{verdict}**. "
            f"{suggest}"
        )

    def _one_liner(
        self,
        time_str: str,
        confidence: str,
        is_expert: bool,
        ampm: Optional[dict],
        acc: Optional[dict],
    ) -> str:
        parts = [f"Clock reads {time_str}"]
        if ampm and ampm.get("period") != "Unknown":
            parts[0] += f" {ampm['period']}"
        parts.append(f"confidence: {confidence}")
        if is_expert:
            parts.append("Expert AI used")
        if acc:
            v = acc.get("verdict", "")
            o = acc.get("offset_minutes", 0)
            if v == "Accurate":
                parts.append("clock is accurate")
            else:
                d = "fast" if o > 0 else "slow"
                parts.append(f"clock is {d} by {abs(o)} min")
        return " | ".join(parts) + "."

    # ------------------------------------------------------------------ #
    def to_dict(self, report: ClockReport) -> dict:
        return {
            "title": report.title,
            "one_liner": report.one_liner,
            "generated_at": report.generated_at,
            "sections": report.sections,
        }


# Singleton
report_generator = ReportGenerator()
