"""
C4 Clock Accuracy Checker (Day 3)
======================================
Compares a detected clock reading against the host machine's current UTC
(and optionally a user-specified timezone) and reports:

  • offset_minutes  — how many minutes the clock is ahead (+) or behind (-)
  • verdict         — "Accurate" | "Slightly Off" | "Needs Adjustment"
  • drift_class     — "On Time" | "Fast" | "Slow"
  • suggestion      — plain-English advice

Owner: Member 4
"""

from __future__ import annotations
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class AccuracyReport:
    detected_time_str: str       # e.g. "3:47"
    reference_time_str: str      # current real time (HH:MM)
    offset_minutes: int          # positive = fast, negative = slow
    drift_class: str             # "On Time" | "Fast" | "Slow"
    verdict: str                 # "Accurate" | "Slightly Off" | "Needs Adjustment"
    suggestion: str
    timezone_used: str


class ClockAccuracyChecker:
    """
    Purely computational — no I/O except reading the system clock.

    Accuracy tiers
    --------------
    ±1 min  → Accurate (clocks rarely have sub-minute displays)
    ±2-5 min → Slightly Off
    > 5 min  → Needs Adjustment
    """

    _ACCURATE_THRESHOLD  = 1    # minutes
    _SLIGHT_THRESHOLD    = 5    # minutes

    # ------------------------------------------------------------------ #
    def check(
        self,
        detected_hour: int,
        detected_minute: int,
        period: Optional[str] = None,    # "AM" | "PM" | None
        tz_offset_hours: float = 0.0,    # UTC offset e.g. +5.5 for IST
    ) -> AccuracyReport:
        """
        Parameters
        ----------
        detected_hour   : 1-12
        detected_minute : 0-59
        period          : "AM" / "PM" if known, else None (compare modulo 12 h)
        tz_offset_hours : UTC offset in hours (offset from UTC, e.g. 5.5 for UTC+5:30)
        """
        tz = timezone(timedelta(hours=tz_offset_hours))
        now = datetime.now(tz)
        tz_name = f"UTC{'+' if tz_offset_hours >= 0 else ''}{tz_offset_hours:g}"

        ref_hour_24 = now.hour
        ref_minute  = now.minute
        ref_total   = ref_hour_24 * 60 + ref_minute   # absolute minutes in the day

        # Convert detected time to absolute minutes assuming best AM/PM match
        det_h_24 = self._resolve_24h(detected_hour, detected_minute, ref_hour_24, period)
        det_total = det_h_24 * 60 + detected_minute

        # Offset (modulo 12 h because the clock can't tell us which 12-hour block)
        raw_offset = det_total - ref_total

        # Normalise into -360 … +360 (max 6-hour window; beyond that is uncertain)
        if raw_offset > 360:
            raw_offset -= 720
        elif raw_offset < -360:
            raw_offset += 720

        offset = raw_offset  # signed minutes

        # Drift class
        if offset > 0:
            drift_class = "Fast"
        elif offset < 0:
            drift_class = "Slow"
        else:
            drift_class = "On Time"

        # Verdict
        abs_off = abs(offset)
        if abs_off <= self._ACCURATE_THRESHOLD:
            verdict = "Accurate"
            suggestion = "Your clock is running correctly — no adjustment needed."
        elif abs_off <= self._SLIGHT_THRESHOLD:
            verdict = "Slightly Off"
            direction = "fast" if offset > 0 else "slow"
            suggestion = (
                f"Your clock is {abs_off} minute(s) {direction}. "
                "A small manual correction is recommended."
            )
        else:
            verdict = "Needs Adjustment"
            direction = "fast" if offset > 0 else "slow"
            suggestion = (
                f"Your clock is significantly {direction} by {abs_off} minute(s). "
                "Please reset it to the correct time."
            )

        detected_str = f"{detected_hour}:{detected_minute:02d}"
        reference_str = f"{ref_hour_24 % 12 or 12}:{ref_minute:02d}"

        return AccuracyReport(
            detected_time_str=detected_str,
            reference_time_str=reference_str,
            offset_minutes=int(offset),
            drift_class=drift_class,
            verdict=verdict,
            suggestion=suggestion,
            timezone_used=tz_name,
        )

    # ------------------------------------------------------------------ #
    def _resolve_24h(
        self,
        hour: int,
        minute: int,
        ref_hour_24: int,
        period: Optional[str],
    ) -> int:
        """Convert 12-hour reading to best-match 24-hour reading."""
        h12 = hour % 12  # 0-11
        if period == "AM":
            return h12
        if period == "PM":
            return h12 + 12
        # No hint — pick whichever 12-hour offset is closest to ref
        candidates = [h12, h12 + 12]
        best = min(candidates, key=lambda c: abs(c - ref_hour_24))
        return best

    # ------------------------------------------------------------------ #
    def to_dict(self, report: AccuracyReport) -> dict:
        return {
            "detected_time": report.detected_time_str,
            "reference_time": report.reference_time_str,
            "offset_minutes": report.offset_minutes,
            "drift_class": report.drift_class,
            "verdict": report.verdict,
            "suggestion": report.suggestion,
            "timezone_used": report.timezone_used,
        }


# Singleton
accuracy_checker = ClockAccuracyChecker()
