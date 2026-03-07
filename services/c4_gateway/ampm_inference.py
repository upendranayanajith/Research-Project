"""
C4 AM/PM Inference Engine (Day 1)
====================================
Infers the most probable AM/PM period for a detected clock time using:
  1. Physics plausibility — how well do the hand angles fit each candidate?
  2. Hour-hand sub-degree position — the hour hand moves 0.5° per minute, so
     a reading of e.g. 6:30 has the hour hand at exactly 195°. If the observed
     angle is within a narrow band around 195° rather than 6° (6:00), the 30-min
     sub-position confirms the reading tightly.
  3. Optional external hint (e.g. morning/afternoon tag from user metadata).

Returns: "AM", "PM", or "Unknown" along with a confidence 0-100.
Owner: Member 4
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class AMPMResult:
    period: str          # "AM" | "PM" | "Unknown"
    confidence: float    # 0 – 100
    reason: str          # human-readable explanation


class AMPMInferenceEngine:
    """
    Stateless engine — all methods are pure functions of the input angles.

    Physics background
    ------------------
    On a 12-hour clock face the hands repeat every 12 hours, so the clock
    itself carries no AM/PM information.  However, we can still attach a
    *probabilistic* statement by reasoning about:
      • Consistency of H-hand position mid-hour  (0.5° / min means at hh:mm
        the expected h-angle = hh*30 + mm*0.5)
      • User-supplied hint (sunrise / daytime / evening tag)
    If no hint is supplied the engine returns "Unknown" with 50 % confidence
    because the raw angles truly cannot disambiguate AM from PM.
    """

    # Typical daylight windows (heuristics only, not hard rules)
    _LIKELY_DAYTIME_HOURS = set(range(6, 22))   # 06:00 – 21:59
    _LIKELY_NIGHT_HOURS   = set(range(0, 6)) | set(range(22, 24))

    # ------------------------------------------------------------------ #
    def infer(
        self,
        hour: int,
        minute: int,
        hand1_angle: float,
        hand2_angle: float,
        user_hint: Optional[str] = None,   # "morning" | "afternoon" | "evening" | "night"
    ) -> AMPMResult:
        """
        Main entry point.

        Parameters
        ----------
        hour, minute : resolved clock reading (1-12, 0-59)
        hand1_angle, hand2_angle : raw detected angles in degrees (0-360)
        user_hint : optional free-text context from the caller
        """
        # --- Step 1: compute expected h-angle for this reading ----------
        h_norm = hour % 12
        expected_h_angle = (h_norm * 30.0 + minute * 0.5) % 360.0

        # How close is the observed h-angle to the expected position?
        observed_h = hand1_angle
        diff = abs(observed_h - expected_h_angle)
        diff = min(diff, 360.0 - diff)
        angle_consistency_score = max(0.0, 100.0 - diff * 3.0)  # shrinks with error

        # --- Step 2: hour-range heuristics ------------------------------
        # Convert 12-hour reading to 0-23 candidates
        am_hour = hour % 12           # 0 … 11
        pm_hour = hour % 12 + 12      # 12 … 23

        am_plausible = am_hour in self._LIKELY_DAYTIME_HOURS or am_hour in self._LIKELY_NIGHT_HOURS
        pm_plausible = pm_hour in self._LIKELY_DAYTIME_HOURS or pm_hour in self._LIKELY_NIGHT_HOURS

        # --- Step 3: user hint override ---------------------------------
        if user_hint:
            hint = user_hint.lower()
            if any(w in hint for w in ("morning", "dawn", "sunrise", "am")):
                return AMPMResult(
                    "AM",
                    min(95.0, 70.0 + angle_consistency_score * 0.25),
                    f"User hint '{user_hint}' → AM. Angle consistency: {angle_consistency_score:.0f}/100.",
                )
            if any(w in hint for w in ("afternoon", "evening", "dusk", "night", "pm")):
                return AMPMResult(
                    "PM",
                    min(95.0, 70.0 + angle_consistency_score * 0.25),
                    f"User hint '{user_hint}' → PM. Angle consistency: {angle_consistency_score:.0f}/100.",
                )

        # --- Step 4: without external hint, return Unknown --------------
        # Give a soft preference based on hour plausibility
        if am_hour in self._LIKELY_DAYTIME_HOURS and pm_hour in self._LIKELY_NIGHT_HOURS:
            period = "AM"
            conf = 55.0
            reason = (
                f"{hour}:{minute:02d} falls in typical daytime (AM more common). "
                f"Angle consistency: {angle_consistency_score:.0f}/100. "
                "Supply a time-of-day hint for higher confidence."
            )
        elif pm_hour in self._LIKELY_DAYTIME_HOURS and am_hour in self._LIKELY_NIGHT_HOURS:
            period = "PM"
            conf = 55.0
            reason = (
                f"{hour}:{minute:02d} in PM range is typical daytime. "
                f"Angle consistency: {angle_consistency_score:.0f}/100. "
                "Supply a time-of-day hint for higher confidence."
            )
        else:
            period = "Unknown"
            conf = 50.0
            reason = (
                f"Clock face cannot disambiguate AM/PM without context. "
                f"Angle consistency score: {angle_consistency_score:.0f}/100."
            )

        return AMPMResult(period, round(conf, 1), reason)

    # ------------------------------------------------------------------ #
    def to_dict(self, result: AMPMResult) -> dict:
        return {
            "period": result.period,
            "confidence": result.confidence,
            "reason": result.reason,
        }


# Singleton
ampm_engine = AMPMInferenceEngine()
