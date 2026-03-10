# app/core/c4_confidence.py
# C4 Confidence Analyzer — Safe Extension, No modifications to C1/C2/C3

from dataclasses import dataclass, asdict
from typing import Literal, List
import math

ConfidenceTier = Literal["CERTAIN", "CONFIDENT", "UNCERTAIN", "UNRELIABLE"]

@dataclass
class C4ConfidenceResult:
    tier: ConfidenceTier
    score: float          # 0.0 – 100.0
    angular_gap: float    # degrees between the two hands
    diagnosis: List[str]  # list of flag strings
    reason: str           # human-readable explanation

    def to_dict(self):
        return asdict(self)

class C4ConfidenceAnalyzer:
    """
    Research Extension for C4 — Physics-Calibrated Confidence Scoring.

    Takes the outputs of _solve_physics() (angles + error) and produces
    a structured confidence assessment. Does NOT modify any existing component.
    """

    # Thresholds (tunable for your dataset)
    TIER_CERTAIN    = 5.0
    TIER_CONFIDENT  = 12.0
    TIER_UNCERTAIN  = 20.0

    OVERLAP_THRESHOLD      = 15.0   # degrees — hands this close = ambiguous
    HOUR_BOUNDARY_MINUTES  = 5      # within 5 min of an exact hour = boundary risk

    def _circular_gap(self, a1: float, a2: float) -> float:
        """Shortest angular distance between two angles on a 360° circle."""
        diff = abs(a1 - a2) % 360
        return min(diff, 360 - diff)

    def _classify_tier(self, error: float) -> ConfidenceTier:
        if error < self.TIER_CERTAIN:
            return "CERTAIN"
        elif error < self.TIER_CONFIDENT:
            return "CONFIDENT"
        elif error < self.TIER_UNCERTAIN:
            return "UNCERTAIN"
        else:
            return "UNRELIABLE"

    def _compute_score(self, error: float, penalties: float) -> float:
        """Linear score: 100 at 0° error, 0 at 20° error, minus penalties."""
        base = max(0.0, 100.0 - (error * 5.0))
        return max(0.0, min(100.0, base - penalties))

    def _is_hour_boundary(self, m: int) -> bool:
        """True if minute is within 5 of the top of the hour (hands near overlap)."""
        return m <= self.HOUR_BOUNDARY_MINUTES or m >= (60 - self.HOUR_BOUNDARY_MINUTES)

    def analyze(self, a1: float, a2: float, error: float, h: int, m: int) -> C4ConfidenceResult:
        """
        Main analysis method.

        Args:
            a1, a2  : detected hand angles in degrees (from C2 / _get_angle)
            error   : angular error score from _solve_physics()
            h, m    : resolved hour and minute from _solve_physics()

        Returns:
            C4ConfidenceResult with tier, score, diagnosis, and reason
        """
        diagnosis = []
        penalties = 0.0
        reasons = []

        # STEP 1 — Angular gap between hands
        gap = self._circular_gap(a1, a2)

        # STEP 2 — Classify error tier
        tier = self._classify_tier(error)

        # STEP 3 — Diagnose risk factors
        if gap < self.OVERLAP_THRESHOLD:
            diagnosis.append("near_overlap")
            penalties += 15.0
            reasons.append(f"hands are only {gap:.1f}° apart (overlap risk)")

        if self._is_hour_boundary(m):
            diagnosis.append("hour_boundary")
            penalties += 5.0
            reasons.append(f"time {h}:{m:02d} is near an hour boundary")

        if error >= self.TIER_UNCERTAIN:
            diagnosis.append("high_error")
            reasons.append(f"physics fit error is high ({error:.1f}°)")

        if not diagnosis:
            diagnosis.append("clean")
            reasons.append("hands are well-separated and error is low")

        # STEP 4 — Final score
        score = self._compute_score(error, penalties)

        # STEP 5 — Human-readable reason
        reason = f"{tier}: " + "; ".join(reasons) + f". Score: {score:.0f}/100."

        return C4ConfidenceResult(
            tier=tier,
            score=round(score, 1),
            angular_gap=round(gap, 2),
            diagnosis=diagnosis,
            reason=reason
        )
