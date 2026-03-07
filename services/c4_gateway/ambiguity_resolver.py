"""
C4 Ambiguity Resolver (Day 2)
=================================
When two hand angles are close together (e.g. 12:00 region) or the physics
fit has many near-equal minima, a single-best-guess answer may be misleading.

This module:
  • Scores every possible (h, m) pair (720 total for 12-hour clock) against
    the observed angles.
  • Returns the top-N candidates sorted by cumulative angular error (ascending).
  • Attaches a normalised confidence % to each candidate.
  • Flags "ambiguous" when the gap between #1 and #2 is smaller than a
    configurable threshold.

Owner: Member 4
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class TimeCandidate:
    hour: int
    minute: int
    time_str: str
    angular_error: float   # lower = better fit
    confidence: float      # 0 – 100, higher = better
    fit_quality: str       # "Excellent" | "Good" | "Marginal" | "Poor"


@dataclass
class AmbiguityResult:
    best: TimeCandidate
    candidates: List[TimeCandidate]
    is_ambiguous: bool
    ambiguity_reason: str


class AmbiguityResolver:
    """
    Scores all 720 minute-positions in a 12-hour cycle and selects top-N.

    Error metric
    ------------
    For each total_minutes t (0 … 719):
        theory_h  = (t * 0.5) mod 360    (hour hand moves 0.5°/min)
        theory_m  = (t * 6.0) mod 360    (minute hand moves 6°/min)

    We try both assignments (a1=hour, a2=minute) and (a1=minute, a2=hour)
    and take whichever gives the lower combined circular error.
    """

    def __init__(self, ambiguity_threshold_deg: float = 5.0):
        """
        Parameters
        ----------
        ambiguity_threshold_deg : if the top-2 angular errors differ by less
            than this value the result is flagged as ambiguous.
        """
        self.threshold = ambiguity_threshold_deg
        self._t = np.arange(0, 720, dtype=float)
        self._theory_h = (self._t * 0.5) % 360.0
        self._theory_m = (self._t * 6.0) % 360.0

    # ------------------------------------------------------------------ #
    def _circular_diff(self, a: np.ndarray, b: float) -> np.ndarray:
        """Minimum circular distance between array `a` and scalar `b`."""
        d = np.abs(a - b)
        return np.minimum(d, 360.0 - d)

    # ------------------------------------------------------------------ #
    def resolve(self, angle1: float, angle2: float, top_n: int = 5) -> AmbiguityResult:
        """
        Parameters
        ----------
        angle1, angle2 : observed hand angles in degrees.
        top_n          : number of top candidates to return.
        """
        # Both assignments
        err_A = self._circular_diff(self._theory_h, angle1) + \
                self._circular_diff(self._theory_m, angle2)
        err_B = self._circular_diff(self._theory_h, angle2) + \
                self._circular_diff(self._theory_m, angle1)

        # Best per position
        err_combined = np.minimum(err_A, err_B)

        # Sort ascending
        sorted_idx = np.argsort(err_combined)

        # Convert errors to confidence (softmax-like normalisation on top_n)
        top_idx = sorted_idx[: max(top_n, 10)]   # take at least 10 for normalisation
        top_err = err_combined[top_idx]

        # Confidence: invert error, then normalise so they sum to 100
        inv_err = 1.0 / (top_err + 1e-6)
        conf_raw = inv_err / inv_err.sum() * 100.0

        def _quality(e: float) -> str:
            if e < 3:   return "Excellent"
            if e < 8:   return "Good"
            if e < 15:  return "Marginal"
            return "Poor"

        candidates: List[TimeCandidate] = []
        for rank, (idx, conf) in enumerate(zip(top_idx[:top_n], conf_raw[:top_n])):
            t = int(self._t[idx])
            h = (t // 60) or 12
            m = t % 60
            candidates.append(TimeCandidate(
                hour=h,
                minute=m,
                time_str=f"{h}:{m:02d}",
                angular_error=round(float(err_combined[idx]), 2),
                confidence=round(float(conf), 1),
                fit_quality=_quality(float(err_combined[idx])),
            ))

        # Ambiguity check: compare #1 vs #2 errors
        if len(candidates) >= 2:
            gap = candidates[1].angular_error - candidates[0].angular_error
            is_ambiguous = gap < self.threshold
            if is_ambiguous:
                ambiguity_reason = (
                    f"Top-2 candidates ({candidates[0].time_str} & {candidates[1].time_str}) "
                    f"differ by only {gap:.1f}° — angles are inconclusive."
                )
            else:
                ambiguity_reason = (
                    f"Clear winner: {candidates[0].time_str} leads by {gap:.1f}°."
                )
        else:
            is_ambiguous = False
            ambiguity_reason = "Single candidate."

        return AmbiguityResult(
            best=candidates[0],
            candidates=candidates,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason,
        )

    # ------------------------------------------------------------------ #
    def to_dict(self, result: AmbiguityResult) -> dict:
        return {
            "best_time": result.best.time_str,
            "is_ambiguous": result.is_ambiguous,
            "ambiguity_reason": result.ambiguity_reason,
            "top_candidates": [
                {
                    "time": c.time_str,
                    "hour": c.hour,
                    "minute": c.minute,
                    "angular_error": c.angular_error,
                    "confidence_pct": c.confidence,
                    "fit_quality": c.fit_quality,
                }
                for c in result.candidates
            ],
        }


# Singleton
ambiguity_resolver = AmbiguityResolver(ambiguity_threshold_deg=5.0)
