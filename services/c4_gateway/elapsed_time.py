"""
C4 Elapsed Time Calculator (Day 4)
=========================================
Given two clock readings (start and end), computes:
  • elapsed_minutes        — total minutes elapsed
  • elapsed_display        — "Xh Ym" human format
  • direction              — "Forward" | "Backward" (did time go forward?)
  • possible_spans         — considering AM/PM ambiguity, list of plausible spans
  • most_probable_span_min — physics-best single answer

Handles the 12-hour ambiguity by returning all plausible spans (there can be
up to 4 interpretations when neither reading has AM/PM tag), and ranking them
by how close they are to a "natural" short duration (people rarely compare two
clocks more than 12 hours apart).

Owner: Member 4
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional



@dataclass
class SpanCandidate:
    start_str: str
    end_str: str
    elapsed_minutes: int
    elapsed_display: str
    direction: str        # "Forward" | "Backward"
    plausibility: str     # "High" | "Medium" | "Low"


@dataclass
class ElapsedTimeResult:
    most_probable: SpanCandidate
    all_spans: List[SpanCandidate]
    notes: str


class ElapsedTimeCalculator:
    """
    Computes elapsed time between two 12-hour clock readings.

    Plausibility ranking heuristic
    --------------------------------
    Shorter positive spans (0-12 h) are considered more probable than
    cross-noon spans (12-24 h) or negative spans (clock went backward).
    """

    def calculate(
        self,
        start_hour: int, start_minute: int,
        end_hour:   int, end_minute:   int,
        start_period: Optional[str] = None,   # "AM" | "PM" | None
        end_period:   Optional[str] = None,   # "AM" | "PM" | None
    ) -> ElapsedTimeResult:
        """
        Parameters
        ----------
        start_hour/minute : reading of the BEFORE clock (1-12, 0-59)
        end_hour/minute   : reading of the AFTER  clock (1-12, 0-59)
        start/end_period  : optional AM/PM labels
        """
        # Build 24h candidates
        start_candidates = self._24h_candidates(start_hour, start_minute, start_period)
        end_candidates   = self._24h_candidates(end_hour,   end_minute,   end_period)

        spans: List[SpanCandidate] = []
        for sh, sm in start_candidates:
            for eh, em in end_candidates:
                start_total = sh * 60 + sm
                end_total   = eh * 60 + em
                elapsed = end_total - start_total

                # Normalise into (-720, +720]
                while elapsed <= -720: elapsed += 1440
                while elapsed >   720: elapsed -= 1440

                spans.append(SpanCandidate(
                    start_str=self._fmt(sh, sm),
                    end_str=self._fmt(eh, em),
                    elapsed_minutes=elapsed,
                    elapsed_display=self._display(abs(elapsed)),
                    direction="Forward" if elapsed >= 0 else "Backward",
                    plausibility=self._plausibility(elapsed),
                ))

        # Sort: High plausibility first, then shortest abs duration
        ordering = {"High": 0, "Medium": 1, "Low": 2}
        spans.sort(key=lambda s: (ordering[s.plausibility], abs(s.elapsed_minutes)))

        # Unique by elapsed_minutes
        seen: set = set()
        unique: List[SpanCandidate] = []
        for s in spans:
            if s.elapsed_minutes not in seen:
                seen.add(s.elapsed_minutes)
                unique.append(s)

        notes = self._build_notes(unique, start_period, end_period)

        return ElapsedTimeResult(
            most_probable=unique[0],
            all_spans=unique[:6],
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    def _24h_candidates(
        self, hour: int, minute: int, period: Optional[str]
    ) -> List[tuple]:
        h12 = hour % 12
        if period == "AM": return [(h12, minute)]
        if period == "PM": return [(h12 + 12, minute)]
        return [(h12, minute), (h12 + 12, minute)]   # both possibilities

    def _fmt(self, h24: int, m: int) -> str:
        h12 = h24 % 12 or 12
        ap  = "AM" if h24 < 12 else "PM"
        return f"{h12}:{m:02d} {ap}"

    def _display(self, total_minutes: int) -> str:
        h = total_minutes // 60
        m = total_minutes % 60
        if h == 0:   return f"{m}m"
        if m == 0:   return f"{h}h"
        return f"{h}h {m}m"

    def _plausibility(self, elapsed: int) -> str:
        if 0 <= elapsed <= 360:   return "High"     # up to 6 h forward
        if 0 < elapsed <= 720:    return "Medium"   # 6-12 h forward
        return "Low"                                  # backward or > 12 h

    def _build_notes(
        self,
        spans: List[SpanCandidate],
        sp: Optional[str],
        ep: Optional[str],
    ) -> str:
        lines = []
        if not sp or not ep:
            lines.append(
                "AM/PM was not provided for one or both clocks, so multiple "
                "interpretations are listed. The most probable span assumes a "
                "short forward duration (≤ 6 hours)."
            )
        if spans and spans[0].direction == "Backward":
            lines.append("Note: the most probable span implies the clock moved backward.")
        if not lines:
            lines.append("AM/PM was provided; single unambiguous span computed.")
        return " ".join(lines)

    # ------------------------------------------------------------------ #
    def to_dict(self, result: ElapsedTimeResult) -> dict:
        def span_dict(s: SpanCandidate) -> dict:
            return {
                "from": s.start_str,
                "to": s.end_str,
                "elapsed_minutes": s.elapsed_minutes,
                "elapsed_display": s.elapsed_display,
                "direction": s.direction,
                "plausibility": s.plausibility,
            }
        return {
            "most_probable": span_dict(result.most_probable),
            "all_spans": [span_dict(s) for s in result.all_spans],
            "notes": result.notes,
        }


# Singleton
elapsed_calculator = ElapsedTimeCalculator()
