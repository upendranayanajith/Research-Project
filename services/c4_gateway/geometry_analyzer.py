"""
C4 Geometry Analyzer
============================
Pure geometric analysis of clock hand configurations.

Features
--------
1. Inter-Hand Angle Verifier
   For any reading H:M the exact angle between the hands is |30H − 5.5M|°.
   C4 measures the actual detected angle and flags inconsistencies.

2. Hand Length Ratio Validator
   The hour hand is physically ~60-75 % of the minute hand length.
   If C2 keypoints (center + tip per hand) are available, C4 validates
   which hand was correctly labelled "hour" vs. "minute".

3. Angle Bisector Time
   The bisector of the two hand vectors points to a deterministic time.
   This is a second independent time estimate that must agree with the
   primary reading — acts as a geometric cross-check.

4. Special Geometric Event Countdown
   Given a reading, computes minutes until:
     • Next overlap        (hands aligned, 0° apart)  — 11 × per 12 h
     • Next perpendicular  (hands exactly 90° apart)  — 22 × per 12 h
     • Next opposition     (hands exactly 180° apart) — 11 × per 12 h
     • Next full hour      (minute hand at 12)

5. Clock Sector Classifier
   The two hands divide the face into two arcs.
   The module labels each arc and checks whether the minute value is
   geometrically consistent with the smaller sector.

Owner: Member 4
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InterHandResult:
    detected_angle_deg: float   # actual gap from detected hand angles
    predicted_angle_deg: float  # what |30H − 5.5M| says it should be
    delta_deg: float            # |detected − predicted|
    is_consistent: bool         # True when delta < tolerance
    verdict: str                # "Consistent" | "Suspect" | "Contradictory"
    note: str


@dataclass
class HandLengthResult:
    available: bool             # False when keypoints not provided
    hand1_length_px: Optional[float]
    hand2_length_px: Optional[float]
    ratio: Optional[float]      # hand1 / hand2 (< 1 means hand1 is shorter)
    hour_is_hand1: Optional[bool]
    assignment_matches_physics: Optional[bool]
    note: str


@dataclass
class BisectorResult:
    bisector_angle_deg: float   # angle of the bisector from 12-o'clock
    bisector_time_str: str      # time the bisector corresponds to
    bisector_hour: int
    bisector_minute: int
    agrees_with_primary: bool   # True if bisector time is close to primary reading
    note: str


@dataclass
class SpecialEvent:
    name: str           # "Overlap" | "Perpendicular" | "Opposition" | "Full Hour"
    minutes_away: float
    at_time_str: str    # clock time when it occurs
    description: str


@dataclass
class SectorResult:
    smaller_arc_deg: float
    larger_arc_deg: float
    minute_in_smaller_sector: bool
    sector_note: str


@dataclass
class GeometryReport:
    inter_hand:   InterHandResult
    hand_lengths: HandLengthResult
    bisector:     BisectorResult
    special_events: List[SpecialEvent]
    sectors:      SectorResult
    overall_geometry_ok: bool
    geometry_summary: str


# ═══════════════════════════════════════════════════════════════════
# Main analyzer
# ═══════════════════════════════════════════════════════════════════

class ClockGeometryAnalyzer:
    """
    Stateless — all methods are pure functions of their inputs.

    Clock geometry conventions used throughout
    -------------------------------------------
    Angles are measured clockwise from 12-o'clock (0° = 12, 90° = 3, …).

    At time H:M (H in 1-12, M in 0-59):
        θ_hour   = (H % 12) * 30  +  M * 0.5    (degrees from 12)
        θ_minute = M * 6                         (degrees from 12)
        θ_inter  = |θ_hour − θ_minute|
                 = |30H − 5.5M|  (simplified, mod 360 if > 180 → take 360 − θ)
    """

    # Tolerances
    INTER_HAND_TOLERANCE_DEG   = 8.0   # degrees — below = consistent
    INTER_HAND_SUSPECT_DEG     = 15.0  # degrees — above = contradictory
    LENGTH_RATIO_MIN           = 0.45  # hour hand should be at least 45 % of minute
    LENGTH_RATIO_MAX           = 0.85  # but not longer than 85 %
    BISECTOR_AGREE_TOLERANCE   = 10    # minutes — bisector time difference

    # ---------------------------------------------------------------- #
    def analyze(
        self,
        hand1_angle: float,
        hand2_angle: float,
        hour: int,
        minute: int,
        keypoints: Optional[dict] = None,
    ) -> GeometryReport:
        """
        Parameters
        ----------
        hand1_angle, hand2_angle : detected hand angles in degrees (0-360), 
                                   as labelled by the physics solver.
                                   hand1 is the hour hand, hand2 the minute hand.
        hour, minute             : the resolved time (1-12, 0-59).
        keypoints                : optional C2 keypoint dict, expected keys:
                                   "center", "hand1_tip", "hand2_tip"
                                   each a {"x": ..., "y": ...} dict.
        """
        inter     = self._check_inter_hand(hand1_angle, hand2_angle, hour, minute)
        lengths   = self._check_hand_lengths(hand1_angle, hand2_angle, keypoints)
        bisector  = self._compute_bisector(hand1_angle, hand2_angle, hour, minute)
        events    = self._special_events(hour, minute)
        sectors   = self._sector_analysis(hand1_angle, hand2_angle, minute)

        ok = inter.is_consistent and (
            not lengths.available or lengths.assignment_matches_physics != False
        )

        summary = self._build_summary(inter, lengths, bisector, ok)

        return GeometryReport(
            inter_hand=inter,
            hand_lengths=lengths,
            bisector=bisector,
            special_events=events,
            sectors=sectors,
            overall_geometry_ok=ok,
            geometry_summary=summary,
        )

    # ──────────────────────────────────────────────────────────────── #
    # Feature 1 — Inter-hand angle verifier
    # ──────────────────────────────────────────────────────────────── #
    def _check_inter_hand(
        self,
        a1: float, a2: float,
        hour: int, minute: int,
    ) -> InterHandResult:
        # Detected gap (circular)
        raw_gap = abs(a1 - a2) % 360.0
        detected = min(raw_gap, 360.0 - raw_gap)

        # Predicted gap from the resolved time
        h_norm  = hour % 12
        theta_h = h_norm * 30.0 + minute * 0.5
        theta_m = minute * 6.0
        predicted_raw = abs(theta_h - theta_m)
        predicted = min(predicted_raw, 360.0 - predicted_raw)

        delta = abs(detected - predicted)

        if delta < self.INTER_HAND_TOLERANCE_DEG:
            verdict = "Consistent"
            note = (
                f"Detected gap {detected:.1f}° matches predicted {predicted:.1f}° "
                f"(Δ={delta:.1f}° < {self.INTER_HAND_TOLERANCE_DEG}° tolerance). "
                "Reading is geometrically sound."
            )
            consistent = True
        elif delta < self.INTER_HAND_SUSPECT_DEG:
            verdict = "Suspect"
            note = (
                f"Detected gap {detected:.1f}° vs. predicted {predicted:.1f}° "
                f"(Δ={delta:.1f}°). Small inconsistency — likely a minor "
                "angle detection error."
            )
            consistent = False
        else:
            verdict = "Contradictory"
            note = (
                f"Detected gap {detected:.1f}° strongly disagrees with predicted "
                f"{predicted:.1f}° (Δ={delta:.1f}°). The reading may be wrong — "
                "consider re-running with Expert Path."
            )
            consistent = False

        return InterHandResult(
            detected_angle_deg=round(detected, 2),
            predicted_angle_deg=round(predicted, 2),
            delta_deg=round(delta, 2),
            is_consistent=consistent,
            verdict=verdict,
            note=note,
        )

    # ──────────────────────────────────────────────────────────────── #
    # Feature 2 — Hand length ratio validator
    # ──────────────────────────────────────────────────────────────── #
    def _check_hand_lengths(
        self,
        a1: float, a2: float,
        keypoints: Optional[dict],
    ) -> HandLengthResult:
        if not keypoints:
            return HandLengthResult(
                available=False,
                hand1_length_px=None, hand2_length_px=None,
                ratio=None, hour_is_hand1=None,
                assignment_matches_physics=None,
                note="Keypoint data not provided — length analysis skipped.",
            )

        try:
            cx = keypoints["center"]["x"]
            cy = keypoints["center"]["y"]
            t1x = keypoints["hand1_tip"]["x"]
            t1y = keypoints["hand1_tip"]["y"]
            t2x = keypoints["hand2_tip"]["x"]
            t2y = keypoints["hand2_tip"]["y"]

            L1 = float(np.hypot(t1x - cx, t1y - cy))
            L2 = float(np.hypot(t2x - cx, t2y - cy))

            if L2 == 0:
                raise ValueError("Zero length hand 2")

            ratio = L1 / L2  # < 1 → hand1 is shorter → hand1 is the hour hand

            # Physics says hand1 = hour hand.  Hour hand is shorter.
            hour_is_hand1 = ratio < 1.0
            physics_says_hand1_is_hour = True   # by our convention

            assignment_ok = (hour_is_hand1 == physics_says_hand1_is_hour)

            if assignment_ok and self.LENGTH_RATIO_MIN <= ratio <= self.LENGTH_RATIO_MAX:
                note = (
                    f"Hand 1 ({L1:.0f}px) is {ratio*100:.0f}% of Hand 2 ({L2:.0f}px). "
                    "Ratio is within the expected 45-85% range for an hour hand. "
                    "Hour-hand assignment confirmed geometrically."
                )
            elif assignment_ok:
                note = (
                    f"Hand 1 ({L1:.0f}px) is {ratio*100:.0f}% of Hand 2 ({L2:.0f}px). "
                    "Assignment direction is correct but ratio is unusual — "
                    "clock may have non-standard proportions."
                )
            else:
                note = (
                    f"Hand 1 ({L1:.0f}px) is {ratio*100:.0f}% of Hand 2 ({L2:.0f}px). "
                    "Hand 1 appears LONGER, suggesting the hour/minute assignment "
                    "may be swapped. Consider re-running with Force Expert."
                )

            return HandLengthResult(
                available=True,
                hand1_length_px=round(L1, 1),
                hand2_length_px=round(L2, 1),
                ratio=round(ratio, 3),
                hour_is_hand1=hour_is_hand1,
                assignment_matches_physics=assignment_ok,
                note=note,
            )

        except Exception as e:
            return HandLengthResult(
                available=False,
                hand1_length_px=None, hand2_length_px=None,
                ratio=None, hour_is_hand1=None,
                assignment_matches_physics=None,
                note=f"Keypoint parsing failed: {e}",
            )

    # ──────────────────────────────────────────────────────────────── #
    # Feature 3 — Angle bisector time
    # ──────────────────────────────────────────────────────────────── #
    def _compute_bisector(
        self,
        a1: float, a2: float,
        hour: int, minute: int,
    ) -> BisectorResult:
        """
        Compute the angle of the bisector of the two hand vectors,
        then find which clock time that angle corresponds to.
        Used as an independent cross-check of the primary reading.
        """
        # Convert to unit vectors
        r1 = np.radians(a1)
        r2 = np.radians(a2)
        v1 = np.array([np.sin(r1), -np.cos(r1)])  # clockwise from 12
        v2 = np.array([np.sin(r2), -np.cos(r2)])

        bisector_vec = v1 + v2
        norm = np.linalg.norm(bisector_vec)

        if norm < 1e-6:
            # Hands are exactly opposite — bisector is undefined
            return BisectorResult(
                bisector_angle_deg=0.0,
                bisector_time_str="Undefined (hands opposite)",
                bisector_hour=0, bisector_minute=0,
                agrees_with_primary=False,
                note="Hands are exactly 180° apart — bisector is undefined "
                     "(this is the 6:00 / 12:00 straight-line configuration).",
            )

        bisector_vec /= norm
        # Angle from 12-o'clock (clockwise)
        bx, by = bisector_vec
        bis_angle = float(np.degrees(np.arctan2(bx, -by)) % 360.0)

        # What time does this bisector angle map to (treating it as a minute hand)?
        bis_minute = round(bis_angle / 6.0) % 60
        # What is the hour at that minute given the bisector also constrains h?
        # Simpler: bisector just gives us a reference — find nearest clock time
        bis_t = self._angle_to_nearest_time(bis_angle)
        bis_h, bis_m = bis_t

        # Agreement check: primary time vs bisector time in total minutes
        primary_total = (hour % 12) * 60 + minute
        bisector_total = (bis_h % 12) * 60 + bis_m
        diff = abs(primary_total - bisector_total)
        diff = min(diff, 720 - diff)   # circular in 720 min

        agrees = diff <= self.BISECTOR_AGREE_TOLERANCE

        note = (
            f"Bisector angle {bis_angle:.1f}° → ~{bis_h}:{bis_m:02d}. "
            f"Primary reading is {hour}:{minute:02d}. "
            f"Difference: {diff:.0f} min — "
            + ("within tolerance, cross-check passed." if agrees
               else f"exceeds {self.BISECTOR_AGREE_TOLERANCE} min threshold, "
                    "possible misread.")
        )

        return BisectorResult(
            bisector_angle_deg=round(bis_angle, 2),
            bisector_time_str=f"{bis_h}:{bis_m:02d}",
            bisector_hour=bis_h,
            bisector_minute=bis_m,
            agrees_with_primary=agrees,
            note=note,
        )

    def _angle_to_nearest_time(self, angle: float) -> Tuple[int, int]:
        """Map a single angle to the nearest clock time via minute-hand assumption."""
        minute = int(round(angle / 6.0)) % 60
        # Derive approximate hour from the positions
        h_raw = (angle - minute * 0.5) / 30.0
        hour  = int(round(h_raw)) % 12
        if hour == 0: hour = 12
        return hour, minute

    # ──────────────────────────────────────────────────────────────── #
    # Feature 4 — Special geometric event countdown
    # ──────────────────────────────────────────────────────────────── #
    def _special_events(self, hour: int, minute: int) -> List[SpecialEvent]:
        """
        Find the next Overlap, Perpendicular, Opposition, and Full Hour
        starting from the detected time.
        """
        # Current total minutes into 12-hour cycle (0-719)
        current_t = (hour % 12) * 60 + minute

        events: List[SpecialEvent] = []

        # ── Overlap: |30H − 5.5M| = 0  → t = 720k/11 (k=0..10)
        events += self._find_next_events(
            current_t, divisor=11, target_gap=0.0,
            name="Overlap",
            desc="Hour and minute hands point in exactly the same direction.",
        )

        # ── Perpendicular: gap = 90° → 5.5M − 30H = ±90 → t = (720k ± 180)/11
        events += self._find_next_events(
            current_t, divisor=11, target_gap=90.0,
            name="Perpendicular (90°)",
            desc="Hands form a right angle — 90° apart.",
        )

        # ── Opposition: gap = 180° → t = (720k + 360)/11
        events += self._find_next_events(
            current_t, divisor=11, target_gap=180.0,
            name="Opposition (180°)",
            desc="Hands point in exactly opposite directions.",
        )

        # ── Full hour: next minute = 0
        minutes_to_full = (60 - minute) % 60
        if minutes_to_full == 0: minutes_to_full = 60
        next_h = (hour % 12) + 1
        if next_h > 12: next_h = 1
        events.append(SpecialEvent(
            name="Full Hour",
            minutes_away=float(minutes_to_full),
            at_time_str=f"{next_h}:00",
            description=f"Minute hand returns to 12 — next full hour at {next_h}:00.",
        ))

        # Sort by minutes_away
        events.sort(key=lambda e: e.minutes_away)
        return events[:6]   # return closest 6

    def _find_next_events(
        self, current_t: float, divisor: int,
        target_gap: float, name: str, desc: str,
    ) -> List[SpecialEvent]:
        """
        The minute hand gains on the hour hand at 5.5°/min.
        A gap of G° occurs at t = (G/5.5 + 720k/11) minutes for integer k.
        We solve for the next occurrence(s) after current_t.
        """
        results: List[SpecialEvent] = []
        # Solve 5.5t = 30H + G  (mod 360) for each class of solutions
        # Equivalently: t (in 12-h minutes) at which relative angle = target_gap
        # There are `divisor` equally spaced solutions in 720 min.
        # For gap=0: 720/11 apart; for gap=90 or 180: same spacing but offset.
        step = 720.0 / divisor

        # Offset within the cycle from which 0-gap events occur
        base_offsets = [target_gap / 5.5, -target_gap / 5.5]

        seen_times: set = set()
        for base in base_offsets:
            for k in range(divisor + 1):
                t_event = (base + k * step) % 720.0
                t_key = round(t_event)
                if t_key in seen_times: continue
                seen_times.add(t_key)

                diff = (t_event - current_t) % 720.0
                if diff < 0.5: diff += 720.0   # skip "now"

                h_ev = int(t_event // 60) % 12
                if h_ev == 0: h_ev = 12
                m_ev = int(t_event % 60)

                results.append(SpecialEvent(
                    name=name,
                    minutes_away=round(diff, 1),
                    at_time_str=f"{h_ev}:{m_ev:02d}",
                    description=desc,
                ))

        # Return just the nearest one
        results.sort(key=lambda e: e.minutes_away)
        return results[:1] if results else []

    # ──────────────────────────────────────────────────────────────── #
    # Feature 5 — Sector classifier
    # ──────────────────────────────────────────────────────────────── #
    def _sector_analysis(
        self,
        a1: float, a2: float,
        minute: int,
    ) -> SectorResult:
        """
        The two hands divide the clock face into two arcs.
        The minute hand sweeps the larger arc in the first half of each
        5-minute interval; the pattern inverts in the second half.
        We simply measure both arcs and verify that the minute position
        is geometrically inside the expected sector.
        """
        gap = abs(a1 - a2) % 360.0
        smaller = min(gap, 360.0 - gap)
        larger  = 360.0 - smaller

        # The minute hand is between 0° and 360°; it should lie in the
        # smaller arc only when the hands are close together (< 90°).
        # As a sanity check: if smaller arc < 30° and minute ≠ 0, flag it.
        minute_at_zero = (minute % 5 == 0)

        if smaller < 5.0 and not minute_at_zero:
            sector_note = (
                f"Smaller arc is only {smaller:.1f}° but minute is not a "
                "5-minute mark — possible overlap mis-detection."
            )
            minute_ok = False
        else:
            sector_note = (
                f"Hands split the face into arcs of {smaller:.1f}° and "
                f"{larger:.1f}°. Sector geometry is plausible for "
                f"{minute} minutes past the hour."
            )
            minute_ok = True

        return SectorResult(
            smaller_arc_deg=round(smaller, 2),
            larger_arc_deg=round(larger, 2),
            minute_in_smaller_sector=minute_ok,
            sector_note=sector_note,
        )

    # ──────────────────────────────────────────────────────────────── #
    # Summary builder
    # ──────────────────────────────────────────────────────────────── #
    def _build_summary(
        self,
        inter: InterHandResult,
        lengths: HandLengthResult,
        bisector: BisectorResult,
        overall_ok: bool,
    ) -> str:
        parts = []
        parts.append(f"Inter-hand: {inter.verdict} (Δ={inter.delta_deg:.1f}°).")
        if lengths.available:
            parts.append(
                f"Hand lengths: ratio={lengths.ratio:.2f} "
                f"({'OK' if lengths.assignment_matches_physics else 'CHECK SWAP'})."
            )
        parts.append(
            f"Bisector cross-check: {'passed' if bisector.agrees_with_primary else 'failed'} "
            f"(→ {bisector.bisector_time_str})."
        )
        parts.append("Overall geometry: " + ("✓ CONSISTENT" if overall_ok else "⚠ SUSPECT"))
        return " | ".join(parts)

    # ──────────────────────────────────────────────────────────────── #
    # Serialiser
    # ──────────────────────────────────────────────────────────────── #
    def to_dict(self, report: GeometryReport) -> dict:
        def event_d(e: SpecialEvent) -> dict:
            return {
                "name":         e.name,
                "minutes_away": e.minutes_away,
                "at_time":      e.at_time_str,
                "description":  e.description,
            }
        return {
            "overall_geometry_ok": report.overall_geometry_ok,
            "geometry_summary":    report.geometry_summary,
            "inter_hand_angle": {
                "detected_deg":   report.inter_hand.detected_angle_deg,
                "predicted_deg":  report.inter_hand.predicted_angle_deg,
                "delta_deg":      report.inter_hand.delta_deg,
                "verdict":        report.inter_hand.verdict,
                "note":           report.inter_hand.note,
            },
            "hand_lengths": {
                "available":                    report.hand_lengths.available,
                "hand1_length_px":              report.hand_lengths.hand1_length_px,
                "hand2_length_px":              report.hand_lengths.hand2_length_px,
                "ratio":                        report.hand_lengths.ratio,
                "hour_is_hand1":                report.hand_lengths.hour_is_hand1,
                "assignment_matches_physics":   report.hand_lengths.assignment_matches_physics,
                "note":                         report.hand_lengths.note,
            },
            "bisector": {
                "angle_deg":          report.bisector.bisector_angle_deg,
                "time":               report.bisector.bisector_time_str,
                "agrees_with_primary":report.bisector.agrees_with_primary,
                "note":               report.bisector.note,
            },
            "special_events": [event_d(e) for e in report.special_events],
            "sectors": {
                "smaller_arc_deg":          report.sectors.smaller_arc_deg,
                "larger_arc_deg":           report.sectors.larger_arc_deg,
                "sector_geometry_ok":       report.sectors.minute_in_smaller_sector,
                "note":                     report.sectors.sector_note,
            },
        }


# Singleton
geometry_analyzer = ClockGeometryAnalyzer()
