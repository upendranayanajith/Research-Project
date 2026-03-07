"""
gauge_reader.py — Core Gauge Reading Logic for C4 Gateway
Converts the needle angle (from C3) into a real-world measurement value.

Data flow:
    C1 → cropped gauge image
    C2 → needle skeleton / keypoints
    C3 → needle angle in degrees  ← INPUT HERE
    C4 → this module converts angle → reading value  ← YOUR PART
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

from gauge_config import GaugeConfig, get_gauge_config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Output data structure
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GaugeReading:
    """
    Final output produced by C4 for a single gauge.
    """
    # ── Core result ──────────────────────────────────────────────────────────
    value: float                  # the calculated measurement value
    unit: str                     # e.g. "PSI", "°C", "km/h"
    gauge_type: str               # gauge type key used for calculation

    # ── Input from C3 ────────────────────────────────────────────────────────
    raw_angle_deg: float          # angle received from C3

    # ── Derived / meta ───────────────────────────────────────────────────────
    percentage: float             # 0–100 % of full scale
    confidence: float             # 0.0–1.0 confidence score
    status: str                   # "normal" | "warning" | "danger" | "out_of_range"
    status_detail: str            # human-readable explanation of status

    # ── Source info ──────────────────────────────────────────────────────────
    source: str = "image"         # "image" | "cctv_stream"
    frame_id: Optional[int] = None  # for CCTV streams, which frame

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "gauge_type": self.gauge_type,
            "raw_angle_deg": self.raw_angle_deg,
            "percentage": self.percentage,
            "confidence": self.confidence,
            "status": self.status,
            "status_detail": self.status_detail,
            "source": self.source,
            "frame_id": self.frame_id,
            "display": f"{self.value} {self.unit}",
        }

    def __str__(self) -> str:
        return (
            f"[{self.gauge_type}] {self.value} {self.unit} "
            f"({self.percentage:.1f}% of scale) — {self.status}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main conversion logic
# ──────────────────────────────────────────────────────────────────────────────

class GaugeReader:
    """
    Converts a needle angle (degrees, from C3) to a real-world reading
    using the GaugeConfig for the detected gauge type.

    Usage:
        reader = GaugeReader()
        result = reader.calculate(angle_deg=45.0, gauge_type="pressure_psi")
        print(result)  # [pressure_psi] 75.0 PSI (75.0% of scale) — normal
    """

    def __init__(self):
        self._history: list[GaugeReading] = []   # for smoothing CCTV streams

    # ── Public API ────────────────────────────────────────────────────────────

    def calculate(
        self,
        angle_deg: float,
        gauge_type: str = "generic_0_100",
        confidence: float = 1.0,
        source: str = "image",
        frame_id: Optional[int] = None,
        smooth: bool = False,
    ) -> GaugeReading:
        """
        Main method: angle (degrees) → GaugeReading.

        Args:
            angle_deg:   Needle angle in degrees from C3.
                         Convention: 0° = 12 o'clock, + = clockwise.
            gauge_type:  Key into GAUGE_CATALOGUE (see gauge_config.py).
            confidence:  C3's confidence score (0–1) for the angle.
            source:      "image" or "cctv_stream".
            frame_id:    Frame number for CCTV streams.
            smooth:      If True, apply temporal smoothing (for video streams).

        Returns:
            GaugeReading with all fields populated.
        """
        config = get_gauge_config(gauge_type)

        # 1. Convert angle → normalised position (0.0 … 1.0)
        position = self._angle_to_position(angle_deg, config)

        # 2. Map position → value on the gauge scale
        raw_value = self._position_to_value(position, config)

        # 3. Clamp to physical limits
        clamped_value = max(config.min_value, min(config.max_value, raw_value))

        # 4. Round to gauge precision
        value = round(clamped_value, config.decimal_places)

        # 5. Percentage of full scale
        percentage = round(
            (value - config.min_value) / max(config.value_span, 1e-9) * 100, 1
        )

        # 6. Determine status
        status, status_detail = self._evaluate_status(value, config)

        # 7. Adjust confidence if angle is out of the gauge's sweep
        if position < 0.0 or position > 1.0:
            confidence *= 0.5
            status = "out_of_range"
            status_detail = (
                f"Angle {angle_deg:.1f}° is outside the gauge sweep "
                f"({config.start_angle}° to {config.end_angle}°)"
            )

        reading = GaugeReading(
            value=value,
            unit=config.unit,
            gauge_type=gauge_type,
            raw_angle_deg=angle_deg,
            percentage=percentage,
            confidence=round(confidence, 3),
            status=status,
            status_detail=status_detail,
            source=source,
            frame_id=frame_id,
        )

        # 8. Optional temporal smoothing for live streams
        if smooth and source == "cctv_stream":
            reading = self._smooth_reading(reading, config)

        self._history.append(reading)
        logger.info("GaugeReading: %s", reading)
        return reading

    def calculate_from_c3_output(
        self,
        c3_output: dict,
        gauge_type: str = "generic_0_100",
        source: str = "image",
        frame_id: Optional[int] = None,
        smooth: bool = False,
    ) -> GaugeReading:
        """
        Convenience wrapper that accepts the raw dict that C3 produces.

        Expected C3 output format (adjust keys to match your C3 actual output):
            {
                "angle": 45.0,          # needle angle in degrees
                "confidence": 0.92,     # C3's confidence
                ...                     # other C3 fields are ignored by C4
            }
        """
        angle_deg = float(c3_output.get("angle", 0.0))
        confidence = float(c3_output.get("confidence", 1.0))
        return self.calculate(
            angle_deg=angle_deg,
            gauge_type=gauge_type,
            confidence=confidence,
            source=source,
            frame_id=frame_id,
            smooth=smooth,
        )

    def clear_history(self):
        """Reset the smoothing history (call between different gauge sessions)."""
        self._history.clear()

    # ── Internal maths ────────────────────────────────────────────────────────

    @staticmethod
    def _angle_to_position(angle_deg: float, config: GaugeConfig) -> float:
        """
        Map a needle angle to a normalised position in [0, 1].

        0.0  = needle pointing at minimum value
        1.0  = needle pointing at maximum value
        Values outside [0, 1] mean the needle is beyond the gauge range.

        The formula handles both clockwise and counter-clockwise gauges.
        """
        span = config.end_angle - config.start_angle      # signed sweep
        if abs(span) < 1e-9:
            return 0.0
        return (angle_deg - config.start_angle) / span

    @staticmethod
    def _position_to_value(position: float, config: GaugeConfig) -> float:
        """
        Linear interpolation: position [0,1] → value [min_value, max_value].
        """
        return config.min_value + position * config.value_span

    @staticmethod
    def _evaluate_status(value: float, config: GaugeConfig) -> tuple[str, str]:
        """
        Compare value against configured thresholds and return
        (status_string, detail_string).
        """
        # Danger checks first (higher priority)
        if config.danger_low is not None and value < config.danger_low:
            return "danger", f"Value {value} {config.unit} is below danger threshold ({config.danger_low})"
        if config.danger_high is not None and value > config.danger_high:
            return "danger", f"Value {value} {config.unit} exceeds danger threshold ({config.danger_high})"

        # Warning checks
        if config.warning_low is not None and value < config.warning_low:
            return "warning", f"Value {value} {config.unit} is below warning threshold ({config.warning_low})"
        if config.warning_high is not None and value > config.warning_high:
            return "warning", f"Value {value} {config.unit} exceeds warning threshold ({config.warning_high})"

        return "normal", f"Value {value} {config.unit} is within normal range"

    def _smooth_reading(self, reading: GaugeReading, config: GaugeConfig, window: int = 5) -> GaugeReading:
        """
        Apply a simple weighted moving average over the last `window` frames.
        More recent readings get higher weight.
        Used for CCTV streams to reduce jitter.
        """
        recent = [r for r in self._history[-window:] if r.gauge_type == reading.gauge_type]
        if not recent:
            return reading

        # Weights: older = 1, newest = len(recent)+1
        weights = list(range(1, len(recent) + 2))   # +1 for the new reading
        values = [r.value for r in recent] + [reading.value]

        smoothed = sum(v * w for v, w in zip(values, weights)) / sum(weights)
        smoothed = round(max(config.min_value, min(config.max_value, smoothed)), config.decimal_places)

        # Rebuild with smoothed value
        percentage = round(
            (smoothed - config.min_value) / max(config.value_span, 1e-9) * 100, 1
        )
        status, status_detail = self._evaluate_status(smoothed, config)

        return GaugeReading(
            value=smoothed,
            unit=reading.unit,
            gauge_type=reading.gauge_type,
            raw_angle_deg=reading.raw_angle_deg,
            percentage=percentage,
            confidence=reading.confidence,
            status=status,
            status_detail=status_detail,
            source=reading.source,
            frame_id=reading.frame_id,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reader = GaugeReader()

    tests = [
        # (angle, gauge_type, expected_approx_value)
        (-135, "pressure_psi",       0),      # min position
        (   0, "pressure_psi",      50),      # midpoint
        ( 135, "pressure_psi",     100),      # max position
        (-135, "temperature_celsius", 0),
        (  45, "temperature_celsius", 70),
        ( 135, "temperature_celsius", 120),
        (   0, "fuel_gauge",          50),
    ]

    print("\n── GaugeReader self-test ──────────────────────────────")
    for angle, gtype, expected in tests:
        r = reader.calculate(angle_deg=angle, gauge_type=gtype)
        ok = "✓" if abs(r.value - expected) < 1.5 else "✗"
        print(f"  {ok}  angle={angle:+6.1f}°  {gtype:<25}  → {r.value:7.2f} {r.unit}  (expected ~{expected})")
    print()