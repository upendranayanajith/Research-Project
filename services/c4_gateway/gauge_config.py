"""
gauge_config.py — Gauge Type Registry for C4 Gateway
Defines all supported gauge types with their scale parameters.
C4 uses this to map angles → real-world readings.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class GaugeConfig:
    """
    Describes a gauge's physical scale so C4 can convert
    a needle angle (degrees) into a real-world value.

    Angle convention (same as C3 output):
        0°   = 12 o'clock (straight up)
        +    = clockwise
        -    = counter-clockwise

    Example — a typical pressure gauge:
        start_angle = -135   (needle at 7 o'clock = minimum)
        end_angle   = +135   (needle at 5 o'clock = maximum)
        min_value   = 0
        max_value   = 100
        unit        = "PSI"
    """
    gauge_type: str          # human label, e.g. "pressure", "temperature"
    start_angle: float       # angle (deg) that corresponds to min_value
    end_angle: float         # angle (deg) that corresponds to max_value
    min_value: float         # lowest reading on the scale
    max_value: float         # highest reading on the scale
    unit: str                # measurement unit string, e.g. "PSI", "°C"
    description: str = ""    # optional human-readable description

    # Optional danger / warning thresholds for reporting
    warning_low: Optional[float] = None
    warning_high: Optional[float] = None
    danger_low: Optional[float] = None
    danger_high: Optional[float] = None

    # Precision: how many decimal places to round the final reading
    decimal_places: int = 1

    @property
    def angular_span(self) -> float:
        """Total sweep of the gauge in degrees."""
        return abs(self.end_angle - self.start_angle)

    @property
    def value_span(self) -> float:
        """Total value range of the gauge."""
        return self.max_value - self.min_value


# ──────────────────────────────────────────────────────────────────────────────
# BUILT-IN GAUGE CATALOGUE
# Add / modify entries here to support new gauge types.
# ──────────────────────────────────────────────────────────────────────────────

GAUGE_CATALOGUE: dict[str, GaugeConfig] = {

    # ── Pressure gauges ───────────────────────────────────────────────────────
    "pressure_psi": GaugeConfig(
        gauge_type="pressure_psi",
        start_angle=-135, end_angle=135,
        min_value=0, max_value=100,
        unit="PSI",
        description="Standard 0–100 PSI pressure gauge",
        warning_high=80, danger_high=95,
        decimal_places=1,
    ),
    "pressure_bar": GaugeConfig(
        gauge_type="pressure_bar",
        start_angle=-135, end_angle=135,
        min_value=0, max_value=10,
        unit="bar",
        description="Standard 0–10 bar pressure gauge",
        warning_high=8, danger_high=9.5,
        decimal_places=2,
    ),
    "pressure_kpa": GaugeConfig(
        gauge_type="pressure_kpa",
        start_angle=-135, end_angle=135,
        min_value=0, max_value=700,
        unit="kPa",
        description="0–700 kPa pressure gauge",
        warning_high=560, danger_high=665,
        decimal_places=0,
    ),

    # ── Temperature gauges ────────────────────────────────────────────────────
    "temperature_celsius": GaugeConfig(
        gauge_type="temperature_celsius",
        start_angle=-135, end_angle=135,
        min_value=0, max_value=120,
        unit="°C",
        description="0–120 °C temperature gauge",
        warning_high=90, danger_high=110,
        decimal_places=1,
    ),
    "temperature_fahrenheit": GaugeConfig(
        gauge_type="temperature_fahrenheit",
        start_angle=-135, end_angle=135,
        min_value=32, max_value=250,
        unit="°F",
        description="32–250 °F temperature gauge",
        warning_high=200, danger_high=230,
        decimal_places=1,
    ),

    # ── Flow / speed gauges ───────────────────────────────────────────────────
    "flow_lpm": GaugeConfig(
        gauge_type="flow_lpm",
        start_angle=-135, end_angle=135,
        min_value=0, max_value=200,
        unit="L/min",
        description="0–200 L/min flow gauge",
        decimal_places=1,
    ),
    "speedometer_kmh": GaugeConfig(
        gauge_type="speedometer_kmh",
        start_angle=-135, end_angle=135,
        min_value=0, max_value=220,
        unit="km/h",
        description="Vehicle speedometer 0–220 km/h",
        warning_high=180,
        decimal_places=0,
    ),
    "speedometer_mph": GaugeConfig(
        gauge_type="speedometer_mph",
        start_angle=-135, end_angle=135,
        min_value=0, max_value=140,
        unit="mph",
        description="Vehicle speedometer 0–140 mph",
        warning_high=110,
        decimal_places=0,
    ),

    # ── Voltage / current ─────────────────────────────────────────────────────
    "voltmeter": GaugeConfig(
        gauge_type="voltmeter",
        start_angle=-90, end_angle=90,
        min_value=0, max_value=300,
        unit="V",
        description="0–300 V voltmeter",
        warning_low=210, warning_high=250,
        danger_low=180, danger_high=260,
        decimal_places=1,
    ),
    "ammeter": GaugeConfig(
        gauge_type="ammeter",
        start_angle=-90, end_angle=90,
        min_value=0, max_value=10,
        unit="A",
        description="0–10 A ammeter",
        decimal_places=2,
    ),

    # ── Fuel / level gauges ───────────────────────────────────────────────────
    "fuel_gauge": GaugeConfig(
        gauge_type="fuel_gauge",
        start_angle=-90, end_angle=90,
        min_value=0, max_value=100,
        unit="%",
        description="Fuel / liquid level gauge 0–100 %",
        warning_low=15, danger_low=5,
        decimal_places=0,
    ),

    # ── Generic / unknown ─────────────────────────────────────────────────────
    "generic_0_100": GaugeConfig(
        gauge_type="generic_0_100",
        start_angle=-135, end_angle=135,
        min_value=0, max_value=100,
        unit="%",
        description="Generic 0–100 % gauge (fallback)",
        decimal_places=1,
    ),
}


def get_gauge_config(gauge_type: str) -> GaugeConfig:
    """
    Retrieve a GaugeConfig by type key.
    Falls back to 'generic_0_100' if not found.
    """
    return GAUGE_CATALOGUE.get(gauge_type, GAUGE_CATALOGUE["generic_0_100"])


def register_custom_gauge(config: GaugeConfig) -> None:
    """
    Register a new gauge type at runtime.
    Useful when C1 detects a gauge with known markings.
    """
    GAUGE_CATALOGUE[config.gauge_type] = config


def list_gauge_types() -> list[str]:
    """Return all registered gauge type keys."""
    return list(GAUGE_CATALOGUE.keys())