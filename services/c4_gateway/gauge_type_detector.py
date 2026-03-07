"""
gauge_type_detector.py — Automatic Gauge Type Identification for C4 Gateway

Determines WHICH gauge type a gauge image belongs to so the correct
GaugeConfig is used for the angle→value calculation.

Detection hierarchy (in order of reliability):
    1. Explicit override — caller passes gauge_type directly
    2. C1 metadata      — C1 may tag the image with a detected class
    3. Scale-text OCR   — read min/max numbers from the cropped image
    4. Visual heuristics — colour, aspect, dial shape
    5. Fallback         — "generic_0_100"
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Simple keyword/label mapping from C1 detection class names ────────────────

_C1_CLASS_TO_GAUGE_TYPE: dict[str, str] = {
    # C1 label           → gauge_config key
    "pressure_gauge":     "pressure_psi",
    "pressure_meter":     "pressure_psi",
    "manometer":          "pressure_bar",
    "thermometer":        "temperature_celsius",
    "temp_gauge":         "temperature_celsius",
    "temperature_gauge":  "temperature_celsius",
    "speedometer":        "speedometer_kmh",
    "tachometer":         "speedometer_kmh",
    "voltmeter":          "voltmeter",
    "ammeter":            "ammeter",
    "fuel_gauge":         "fuel_gauge",
    "flow_meter":         "flow_lpm",
    "flow_gauge":         "flow_lpm",
}

# ── Unit-string → gauge_config key ────────────────────────────────────────────

_UNIT_TEXT_TO_GAUGE_TYPE: dict[str, str] = {
    "psi":   "pressure_psi",
    "bar":   "pressure_bar",
    "kpa":   "pressure_kpa",
    "mpa":   "pressure_kpa",
    "°c":    "temperature_celsius",
    "c":     "temperature_celsius",
    "°f":    "temperature_fahrenheit",
    "f":     "temperature_fahrenheit",
    "km/h":  "speedometer_kmh",
    "kmh":   "speedometer_kmh",
    "mph":   "speedometer_mph",
    "v":     "voltmeter",
    "volt":  "voltmeter",
    "a":     "ammeter",
    "amp":   "ammeter",
    "l/min": "flow_lpm",
    "lpm":   "flow_lpm",
}


class GaugeTypeDetector:
    """
    Determines the gauge type (key into GAUGE_CATALOGUE) from available signals.

    C4 calls this BEFORE GaugeReader so it knows which config to use.
    """

    def detect(
        self,
        *,
        gauge_type_override: Optional[str] = None,
        c1_metadata: Optional[dict] = None,
        ocr_text: Optional[str] = None,
        min_value_hint: Optional[float] = None,
        max_value_hint: Optional[float] = None,
    ) -> str:
        """
        Determine gauge type from available inputs.

        Args:
            gauge_type_override:  If the caller already knows the type, pass it here.
                                  This short-circuits all detection logic.
            c1_metadata:          Dict from C1 component, may contain 'class', 'label',
                                  'unit', 'confidence', etc.
            ocr_text:             Raw text read from the gauge face image (if available).
            min_value_hint:       If scale min is known from OCR/metadata.
            max_value_hint:       If scale max is known from OCR/metadata.

        Returns:
            A gauge type key string (always valid — falls back to 'generic_0_100').
        """

        # ── 1. Explicit override ──────────────────────────────────────────────
        if gauge_type_override:
            logger.info("GaugeTypeDetector: using override '%s'", gauge_type_override)
            return gauge_type_override

        # ── 2. C1 metadata ────────────────────────────────────────────────────
        if c1_metadata:
            result = self._from_c1_metadata(c1_metadata)
            if result:
                logger.info("GaugeTypeDetector: from C1 metadata → '%s'", result)
                return result

        # ── 3. OCR text on gauge face ─────────────────────────────────────────
        if ocr_text:
            result = self._from_ocr_text(ocr_text)
            if result:
                logger.info("GaugeTypeDetector: from OCR text → '%s'", result)
                return result

        # ── 4. Min/max value hints (e.g. from OCR numbers only) ───────────────
        if min_value_hint is not None or max_value_hint is not None:
            result = self._from_value_hints(min_value_hint, max_value_hint)
            if result:
                logger.info("GaugeTypeDetector: from value hints → '%s'", result)
                return result

        # ── 5. Fallback ───────────────────────────────────────────────────────
        logger.warning("GaugeTypeDetector: could not identify gauge type, using fallback")
        return "generic_0_100"

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _from_c1_metadata(meta: dict) -> Optional[str]:
        """Extract gauge type from C1 output metadata."""
        # Try direct 'gauge_type' field (if C1 already classified it)
        if "gauge_type" in meta:
            return str(meta["gauge_type"])

        # Try 'class' or 'label' field
        for key in ("class", "label", "class_name"):
            label = str(meta.get(key, "")).lower().strip().replace(" ", "_")
            if label in _C1_CLASS_TO_GAUGE_TYPE:
                return _C1_CLASS_TO_GAUGE_TYPE[label]

        # Try unit string from C1
        unit = str(meta.get("unit", "")).lower().strip()
        if unit in _UNIT_TEXT_TO_GAUGE_TYPE:
            return _UNIT_TEXT_TO_GAUGE_TYPE[unit]

        return None

    @staticmethod
    def _from_ocr_text(text: str) -> Optional[str]:
        """
        Look for unit strings or keywords in OCR text from the gauge face.
        Checks unit tokens first, then class keywords.
        """
        lower = text.lower()

        # Check unit tokens
        for token, gauge_type in _UNIT_TEXT_TO_GAUGE_TYPE.items():
            # word-boundary match to avoid false positives (e.g. "A" inside "bar")
            if re.search(r'\b' + re.escape(token) + r'\b', lower):
                return gauge_type

        # Check class keywords
        for keyword, gauge_type in _C1_CLASS_TO_GAUGE_TYPE.items():
            if keyword.replace("_", " ") in lower:
                return gauge_type

        return None

    @staticmethod
    def _from_value_hints(
        min_v: Optional[float],
        max_v: Optional[float],
    ) -> Optional[str]:
        """
        Very rough heuristic: guess gauge type from numeric scale range.
        Only used as last resort before fallback.
        """
        if max_v is None:
            return None

        # Typical ranges as heuristics
        if 0 <= (min_v or 0) < 5 and 50 <= max_v <= 150:
            return "pressure_psi"
        if 0 <= (min_v or 0) < 2 and 5 <= max_v <= 15:
            return "pressure_bar"
        if 0 <= (min_v or 0) < 20 and 100 <= max_v <= 150:
            return "temperature_celsius"
        if 30 <= (min_v or 30) <= 35 and 200 <= max_v <= 260:
            return "temperature_fahrenheit"
        if 0 <= (min_v or 0) < 5 and 180 <= max_v <= 260:
            return "speedometer_kmh"
        if 0 <= (min_v or 0) < 5 and 120 <= max_v <= 160:
            return "speedometer_mph"

        return None


# ── Convenience singleton ─────────────────────────────────────────────────────

_detector = GaugeTypeDetector()


def detect_gauge_type(
    gauge_type_override: Optional[str] = None,
    c1_metadata: Optional[dict] = None,
    ocr_text: Optional[str] = None,
    min_value_hint: Optional[float] = None,
    max_value_hint: Optional[float] = None,
) -> str:
    """Module-level convenience wrapper around GaugeTypeDetector.detect()."""
    return _detector.detect(
        gauge_type_override=gauge_type_override,
        c1_metadata=c1_metadata,
        ocr_text=ocr_text,
        min_value_hint=min_value_hint,
        max_value_hint=max_value_hint,
    )


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ({"class": "pressure_gauge"},           None,           "pressure_psi"),
        ({"unit": "°C"},                         None,           "temperature_celsius"),
        (None,                                   "MAX 100 PSI",  "pressure_psi"),
        (None,                                   "0-120 °C",     "temperature_celsius"),
        (None,                                   "0-10 bar",     "pressure_bar"),
        ({},                                     "",             "generic_0_100"),
    ]

    print("\n── GaugeTypeDetector self-test ──────────────────────────")
    for meta, ocr, expected in tests:
        result = detect_gauge_type(c1_metadata=meta, ocr_text=ocr)
        ok = "✓" if result == expected else "✗"
        print(f"  {ok}  meta={str(meta):<30}  ocr={str(ocr):<20}  → {result}  (expected {expected})")
    print()