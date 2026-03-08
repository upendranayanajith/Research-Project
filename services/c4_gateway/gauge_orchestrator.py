"""
gauge_orchestrator.py — C4 Gateway: Gauge Reading Orchestrator

This is the TOP-LEVEL entry point for ALL gauge reading requests.
It wires together:
    - gauge_type_detector  → figures out WHICH gauge
    - gauge_reader         → converts angle → value
    - GaugeReading output  → structured result sent back to the API / caller

Supports TWO input modes:
    1. Uploaded image  — single call, one result
    2. Live CCTV stream — continuous frames, smoothed results

Typical call (from main.py or your API handler):
    from gauge_orchestrator import GaugeOrchestrator

    orch = GaugeOrchestrator()

    # From uploaded image:
    result = orch.process_image(c3_output={"angle": 45.0, "confidence": 0.93},
                                 c1_metadata={"class": "pressure_gauge"})
    print(result)

    # From CCTV frame:
    result = orch.process_stream_frame(c3_output={"angle": 45.0, "confidence": 0.93},
                                        c1_metadata={"unit": "PSI"},
                                        frame_id=1042)
    print(result)
"""

import logging
from typing import Optional

from .gauge_config import GaugeConfig, get_gauge_config, register_custom_gauge
from .gauge_reader import GaugeReader, GaugeReading
from .gauge_type_detector import detect_gauge_type

logger = logging.getLogger(__name__)


class GaugeOrchestrator:
    """
    Central coordinator for C4's gauge reading pipeline.

    One instance can handle both uploaded images and live CCTV streams.
    For streams, it maintains a small history per gauge to smooth readings.
    """

    def __init__(self):
        self._reader = GaugeReader()
        # Per-stream readers keyed by camera/stream id for isolated smoothing
        self._stream_readers: dict[str, GaugeReader] = {}

    # ── PUBLIC: Uploaded image ─────────────────────────────────────────────────

    def process_image(
        self,
        c3_output: dict,
        c1_metadata: Optional[dict] = None,
        gauge_type_override: Optional[str] = None,
        ocr_text: Optional[str] = None,
    ) -> GaugeReading:
        """
        Process a single uploaded gauge image.

        Args:
            c3_output:            Output from C3 component.
                                  Must contain at minimum: {"angle": <float>}
                                  Optional: {"confidence": <float 0-1>}

            c1_metadata:          Output from C1 component.
                                  May contain: {"class": "...", "unit": "...", ...}
                                  Used for automatic gauge type detection.

            gauge_type_override:  Force a specific gauge type key (skips detection).
                                  See gauge_config.GAUGE_CATALOGUE for valid keys.

            ocr_text:             Text read from the gauge face (if available).
                                  Used to detect gauge type and units.

        Returns:
            GaugeReading — fully populated result object.
        """
        gauge_type = detect_gauge_type(
            gauge_type_override=gauge_type_override,
            c1_metadata=c1_metadata,
            ocr_text=ocr_text,
        )

        reading = self._reader.calculate_from_c3_output(
            c3_output=c3_output,
            gauge_type=gauge_type,
            source="image",
            smooth=False,
        )

        logger.info("process_image result: %s", reading)
        return reading

    # ── PUBLIC: Live CCTV stream ───────────────────────────────────────────────

    def process_stream_frame(
        self,
        c3_output: dict,
        c1_metadata: Optional[dict] = None,
        gauge_type_override: Optional[str] = None,
        ocr_text: Optional[str] = None,
        frame_id: Optional[int] = None,
        stream_id: str = "default",
        smooth: bool = True,
    ) -> GaugeReading:
        """
        Process a single frame from a live CCTV stream.

        Args:
            c3_output:           Output from C3 (same format as process_image).
            c1_metadata:         Output from C1.
            gauge_type_override: Force gauge type (optional).
            ocr_text:            OCR text from gauge face (optional).
            frame_id:            Frame index in the video stream.
            stream_id:           Unique identifier for this camera/stream.
                                 Separate streams get separate smoothing histories.
            smooth:              Apply temporal smoothing (default True for streams).

        Returns:
            GaugeReading with source="cctv_stream".
        """
        # Ensure this stream has its own reader for isolated smoothing
        if stream_id not in self._stream_readers:
            self._stream_readers[stream_id] = GaugeReader()

        reader = self._stream_readers[stream_id]

        gauge_type = detect_gauge_type(
            gauge_type_override=gauge_type_override,
            c1_metadata=c1_metadata,
            ocr_text=ocr_text,
        )

        reading = reader.calculate_from_c3_output(
            c3_output=c3_output,
            gauge_type=gauge_type,
            source="cctv_stream",
            frame_id=frame_id,
            smooth=smooth,
        )

        logger.info("process_stream_frame [%s] frame=%s result: %s", stream_id, frame_id, reading)
        return reading

    def reset_stream(self, stream_id: str = "default"):
        """Clear smoothing history for a stream (e.g. when camera switches gauge)."""
        if stream_id in self._stream_readers:
            self._stream_readers[stream_id].clear_history()

    # ── PUBLIC: Multi-gauge batch ──────────────────────────────────────────────

    def process_multiple_gauges(
        self,
        gauge_list: list[dict],
        source: str = "image",
        frame_id: Optional[int] = None,
        stream_id: str = "default",
    ) -> list[GaugeReading]:
        """
        Process multiple gauges detected in a single image or frame.
        Useful when C1 finds more than one gauge in the scene.

        Args:
            gauge_list: List of dicts, each with:
                {
                    "c3_output":            {"angle": float, "confidence": float},
                    "c1_metadata":          {...},           # optional
                    "gauge_type_override":  "pressure_psi",  # optional
                    "ocr_text":             "...",           # optional
                }
            source:   "image" | "cctv_stream"
            frame_id: Frame number (for streams)
            stream_id: Camera ID (for streams)

        Returns:
            List of GaugeReading objects in the same order as gauge_list.
        """
        results = []
        for i, item in enumerate(gauge_list):
            try:
                if source == "cctv_stream":
                    r = self.process_stream_frame(
                        c3_output=item["c3_output"],
                        c1_metadata=item.get("c1_metadata"),
                        gauge_type_override=item.get("gauge_type_override"),
                        ocr_text=item.get("ocr_text"),
                        frame_id=frame_id,
                        stream_id=f"{stream_id}_gauge{i}",
                    )
                else:
                    r = self.process_image(
                        c3_output=item["c3_output"],
                        c1_metadata=item.get("c1_metadata"),
                        gauge_type_override=item.get("gauge_type_override"),
                        ocr_text=item.get("ocr_text"),
                    )
                results.append(r)
            except Exception as exc:
                logger.error("Error processing gauge %d: %s", i, exc)
                # Return a safe fallback reading rather than crashing the batch
                results.append(GaugeReading(
                    value=0.0, unit="?", gauge_type="unknown",
                    raw_angle_deg=0.0, percentage=0.0, confidence=0.0,
                    status="error", status_detail=str(exc),
                    source=source, frame_id=frame_id,
                ))
        return results

    # ── PUBLIC: Register custom gauge at runtime ───────────────────────────────

    def register_gauge(
        self,
        gauge_type: str,
        start_angle: float,
        end_angle: float,
        min_value: float,
        max_value: float,
        unit: str,
        description: str = "",
        warning_high: Optional[float] = None,
        warning_low: Optional[float] = None,
        danger_high: Optional[float] = None,
        danger_low: Optional[float] = None,
        decimal_places: int = 1,
    ):
        """
        Register a custom gauge type at runtime.
        After calling this, use the gauge_type key in any process_* call.

        Example:
            orch.register_gauge(
                gauge_type="boiler_pressure",
                start_angle=-135, end_angle=135,
                min_value=0, max_value=16,
                unit="bar",
                danger_high=14,
            )
        """
        config = GaugeConfig(
            gauge_type=gauge_type,
            start_angle=start_angle, end_angle=end_angle,
            min_value=min_value, max_value=max_value,
            unit=unit, description=description,
            warning_high=warning_high, warning_low=warning_low,
            danger_high=danger_high, danger_low=danger_low,
            decimal_places=decimal_places,
        )
        register_custom_gauge(config)
        logger.info("Registered custom gauge: %s", gauge_type)


# ── Convenience module-level singleton ────────────────────────────────────────

_orchestrator = GaugeOrchestrator()


def process_image(c3_output: dict, c1_metadata=None, gauge_type_override=None, ocr_text=None) -> GaugeReading:
    return _orchestrator.process_image(c3_output, c1_metadata, gauge_type_override, ocr_text)


def process_stream_frame(c3_output: dict, c1_metadata=None, gauge_type_override=None,
                          ocr_text=None, frame_id=None, stream_id="default") -> GaugeReading:
    return _orchestrator.process_stream_frame(c3_output, c1_metadata, gauge_type_override,
                                              ocr_text, frame_id, stream_id)


def process_multiple_gauges(gauge_list: list, source="image", frame_id=None, stream_id="default"):
    return _orchestrator.process_multiple_gauges(gauge_list, source, frame_id, stream_id)


# ── Demo / quick test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    orch = GaugeOrchestrator()

    print("\n═══ DEMO 1: Uploaded image — pressure gauge ═══")
    r = orch.process_image(
        c3_output={"angle": 45.0, "confidence": 0.95},
        c1_metadata={"class": "pressure_gauge"},
    )
    print(f"  Reading : {r.value} {r.unit}")
    print(f"  Status  : {r.status} — {r.status_detail}")
    print(f"  % scale : {r.percentage}%")

    print("\n═══ DEMO 2: Uploaded image — temperature gauge via OCR ═══")
    r = orch.process_image(
        c3_output={"angle": -45.0, "confidence": 0.88},
        ocr_text="0  30  60  90  120 °C",
    )
    print(f"  Reading : {r.value} {r.unit}")

    print("\n═══ DEMO 3: CCTV stream — 10 frames, smoothed ═══")
    angles = [-100, -95, -90, -85, -80, -75, -70, -65, -60, -55]
    for i, a in enumerate(angles):
        r = orch.process_stream_frame(
            c3_output={"angle": a, "confidence": 0.9},
            gauge_type_override="pressure_psi",
            frame_id=i,
            stream_id="cam1",
        )
        print(f"  frame {i:02d}  angle={a:+.1f}°  smoothed={r.value} {r.unit}")

    print("\n═══ DEMO 4: Multiple gauges in one image ═══")
    results = orch.process_multiple_gauges([
        {"c3_output": {"angle": 45}, "gauge_type_override": "pressure_psi"},
        {"c3_output": {"angle": -30}, "gauge_type_override": "temperature_celsius"},
        {"c3_output": {"angle": 90}, "gauge_type_override": "fuel_gauge"},
    ])
    for r in results:
        print(f"  {r.gauge_type:<25}  {r.value:7.2f} {r.unit}  [{r.status}]")

    print("\n═══ DEMO 5: Custom gauge registered at runtime ═══")
    orch.register_gauge(
        gauge_type="custom_boiler",
        start_angle=-120, end_angle=120,
        min_value=0, max_value=16,
        unit="bar",
        danger_high=14,
    )
    r = orch.process_image(
        c3_output={"angle": 60},
        gauge_type_override="custom_boiler",
    )
    print(f"  Reading : {r.value} {r.unit}  [{r.status}]")
    print()