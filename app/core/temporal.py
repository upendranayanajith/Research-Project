"""
app/core/temporal.py
====================
Tier 1.4 — Temporal Consistency for HARP live video mode.

Implements Kalman filtering on clock hand angles across video frames:
  - Smooths jitter from per-frame YOLO keypoint noise
  - Detects and flags sudden spikes (impossible jumps between frames)
  - Provides a Temporal XAI report (stability score, trend, spike count)

Usage in live mode (WebRTC / RTSP):
    tracker = TemporalTracker()
    smoothed_h, smoothed_m, spikes = tracker.update(raw_h, raw_m, unc_h, unc_m)
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Helper: circular angle difference  (handles 0°/360° wrap)
# ---------------------------------------------------------------------------
def _angle_diff(a: float, b: float) -> float:
    """Returns the signed shortest difference a - b ∈ (-180, 180]."""
    d = (a - b) % 360
    return d - 360 if d > 180 else d


def _circular_mean(angles: list) -> float:
    """Compute the circular mean of a list of angles (degrees)."""
    if not angles:
        return 0.0
    rads = [np.deg2rad(a) for a in angles]
    sin_mean = np.mean([np.sin(r) for r in rads])
    cos_mean = np.mean([np.cos(r) for r in rads])
    return float(np.degrees(np.arctan2(sin_mean, cos_mean)) % 360)


# ---------------------------------------------------------------------------
# AngleKalmanFilter — 1D Kalman for a single circular angle
# ---------------------------------------------------------------------------
class AngleKalmanFilter:
    """
    Lightweight 1-D Kalman filter designed for circular angle tracking.

    State:  angle (degrees, 0-360)
    Model:  constant-velocity assumption (clock hands move slowly)

    Tuning parameters
    -----------------
    process_noise (Q):   How much the angle can change between frames.
                         Higher → filter tracks faster but is noisier.
    measurement_noise (R): Expected sensor noise in degrees.
                         Higher → filter trusts measurements less.
    spike_threshold:     Jump (degrees) treated as a spike (frame drop /
                         C2 keypoint swap).
    """

    def __init__(
        self,
        process_noise: float = 1.5,
        measurement_noise: float = 5.0,
        spike_threshold: float = 45.0,
    ):
        self.Q = process_noise        # Process noise covariance
        self.R = measurement_noise    # Measurement noise covariance
        self.spike_threshold = spike_threshold

        # State
        self.angle: Optional[float] = None   # Current estimate (degrees)
        self.P: float = 10.0                  # Error covariance (uncertainty)
        self.spike_count: int = 0

    def reset(self):
        self.angle = None
        self.P = 10.0
        self.spike_count = 0

    def update(self, measurement: float, measurement_noise_override: float = None) -> Tuple[float, bool]:
        """
        Feed a new angle measurement and return the smoothed estimate.

        Args:
            measurement:               Raw angle in degrees [0, 360).
            measurement_noise_override: Optional per-measurement R (e.g. from MC Dropout std).

        Returns:
            (smoothed_angle, is_spike)
        """
        R = measurement_noise_override if measurement_noise_override is not None else self.R

        # First frame — initialise directly
        if self.angle is None:
            self.angle = measurement
            self.P = R
            return self.angle, False

        # --- Predict step ---
        # Constant-velocity: angle stays the same, covariance grows slightly
        P_pred = self.P + self.Q

        # --- Spike detection (before update) ---
        diff = abs(_angle_diff(measurement, self.angle))
        is_spike = diff > self.spike_threshold
        if is_spike:
            self.spike_count += 1
            # Spike: reject measurement, inflate covariance slightly and return
            self.P = min(P_pred * 1.5, 360.0)
            return self.angle, True

        # --- Update step (Kalman gain) ---
        K = P_pred / (P_pred + R)                           # Kalman gain
        diff_signed = _angle_diff(measurement, self.angle)  # Circular delta
        self.angle = (self.angle + K * diff_signed) % 360   # Update estimate
        self.P = (1 - K) * P_pred                           # Update covariance

        return self.angle, False


# ---------------------------------------------------------------------------
# TemporalTracker — manages two filters (hour + minute hands)
# ---------------------------------------------------------------------------
class TemporalTracker:
    """
    Tracks both clock hands across video frames using independent Kalman filters.
    Also maintains a rolling history for the Temporal XAI report.

    For gauge mode: only hand1 filter is used (needle angle).
    """

    HISTORY_LEN = 30   # ~1 second at 30 FPS

    def __init__(
        self,
        process_noise: float = 1.5,
        measurement_noise: float = 5.0,
        spike_threshold: float = 45.0,
    ):
        self.kf1 = AngleKalmanFilter(process_noise, measurement_noise, spike_threshold)
        self.kf2 = AngleKalmanFilter(process_noise, measurement_noise, spike_threshold)

        self._history1: deque = deque(maxlen=self.HISTORY_LEN)  # hand1 raw angles
        self._history2: deque = deque(maxlen=self.HISTORY_LEN)  # hand2 raw angles
        self._smoothed1: deque = deque(maxlen=self.HISTORY_LEN)
        self._smoothed2: deque = deque(maxlen=self.HISTORY_LEN)
        self._spike_log: deque = deque(maxlen=self.HISTORY_LEN)  # list of (frame, hand)
        self._frame_idx: int = 0

    def reset(self):
        """Call when a new clock is detected (scene change / detection gap)."""
        self.kf1.reset()
        self.kf2.reset()
        self._history1.clear()
        self._history2.clear()
        self._smoothed1.clear()
        self._smoothed2.clear()
        self._spike_log.clear()
        self._frame_idx = 0

    def update(
        self,
        a1: float,
        a2: float,
        uncertainty1: float = None,
        uncertainty2: float = None,
    ) -> Tuple[float, float, dict]:
        """
        Update filters and return smoothed angles.

        Args:
            a1:            Raw hand1 (hour) angle, degrees.
            a2:            Raw hand2 (minute) angle, degrees.
            uncertainty1:  MC Dropout std for hand1 (feeds into measurement noise R).
            uncertainty2:  MC Dropout std for hand2.

        Returns:
            (smoothed_a1, smoothed_a2, spike_info dict)
        """
        self._frame_idx += 1

        # Use MC Dropout uncertainty as per-frame measurement noise (if available)
        r1 = max(uncertainty1, 1.0) if uncertainty1 else None
        r2 = max(uncertainty2, 1.0) if uncertainty2 else None

        s1, spike1 = self.kf1.update(a1, r1)
        s2, spike2 = self.kf2.update(a2, r2)

        self._history1.append(a1)
        self._history2.append(a2)
        self._smoothed1.append(s1)
        self._smoothed2.append(s2)

        spikes_this_frame = []
        if spike1:
            spikes_this_frame.append("Hand1")
            self._spike_log.append((self._frame_idx, "Hand1"))
        if spike2:
            spikes_this_frame.append("Hand2")
            self._spike_log.append((self._frame_idx, "Hand2"))

        spike_info = {
            "spikes_this_frame": spikes_this_frame,
            "total_spikes_h1": self.kf1.spike_count,
            "total_spikes_h2": self.kf2.spike_count,
            "smoothed_a1": round(s1, 2),
            "smoothed_a2": round(s2, 2),
            "frame_idx": self._frame_idx,
        }
        return s1, s2, spike_info

    def get_temporal_xai(self) -> dict:
        """
        Returns a Temporal XAI stability report for the current rolling window.

        Includes:
          - stability_score: 0-100 (100 = perfectly smooth)
          - hand1/2_variance: circular variance of raw angles (degrees²)
          - correction_magnitude: mean |raw - smoothed| (Kalman correction applied)
          - spike_rate: spikes per frame in the rolling window
          - trend: "Stable", "Drifting", or "Unstable"
        """
        if len(self._history1) < 3:
            return {
                "status": "Initialising",
                "frames_seen": self._frame_idx,
                "message": f"Warming up Kalman filter ({self._frame_idx}/{self.HISTORY_LEN} frames).",
            }

        h1 = list(self._history1)
        h2 = list(self._history2)
        s1 = list(self._smoothed1)
        s2 = list(self._smoothed2)

        # Circular variance (using angular spread)
        def circ_var(angles):
            rads = np.deg2rad(angles)
            r = np.sqrt(np.mean(np.cos(rads))**2 + np.mean(np.sin(rads))**2)
            return float(round((1 - r) * 360, 2))  # in degrees²

        var1 = circ_var(h1)
        var2 = circ_var(h2)

        # Mean correction: how much Kalman is adjusting per frame
        corr1 = float(np.mean([abs(_angle_diff(r, s)) for r, s in zip(h1, s1)]))
        corr2 = float(np.mean([abs(_angle_diff(r, s)) for r, s in zip(h2, s2)]))
        mean_correction = round((corr1 + corr2) / 2, 2)

        total_spikes = self.kf1.spike_count + self.kf2.spike_count
        spike_rate = total_spikes / max(self._frame_idx, 1)

        # Stability score: penalise variance + correction + spikes
        stability = max(0.0, 100.0 - var1 - var2 - mean_correction * 2 - spike_rate * 200)
        stability = round(min(stability, 100.0), 1)

        # Trend classification
        if stability >= 75:
            trend = "Stable 🟢"
        elif stability >= 40:
            trend = "Drifting 🟡"
        else:
            trend = "Unstable 🔴"

        return {
            "status": "Active",
            "frames_seen": self._frame_idx,
            "stability_score": stability,
            "trend": trend,
            "hand1_variance_deg": var1,
            "hand2_variance_deg": var2,
            "mean_kalman_correction_deg": mean_correction,
            "total_spike_count": total_spikes,
            "spike_rate_per_frame": round(spike_rate, 4),
            "message": (
                f"{trend}: Kalman applying avg {mean_correction:.1f}° correction/frame. "
                f"{total_spikes} spike(s) rejected in {self._frame_idx} frames."
            ),
        }
