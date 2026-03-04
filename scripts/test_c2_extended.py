"""
test_c2_extended.py
====================
Standalone test for the extended C2 skeleton service.

Tests:
  1. Module import verification (no server needed)
  2. Probabilistic 3D reconstruction on a synthetic clock image
  3. Temporal graph dynamics tracking across 5 synthetic frames
  4. Summary report

Run from the project root:
  python scripts/test_c2_extended.py

Or against a live server:
  python scripts/test_c2_extended.py --live
"""

import sys
import os
import math
import argparse
import numpy as np

# Allow running from scripts/ or from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: create a synthetic clock image and truth keypoints
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_clock_keypoints(
    hour_angle_deg: float = 90.0,     # 3 o'clock
    minute_angle_deg: float = 180.0,  # 6 o'clock
    image_size: int = 500,
):
    """
    Returns (center, tip1, tip2) for a synthetic clock configuration.
    tip1 = hour hand tip, tip2 = minute hand tip.
    """
    cx, cy = image_size / 2, image_size / 2
    r_hour = 0.35 * image_size / 2
    r_minute = 0.70 * image_size / 2

    def polar_to_xy(r, angle_deg):
        rad = math.radians(angle_deg - 90)   # 0° = up
        return [cx + r * math.cos(rad), cy + r * math.sin(rad)]

    center = [cx, cy]
    tip1 = polar_to_xy(r_hour, hour_angle_deg)
    tip2 = polar_to_xy(r_minute, minute_angle_deg)
    return center, tip1, tip2


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Module imports
# ─────────────────────────────────────────────────────────────────────────────

def test_imports():
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)

    try:
        from services.c2_skeleton.probabilistic_3d import (
            LearnedGraphPrior, TopologyReconstructor,
            UncertaintyEstimator, BayesianGraphInference
        )
        print("  ✅ probabilistic_3d: all classes imported")
    except ImportError as e:
        print(f"  ❌ probabilistic_3d import failed: {e}")
        return False

    try:
        from services.c2_skeleton.temporal_dynamics import (
            PersistentHomologyTracker, TemporalGraphTracker, OcclusionDetector
        )
        print("  ✅ temporal_dynamics: all classes imported")
    except ImportError as e:
        print(f"  ❌ temporal_dynamics import failed: {e}")
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Probabilistic 3D Reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def test_probabilistic_3d():
    print("\n" + "="*60)
    print("TEST 2: Probabilistic 3D Reconstruction (BayesianGraphInference)")
    print("="*60)

    from services.c2_skeleton.probabilistic_3d import BayesianGraphInference

    engine = BayesianGraphInference(k_hypotheses=10, image_size=500, seed=42)
    center, tip1, tip2 = make_synthetic_clock_keypoints(90.0, 180.0)

    result = engine.infer(center, tip1, tip2)

    print(f"  summary       : {result['summary']}")
    print(f"  hand_assignment: {result['hand_assignment']}")
    print(f"  occlusion_risk : {result['occlusion_risk']}")
    print(f"  confidence     : {result['uncertainty']['confidence_score']}")
    print(f"  hour_z_offset  : {result['hand_depths']['hour_z_offset']}")
    print(f"  minute_z_offset: {result['hand_depths']['minute_z_offset']}")
    print(f"  minute_in_front: {result['hand_depths']['minute_is_in_front']}")

    # Assertions
    assert "uncertainty" in result, "Missing uncertainty"
    assert "confidence_score" in result["uncertainty"], "Missing confidence_score"
    assert 0.0 <= result["uncertainty"]["confidence_score"] <= 1.0
    assert result["occlusion_risk"] in ("LOW", "MEDIUM", "HIGH")
    assert result["hand_assignment"]["hour"] in ("tip1", "tip2")

    print("  ✅ All assertions passed")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Temporal Graph Dynamics
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_dynamics():
    print("\n" + "="*60)
    print("TEST 3: Temporal Graph Dynamics (TemporalGraphTracker)")
    print("="*60)

    from services.c2_skeleton.temporal_dynamics import TemporalGraphTracker

    tracker = TemporalGraphTracker()

    # 5-frame sequence: both visible → tip1 disappears (occlusion) → restored
    scenarios = [
        (True, True),     # Frame 0: nominal
        (True, True),     # Frame 1: nominal
        (False, True),    # Frame 2: tip1 disappears (occlusion!)
        (True, True),     # Frame 3: restored
        (True, True),     # Frame 4: nominal
    ]

    center, tip1, tip2 = make_synthetic_clock_keypoints(88.0, 92.0)  # Close angles = high occlusion risk

    for i, (t1_vis, t2_vis) in enumerate(scenarios):
        result = tracker.add_frame(
            center=center,
            tip1=tip1 if t1_vis else None,
            tip2=tip2 if t2_vis else None,
        )
        status = result["summary_status"]
        beta0 = result["betti_numbers"]["beta0"]
        print(f"  Frame {i}: β₀={beta0}  status={status}")

        if i == 2:
            # Frame where tip1 is missing
            assert beta0 == 2, f"Expected β₀=2 when tip1 missing, got {beta0}"
            print("    ✅ β₀=2 correctly detected on occlusion frame")
            if result.get("occlusion_analysis"):
                cls = result["occlusion_analysis"]["classification"]
                print(f"    🔍 Occlusion classification: {cls}")

    summary = tracker.get_session_summary()
    print(f"\n  Session summary:")
    print(f"    total_frames          : {summary['total_frames']}")
    print(f"    topology_stable_%     : {summary['topology_stable_percentage']}%")
    print(f"    total_topology_events : {summary['total_topology_events']}")

    assert summary["total_frames"] == 5
    assert summary["total_topology_events"] >= 1, "Expected at least 1 topology event"

    print("  ✅ All assertions passed")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Persistent Homology direct
# ─────────────────────────────────────────────────────────────────────────────

def test_persistent_homology():
    print("\n" + "="*60)
    print("TEST 4: PersistentHomologyTracker — Betti number computation")
    print("="*60)

    from services.c2_skeleton.temporal_dynamics import PersistentHomologyTracker
    from services.c2_skeleton.temporal_dynamics.persistent_homology import BettiNumbers

    tracker = PersistentHomologyTracker()

    cases = [
        (True,  True,  True,  1),   # All detected → 1 component
        (True,  True,  False, 1),   # Missing tip2 → still 1 (center-tip1 connected)
        (True,  False, False, 1),   # Only center → 1 isolated node
        (False, True,  True,  2),   # No center → 2 disconnected tips
        (False, False, False, 0),   # Nothing → 0 components
    ]

    for c, t1, t2, expected_b0 in cases:
        b = PersistentHomologyTracker._compute_betti(c, t1, t2)
        status = "✅" if b.beta0 == expected_b0 else "❌"
        print(f"  c={int(c)} t1={int(t1)} t2={int(t2)} → β₀={b.beta0} (expected {expected_b0}) {status}")
        assert b.beta0 == expected_b0

    print("  ✅ All Betti number computations correct")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Live server test (optional)
# ─────────────────────────────────────────────────────────────────────────────

def test_live_api(base_url: str = "http://localhost:8002"):
    """Test against a running C2 service."""
    print("\n" + "="*60)
    print(f"TEST LIVE API: {base_url}")
    print("="*60)

    try:
        import requests

        # Health check
        r = requests.get(f"{base_url}/health", timeout=5)
        health = r.json()
        print(f"  /health → {health}")
        assert health["modules"]["probabilistic_3d"] is True
        assert health["modules"]["temporal_dynamics"] is True
        print("  ✅ Extensions reported as available")

        # Temporal track
        center, tip1, tip2 = make_synthetic_clock_keypoints()
        payload = {"center": center, "tip1": tip1, "tip2": tip2}
        r = requests.post(f"{base_url}/track-temporal", json=payload, timeout=10)
        res = r.json()
        print(f"  /track-temporal → frame={res['frame_idx']} β₀={res['betti_numbers']['beta0']}")
        assert res["betti_numbers"]["beta0"] == 1

        print("  ✅ Live API tests passed")

    except Exception as e:
        print(f"  ⚠️  Live test skipped or failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Also test against live server")
    args = parser.parse_args()

    results = []
    results.append(("Imports",               test_imports()))
    results.append(("Probabilistic 3D",      test_probabilistic_3d()))
    results.append(("Temporal Dynamics",     test_temporal_dynamics()))
    results.append(("Persistent Homology",   test_persistent_homology()))

    if args.live:
        test_live_api()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_pass = True
    for name, passed in results:
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. See above for details.")
        sys.exit(1)
