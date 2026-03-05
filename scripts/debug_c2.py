import sys
import math
sys.path.insert(0, 'd:/Research-Project')

# ---- Test Betti Numbers ----
print("=== Testing Betti Numbers ===")
from services.c2_skeleton.temporal_dynamics.persistent_homology import PersistentHomologyTracker

cases = [
    (True, True, True, 1),
    (True, True, False, 1),
    (True, False, False, 1),
    (False, True, True, 2),
    (False, False, False, 0),
]
all_ok = True
for c, t1, t2, expected in cases:
    b = PersistentHomologyTracker._compute_betti(c, t1, t2)
    ok = b.beta0 == expected
    print(f"  c={int(c)} t1={int(t1)} t2={int(t2)} -> b0={b.beta0} expected={expected} {'OK' if ok else 'FAIL'}")
    if not ok:
        all_ok = False

print("BETTI OK" if all_ok else "BETTI FAIL")

# ---- Test TemporalTracker ----
print("\n=== Testing TemporalTracker ===")
from services.c2_skeleton.temporal_dynamics import TemporalGraphTracker

tracker = TemporalGraphTracker()
cx, cy = 250.0, 250.0
r_hour = 87.5
r_minute = 175.0

def polar(r, deg):
    rad = math.radians(deg - 90)
    return [cx + r * math.cos(rad), cy + r * math.sin(rad)]

center = [cx, cy]
tip1 = polar(r_hour, 88)
tip2 = polar(r_minute, 92)

scenarios = [(True, True), (True, True), (False, True), (True, True), (True, True)]
for i, (t1v, t2v) in enumerate(scenarios):
    r = tracker.add_frame(center, tip1 if t1v else None, tip2 if t2v else None)
    b0 = r["betti_numbers"]["beta0"]
    status = r["summary_status"]
    print(f"  Frame {i}: beta0={b0} status={status}")
    if i == 2:
        occ = r.get("occlusion_analysis")
        print(f"  occlusion_analysis: {occ}")
        if b0 != 2:
            print(f"  FAIL: expected beta0=2 on frame {i}, got {b0}")

summary = tracker.get_session_summary()
print(f"Session: total_frames={summary['total_frames']} events={summary['total_topology_events']}")

# ---- Test Bayesian Inference ----
print("\n=== Testing BayesianGraphInference ===")
from services.c2_skeleton.probabilistic_3d import BayesianGraphInference

engine = BayesianGraphInference(k_hypotheses=10, image_size=500, seed=42)

def kp_from_angle(h_deg, m_deg):
    c = [250.0, 250.0]
    t1 = polar(87.5, h_deg)
    t2 = polar(175.0, m_deg)
    return c, t1, t2

c, t1, t2 = kp_from_angle(90, 180)
result = engine.infer(c, t1, t2)
print(f"  summary: {result['summary']}")
print(f"  conf: {result['uncertainty']['confidence_score']}")
print(f"  occlusion_risk: {result['occlusion_risk']}")
print(f"  hand_assignment: {result['hand_assignment']}")

assert 0.0 <= result["uncertainty"]["confidence_score"] <= 1.0, "Confidence out of range"
assert result["occlusion_risk"] in ("LOW", "MEDIUM", "HIGH"), "Bad occlusion risk"
print("Bayesian OK")

print("\n=== ALL TESTS COMPLETE ===")
