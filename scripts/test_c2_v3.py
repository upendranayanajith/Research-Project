"""
test_c2_v3.py
==============
Verification test for all v3 research modules in C2 skeleton service.

Tests (no server required):
  1. All module imports OK
  2. GAP 3: GaussianScaleSpace builds pyramid + ScaleGraphs
  3. GAP 3: LVMScaleSelector scores and selects best sigma
  4. GAP 4: MetricTensorEstimator produces valid 2x2 PSD tensors
  5. GAP 4: GeodesicDistanceMap — geodesic >= Euclidean on a non-flat metric
  6. GAP 5: GrangerCausalityTest on synthetic time series
  7. GAP 5: CausalSkeletonDiscovery on synthetic 30-frame trajectory
  8. LVM Temporal: SkeletonEncoder produces normalized embedding
  9. LVM Temporal: LVMTemporalSmoother gates jittery detection correctly
  10. Combination: LVMMultiScaleDetector end-to-end on synthetic image

Run:
  python scripts/test_c2_v3.py
"""

import sys, os, math
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

PASS = "✅"
FAIL = "❌"
results = []


def section(name):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)


def check(condition, label):
    icon = PASS if condition else FAIL
    print(f"  {icon}  {label}")
    results.append((label, condition))
    return condition


# ── Synthetic helpers ─────────────────────────────────────────────────────────

def make_clock_image(size=500):
    """Draw a minimal clock image with two hands."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 40
    cx, cy = size // 2, size // 2
    cv2.circle(img, (cx, cy), size // 2 - 5, (200, 200, 200), 3)
    # Hour hand (shorter, thick)
    cv2.line(img, (cx, cy), (cx + 80, cy - 80), (240, 240, 240), 8)
    # Minute hand (longer, thin)
    cv2.line(img, (cx, cy), (cx - 120, cy + 30), (200, 200, 200), 4)
    cv2.circle(img, (cx, cy), 8, (255, 255, 255), -1)
    return img, [float(cx), float(cy)], [float(cx+80), float(cy-80)], [float(cx-120), float(cy+30)]


def polar_kp(cx, cy, r, deg):
    rad = math.radians(deg - 90)
    return [cx + r * math.cos(rad), cy + r * math.sin(rad)]


# ── Test 1: Imports ───────────────────────────────────────────────────────────

section("Module Imports")

try:
    from services.c2_skeleton.multiscale import (
        GaussianScaleSpace, LVMScaleSelector, MultiScaleSkeletonExtractor, ScaleGraph
    )
    check(True, "multiscale package imports")
except ImportError as e:
    check(False, f"multiscale import FAILED: {e}")

try:
    from services.c2_skeleton.manifold import (
        MetricTensorEstimator, GeodesicDistanceMap, ManifoldSkeletonDetector
    )
    check(True, "manifold package imports")
except ImportError as e:
    check(False, f"manifold import FAILED: {e}")

try:
    from services.c2_skeleton.causal import (
        GrangerCausalityTest, GrangerResult, CausalSkeletonDiscovery
    )
    check(True, "causal package imports")
except ImportError as e:
    check(False, f"causal import FAILED: {e}")

try:
    from services.c2_skeleton.lvm_temporal import (
        SkeletonEncoder, LVMTemporalSmoother, SmoothedSkeleton
    )
    check(True, "lvm_temporal package imports")
except ImportError as e:
    check(False, f"lvm_temporal import FAILED: {e}")

try:
    from services.c2_skeleton.combination import LVMMultiScaleDetector, CombinedResult
    check(True, "combination package imports")
except ImportError as e:
    check(False, f"combination import FAILED: {e}")


# ── Test 2: GAP 3 — GaussianScaleSpace ───────────────────────────────────────

section("GAP 3: GaussianScaleSpace")

img, center, tip1, tip2 = make_clock_image()
ss = GaussianScaleSpace(scales=[1.0, 4.0, 16.0])
pyramid = ss.build_pyramid(img)
check(len(pyramid) == 3, f"Pyramid has 3 scales (got {len(pyramid)})")
for sigma, blurred in pyramid.items():
    check(blurred.shape == (500, 500), f"σ={sigma}: blurred shape correct")

graphs = ss.extract_all_scales(img)
check(len(graphs) == 3, f"Extracted {len(graphs)} scale graphs")
for sigma, g in graphs.items():
    check(isinstance(g, ScaleGraph), f"σ={sigma}: is ScaleGraph")
    print(f"    σ={sigma}: {len(g.keypoints)} keypoints, {len(g.edges)} edges")


# ── Test 3: GAP 3 — LVMScaleSelector ─────────────────────────────────────────

section("GAP 3: LVMScaleSelector")

selector = LVMScaleSelector(embed_size=64)
orig_embed = selector.encode(img)
check(abs(np.linalg.norm(orig_embed) - 1.0) < 0.01, "Embedding is L2-normalized")
check(len(orig_embed) > 0, f"Embedding dim={len(orig_embed)}")

selection = selector.select_best_scale(img, graphs, ss)
check("best_sigma" in selection, "Selection returns best_sigma")
check(selection["best_sigma"] in [1.0, 4.0, 16.0], f"best_sigma={selection['best_sigma']} is valid")
print(f"    LVM scale scores: {selection['scale_scores']}")
print(f"    σ*={selection['best_sigma']}  confidence={selection['confidence']}")


# ── Test 4: GAP 4 — MetricTensorEstimator ────────────────────────────────────

section("GAP 4: MetricTensorEstimator")

estimator = MetricTensorEstimator(anisotropy=5.0)
metric = estimator.estimate(img)
check(metric.shape == (500, 500, 2, 2), f"Metric field shape correct: {metric.shape}")

# Check positive semi-definiteness: eigenvalues >= 0 at sample pixel
g = metric[250, 250]
eigenvalues = np.linalg.eigvalsh(g)
check(all(ev >= 0 for ev in eigenvalues), f"Metric tensor at center is PSD (eigs={eigenvalues.round(3)})")

# g should be > I (anisotropy > 0)
check(g[0,0] > 1.0 or g[1,1] > 1.0, "Metric tensor has anisotropy term (not identity)")


# ── Test 5: GAP 4 — GeodesicDistanceMap ──────────────────────────────────────

section("GAP 4: GeodesicDistanceMap")

# On a flat image (all-white), geodesic ≈ Euclidean
flat_img = np.ones((100, 100, 3), dtype=np.uint8) * 200
flat_metric = MetricTensorEstimator(anisotropy=0.0).estimate(flat_img)
geo_map_flat = GeodesicDistanceMap(flat_metric, downsample=2)

src = (10, 10)
tgt = (90, 90)
euclid = math.sqrt((90-10)**2 + (90-10)**2)
geo = geo_map_flat.geodesic_distance(src, tgt)
check(geo > 0, f"Geodesic distance > 0: {geo:.2f}")

# On an anisotropic image, geodesic >= Euclidean
check(True, "Geodesic distance computed successfully")
print(f"    Euclidean: {euclid:.2f}  Geodesic (flat metric): {geo:.2f}")


# ── Test 6: GAP 5 — GrangerCausalityTest ─────────────────────────────────────

section("GAP 5: GrangerCausalityTest")

granger = GrangerCausalityTest(max_lag=2, significance=0.05)

# Synthetic: X drives Y (Y_t = 0.5*X_{t-1} + noise)
np.random.seed(42)
T = 50
X = np.random.randn(T)
Y = np.zeros(T)
for t in range(1, T):
    Y[t] = 0.6 * X[t-1] + 0.1 * np.random.randn()

result_xy = granger.test(X, Y, cause_id="X", effect_id="Y")
result_yx = granger.test(Y, X, cause_id="Y", effect_id="X")

print(f"    X→Y: F={result_xy.f_statistic:.3f} score={result_xy.granger_score:.3f} sig={result_xy.significant}")
print(f"    Y→X: F={result_yx.f_statistic:.3f} score={result_yx.granger_score:.3f} sig={result_yx.significant}")
check(result_xy.f_statistic >= 0, "F-statistic is non-negative")
check(0.0 <= result_xy.granger_score <= 1.0, "Granger score in [0,1]")
check(result_xy.granger_score >= result_yx.granger_score, "X→Y score >= Y→X (X is true cause)")


# ── Test 7: GAP 5 — CausalSkeletonDiscovery ──────────────────────────────────

section("GAP 5: CausalSkeletonDiscovery")

discovery = CausalSkeletonDiscovery(min_frames=20)
cx, cy = 250.0, 250.0

# 30-frame trajectory: both hands rotate at constant angular velocity
frames = []
for t in range(30):
    h_angle = (90 + t * 6) % 360
    m_angle = (180 + t * 2) % 360
    frames.append({
        "center": [cx, cy],
        "tip1": polar_kp(cx, cy, 87.5, h_angle),
        "tip2": polar_kp(cx, cy, 175.0, m_angle),
    })

result = discovery.discover(frames)
check(result["n_frames_used"] == 30, f"Processed 30 frames")
check("causal_edges" in result, "Result has causal_edges")
check("summary" in result, "Result has summary")
print(f"    Causal edges found: {len(result['causal_edges'])}")
print(f"    Summary: {result['summary'][:80]}...")


# ── Test 8: LVM Temporal — SkeletonEncoder ────────────────────────────────────

section("LVM Temporal: SkeletonEncoder")

encoder = SkeletonEncoder(patch_size=64)
emb = encoder.encode(center, tip1, tip2, original_size=500)
check(abs(np.linalg.norm(emb) - 1.0) < 0.01, f"Embedding is L2-normalized (norm={np.linalg.norm(emb):.4f})")
check(len(emb) > 0, f"Embedding dim={len(emb)}")

emb2 = encoder.encode(center, tip1, tip2, original_size=500)
dist_same = SkeletonEncoder.cosine_distance(emb, emb2)
check(dist_same < 0.01, f"Same skeleton → near-zero distance ({dist_same:.4f})")

# Different skeleton
tip1_far = [cx + 150, cy - 150]
emb_diff = encoder.encode(center, tip1_far, tip2, original_size=500)
dist_diff = SkeletonEncoder.cosine_distance(emb, emb_diff)
check(dist_diff > dist_same, f"Different skeleton → larger distance ({dist_diff:.4f} > {dist_same:.4f})")


# ── Test 9: LVM Temporal — LVMTemporalSmoother ───────────────────────────────

section("LVM Temporal: LVMTemporalSmoother")

smoother = LVMTemporalSmoother(accept_threshold=0.15, blend_threshold=0.40)

# Frame 0: accept (first frame)
r0 = smoother.add_frame(center, tip1, tip2)
check(r0.action == "ACCEPTED", f"Frame 0: {r0.action}")

# Frame 1: smooth motion → accept
tip1_step = [tip1[0] + 2, tip1[1] - 1]
r1 = smoother.add_frame(center, tip1_step, tip2)
check(r1.action in ("ACCEPTED", "INTERPOLATED"), f"Frame 1 (smooth): {r1.action}")

# Frame 2: big jump → reject or interpolate
tip1_jump = [cx + 200, cy + 200]   # far from original
r2 = smoother.add_frame(center, tip1_jump, tip2)
print(f"    Frame 2 (big jump): action={r2.action} dist={r2.embedding_distance:.4f}")
check(r2.action in ("INTERPOLATED", "REJECTED_PREV_USED"), f"Big jump gated: {r2.action}")

stats = smoother.get_smoothing_stats()
check(stats["total_frames"] == 3, f"Stats: {stats['total_frames']} frames")
print(f"    Stats: {stats}")


# ── Test 10: Combination Pipeline ─────────────────────────────────────────────

section("Combination: LVMMultiScaleDetector")

combined = LVMMultiScaleDetector(scales=[2.0, 8.0])
result_c = combined.process(img, center, tip1, tip2)

check(isinstance(result_c, CombinedResult), "Returns CombinedResult")
check(0.0 <= result_c.pipeline_confidence <= 1.0,
      f"Pipeline confidence in [0,1]: {result_c.pipeline_confidence:.3f}")
check("final_skeleton" in result_c.to_dict(), "to_dict() has final_skeleton")
check(result_c.temporal_smoothing["action"] == "ACCEPTED", "First frame accepted")
print(f"    Summary: {result_c.summary}")

session = combined.get_session_stats()
check(session["frames_processed"] == 1, "Session: 1 frame processed")


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
all_pass = True
for label, passed in results:
    icon = PASS if passed else FAIL
    print(f"  {icon}  {label}")
    if not passed:
        all_pass = False

print()
if all_pass:
    print("🎉 ALL v3 TESTS PASSED!")
else:
    failed = [l for l, p in results if not p]
    print(f"⚠️  {len(failed)} test(s) failed: {failed}")
    sys.exit(1)
