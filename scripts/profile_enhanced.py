"""Profile which sub-module is slow in the enhanced endpoint."""
import time, sys, os
sys.path.insert(0, '.')
import numpy as np, cv2

img = np.ones((500, 500, 3), dtype=np.uint8) * 60
cv2.circle(img, (250, 250), 245, (200, 200, 200), 3)
cv2.line(img, (250, 250), (330, 170), (240, 240, 240), 8)
cv2.line(img, (250, 250), (130, 280), (200, 200, 200), 4)

center, tip1, tip2 = [250.0, 250.0], [330.0, 170.0], [130.0, 280.0]

# 1. Bayesian 3D
t = time.time()
from services.c2_skeleton.probabilistic_3d import BayesianGraphInference
eng = BayesianGraphInference(k_hypotheses=10, image_size=500)
eng.infer(center, tip1, tip2)
print(f"Bayesian 3D:    {time.time()-t:.2f}s")

# 2. Multi-scale
t = time.time()
from services.c2_skeleton.multiscale import MultiScaleSkeletonExtractor
ms = MultiScaleSkeletonExtractor()
ms.extract(img)
print(f"Multi-scale:    {time.time()-t:.2f}s")

# 3. Manifold
t = time.time()
from services.c2_skeleton.manifold import ManifoldSkeletonDetector
mf = ManifoldSkeletonDetector()
mf.detect(img, center, tip1, tip2)
print(f"Manifold:       {time.time()-t:.2f}s")

# 4. Temporal
t = time.time()
from services.c2_skeleton.temporal_dynamics import TemporalGraphTracker
tt = TemporalGraphTracker(max_history=100)
tt.add_frame(center, tip1, tip2)
print(f"Temporal:       {time.time()-t:.2f}s")

# 5. Visualizations
t = time.time()
from services.c2_skeleton.viz import ResearchVisualizer
ResearchVisualizer.render_scale_pyramid(img, {"1.0": 0.5, "4.0": 0.9}, 4.0)
ResearchVisualizer.render_confidence_gauge(0.7, "LOW")
ResearchVisualizer.render_curvature_heatmap(img, {}, "FLAT")
ResearchVisualizer.render_comparison(img, center, tip1, tip2, 0.7, "LOW", 4.0)
ResearchVisualizer.render_betti_badge(1, 0, "NOMINAL")
ResearchVisualizer.render_impact_kpis(0.7, 4.0, "LOW", "FLAT", 1)
print(f"Visualizations: {time.time()-t:.2f}s")
