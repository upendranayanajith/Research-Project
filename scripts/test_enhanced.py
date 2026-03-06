"""Test enhanced endpoint with increased timeout."""
import requests, cv2, numpy as np

img = np.ones((500, 500, 3), dtype=np.uint8) * 60
cv2.circle(img, (250, 250), 245, (200, 200, 200), 3)
cv2.line(img, (250, 250), (330, 170), (240, 240, 240), 8)
cv2.line(img, (250, 250), (130, 280), (200, 200, 200), 4)
_, buf = cv2.imencode('.jpg', img)

files = {'file': ('test.jpg', buf.tobytes(), 'image/jpeg')}
print("Calling /extract-skeleton-enhanced (timeout=60s)...")
r = requests.post('http://localhost:8002/extract-skeleton-enhanced', files=files, timeout=60)
d = r.json()

if 'error' in d:
    print("ERROR:", d["error"])
else:
    print("keypoints:", list(d["keypoints"].keys()))
    print("angles:", d["angles"])
    e = d.get('enhanced', {})
    print("enhanced keys:", list(e.keys()))
    r3d = e.get("reconstruction_3d", {})
    print("  3D confidence:", r3d.get("confidence"))
    print("  occlusion_risk:", r3d.get("occlusion_risk"))
    sa = e.get("scale_analysis", {})
    print("  best_sigma:", sa.get("best_sigma"))
    mf = e.get("manifold", {})
    print("  surface:", mf.get("surface_classification"))
    tmp = e.get("temporal", {})
    print("  beta0:", tmp.get("beta0"))
    rv = d.get('research_visuals', {})
    print("research_visuals:", list(rv.keys()))
    for k, v in rv.items():
        print(f"  {k}: {len(v)} chars")
    print()
    print("ENHANCED ENDPOINT WORKING")
