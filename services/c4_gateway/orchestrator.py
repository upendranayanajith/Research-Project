"""
C4 Orchestrator — Calls C1, C2, C3 services via HTTP and assembles results
Owner: Member 4
"""
import requests
import base64
import time


# Service URLs (configurable via environment variables)
C1_URL = "http://localhost:8001"
C2_URL = "http://localhost:8002"
C3_URL = "http://localhost:8003"


def call_c1(image_bytes: bytes, filename: str = "image.jpg") -> dict:
    """Call C1 Localization Service."""
    try:
        resp = requests.post(
            f"{C1_URL}/localize",
            files={"file": (filename, image_bytes, "image/jpeg")},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "C1 service unavailable", "found": False}
    except Exception as e:
        return {"error": f"C1 call failed: {str(e)}", "found": False}


def call_c2(cropped_b64: str) -> dict:
    """Call C2 Skeleton Service with base64 cropped image."""
    try:
        img_bytes = base64.b64decode(cropped_b64)
        resp = requests.post(
            f"{C2_URL}/extract-skeleton",
            files={"file": ("cropped.jpg", img_bytes, "image/jpeg")},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "C2 service unavailable"}
    except Exception as e:
        return {"error": f"C2 call failed: {str(e)}"}


def call_c2_enhanced(cropped_b64: str) -> dict:
    """
    Call C2 Enhanced Skeleton Service — returns research visuals + metrics.
    Falls back to basic /extract-skeleton if enhanced endpoint is unavailable.
    """
    try:
        img_bytes = base64.b64decode(cropped_b64)
        resp = requests.post(
            f"{C2_URL}/extract-skeleton-enhanced",
            files={"file": ("cropped.jpg", img_bytes, "image/jpeg")},
            timeout=60   # longer timeout — runs all research algorithms
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        # Fallback to basic endpoint
        return call_c2(cropped_b64)


def call_c3(cropped_b64: str, keypoints: dict, rough_angles: dict) -> dict:
    """Call C3 Angle Refinement Service."""
    try:
        resp = requests.post(
            f"{C3_URL}/refine-angles",
            json={
                "image": cropped_b64,
                "keypoints": keypoints,
                "rough_angles": rough_angles
            },
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "C3 service unavailable", "refined": False}
    except Exception as e:
        return {"error": f"C3 call failed: {str(e)}", "refined": False}


def check_services() -> dict:
    """Check health of all downstream services."""
    status = {}
    for name, url in [("C1", C1_URL), ("C2", C2_URL), ("C3", C3_URL)]:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            status[name] = resp.json() if resp.status_code == 200 else {"status": "error"}
        except:
            status[name] = {"status": "unreachable"}
    return status
