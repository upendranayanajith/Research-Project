"""
C4 Physics Solver — Time Resolution from Angles
Owner: Member 4
"""
import numpy as np


class PhysicsSolver:
    def __init__(self):
        self.possible_minutes = np.arange(0, 720)
        self.theory_h = (self.possible_minutes * 0.5) % 360
        self.theory_m = (self.possible_minutes * 6.0) % 360

    def solve(self, a1: float, a2: float) -> dict:
        """Resolve two angles into hour:minute via physics constraints."""
        err_a = np.abs(a1 - self.theory_h) + np.abs(a2 - self.theory_m)
        err_a = np.minimum(err_a, 720 - err_a)
        err_b = np.abs(a2 - self.theory_h) + np.abs(a1 - self.theory_m)
        err_b = np.minimum(err_b, 720 - err_b)

        if np.min(err_a) < np.min(err_b):
            idx = np.argmin(err_a)
            h = int(idx // 60) if int(idx // 60) != 0 else 12
            m = int(idx % 60)
            error = float(np.min(err_a))
        else:
            idx = np.argmin(err_b)
            h = int(idx // 60) if int(idx // 60) != 0 else 12
            m = int(idx % 60)
            error = float(np.min(err_b))

        return {
            "hour": h,
            "minute": m,
            "time": f"{h}:{m:02d}",
            "error": round(error, 2),
            "reasoning": f"Physics: H={a1:.1f}°, M={a2:.1f}° → Time={h}:{m:02d}"
        }


# Singleton
physics_solver = PhysicsSolver()
