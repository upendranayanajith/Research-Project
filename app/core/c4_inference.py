import numpy as np

class TimeInferenceEngine:
    def __init__(self):
        # Pre-calculate theoretical angles for every minute in a 12-hour cycle (0-719)
        self.POSSIBLE_MINUTES = np.arange(0, 720)
        
    def get_angle_diff(self, a1, a2):
        """Smallest difference between two angles (0-360)"""
        diff = np.abs(a1 - a2)
        return np.minimum(diff, 360 - diff)

    def analyze(self, angle1, angle2):
        """
        Input: Two angles (we don't know which is Hour/Minute).
        Output: Dictionary with time, confidence, and reasoning trace.
        """
        # 1. Physics Simulation
        theory_h = (self.POSSIBLE_MINUTES * 0.5) % 360
        theory_m = (self.POSSIBLE_MINUTES * 6) % 360

        # 2. Test Hypothesis A: Angle1=Hour, Angle2=Minute
        error_std = self.get_angle_diff(angle1, theory_h) + self.get_angle_diff(angle2, theory_m)
        
        # 3. Test Hypothesis B: Angle2=Hour, Angle1=Minute (Swapped)
        error_swap = self.get_angle_diff(angle2, theory_h) + self.get_angle_diff(angle1, theory_m)

        # 4. Find Best Fits
        best_std_idx = np.argmin(error_std)
        best_swap_idx = np.argmin(error_swap)
        
        min_error_std = error_std[best_std_idx]
        min_error_swap = error_swap[best_swap_idx]

        # 5. Decision Logic
        if min_error_std < min_error_swap:
            final_idx = best_std_idx
            confidence_score = max(0, 100 - min_error_std)
            decision = "Standard Configuration"
            accepted_h_angle = angle1
            accepted_m_angle = angle2
            rejected_scenario = f"Swapped Hands (Error: {min_error_swap:.1f}°)"
        else:
            final_idx = best_swap_idx
            confidence_score = max(0, 100 - min_error_swap)
            decision = "Swapped Hands Correction"
            accepted_h_angle = angle2
            accepted_m_angle = angle1
            rejected_scenario = f"Standard Config (Error: {min_error_std:.1f}°)"

        # 6. Convert to Time
        h = int(final_idx // 60)
        if h == 0: h = 12
        m = int(final_idx % 60)
        time_str = f"{h}:{m:02d}"

        # 7. Generate Reasoning Trace text
        trace = (
            f"--- C4 REASONING ENGINE LOG ---\n"
            f"1. OBSERVED: Line A at {angle1:.1f}°, Line B at {angle2:.1f}°\n"
            f"2. HYPOTHESIS GENERATION: Tested 720 physical time states.\n"
            f"3. EVALUATION:\n"
            f"   - Option A (A=Hour): Best fit error {min_error_std:.1f}°\n"
            f"   - Option B (B=Hour): Best fit error {min_error_swap:.1f}°\n"
            f"4. CONCLUSION: Accepted '{decision}'.\n"
            f"   - Rejected '{rejected_scenario}' due to geometric inconsistency.\n"
            f"5. RESULT: Converged to {time_str} with {confidence_score:.1f}% confidence."
        )

        return {
            "time": time_str,
            "hour_angle": float(accepted_h_angle),
            "minute_angle": float(accepted_m_angle),
            "confidence": f"{confidence_score:.2f}%",
            "trace": trace
        }