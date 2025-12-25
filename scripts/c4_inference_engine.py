import numpy as np
import json

class TimeInferenceEngine:
    def __init__(self):
        self.POSSIBLE_MINUTES = np.arange(0, 720) # 00:00 to 11:59

    def get_angle_diff(self, a1, a2):
        diff = np.abs(a1 - a2)
        return np.minimum(diff, 360 - diff)

    def analyze(self, angle1, angle2):
        """
        Returns a rich 'Reasoning Object' suitable for the frontend.
        """
        # 1. Physical Simulation
        theory_h = (self.POSSIBLE_MINUTES * 0.5) % 360
        theory_m = (self.POSSIBLE_MINUTES * 6) % 360

        # 2. Error Calculation (Two Hypotheses: Standard vs Swapped)
        error_std = self.get_angle_diff(angle1, theory_h) + self.get_angle_diff(angle2, theory_m)
        error_swap = self.get_angle_diff(angle2, theory_h) + self.get_angle_diff(angle1, theory_m)

        # 3. Find Best Fits
        best_std_idx = np.argmin(error_std)
        best_swap_idx = np.argmin(error_swap)
        
        min_error_std = error_std[best_std_idx]
        min_error_swap = error_swap[best_swap_idx]

        # 4. Decision Logic
        if min_error_std < min_error_swap:
            final_idx = best_std_idx
            confidence = max(0, 100 - min_error_std) # Simple confidence score
            decision = "Standard Orientation"
            rejected = f"Swapped Orientation (Error: {min_error_swap:.1f}°)"
        else:
            final_idx = best_swap_idx
            confidence = max(0, 100 - min_error_swap)
            decision = "Swapped Hands (Correction applied)"
            rejected = f"Standard Orientation (Error: {min_error_std:.1f}°)"

        # 5. Format Output
        h = int(final_idx // 60)
        if h == 0: h = 12
        m = int(final_idx % 60)

        # 6. Generate "Reasoning Trace" (The LLM-like explanation)
        explanation = (
            f"Observed angles {angle1:.1f}° and {angle2:.1f}°.\n"
            f"1. Evaluated 720 time possibilities.\n"
            f"2. Rejected '{rejected}' due to geometric inconsistency.\n"
            f"3. Accepted '{decision}' with {confidence:.1f}% physical alignment.\n"
            f"4. Result converges to {h}:{m:02d}."
        )

        return {
            "time": f"{h}:{m:02d}",
            "hour_angle": angle1 if decision == "Standard" else angle2,
            "minute_angle": angle2 if decision == "Standard" else angle1,
            "confidence": f"{confidence:.2f}%",
            "reasoning_trace": explanation
        }