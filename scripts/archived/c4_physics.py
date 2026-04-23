import numpy as np

class ClockPhysicsEngine:
    def __init__(self):
        # We check every minute from 00:00 to 11:59
        self.POSSIBLE_MINUTES = np.arange(0, 720)
    
    def get_angle_diff(self, a1, a2):
        """Calculates smallest difference between two angles (0-360)"""
        diff = np.abs(a1 - a2)
        return np.minimum(diff, 360 - diff)

    def solve_time(self, angle1, angle2, length1=None, length2=None, top_n=3, sigma=15.0):
        """
        Input: Two raw angles from the image (we don't know which is Hour/Minute).
               length1, length2: (Optional) Physical lengths of the hands in pixels.
               top_n: Number of candidate times to return
               sigma: Standard deviation for Gaussian confidence scaling (degrees)
        Output: Best fit (Hour, Minute, Error Score, Confidence %, Candidates List, Telemetry Log)
        """
        telemetry_log = "[C4 Physics Initialized]"
        # 1. Calculate Theoretical Angles for ALL 720 possible times
        # Hour hand moves 0.5 degrees per minute (360 deg / 720 min)
        theory_h = (self.POSSIBLE_MINUTES * 0.5) % 360
        # Minute hand moves 6 degrees per minute.
        theory_m = (self.POSSIBLE_MINUTES * 6) % 360
        
        # 2. Test Hypothesis A: Angle1 is Hour, Angle2 is Minute.
        # Error = distance(Angle1, TheoryH) + distance(Angle2, TheoryM).
        error_a = self.get_angle_diff(angle1, theory_h) + \
                  self.get_angle_diff(angle2, theory_m)
                  
        # 3. Test Hypothesis B: Angle1 is Minute, Angle2 is Hour (SWAPPED)
        error_b = self.get_angle_diff(angle2, theory_h) + \
                  self.get_angle_diff(angle1, theory_m)
                  
        # --- UPSTREAM HEURISTIC DATA FUSION: HAND MORPHOLOGY WEIGHTING ---
        # The Minute hand should physically be longer than the Hour hand.
        if length1 is not None and length2 is not None and length1 > 0 and length2 > 0:
            ratio = length1 / length2
            penalty = 30.0 # Add 30 degrees of artificial error to impossible physical hypothesis
            
            # If Hand 1 is distinctly shorter (Hour hand): Hypothesis A is correct.
            if ratio < 0.9: 
                error_b += penalty # Penalize B (which says Angle1=Minute)
                telemetry_log += f" | Hand 1 ({length1:.1f}px) < Hand 2 ({length2:.1f}px) -> Rewarded Hypothesis A (Ang1=Hour), Applied {penalty}deg penalty to B."
            # If Hand 1 is distinctly longer (Minute hand): Hypothesis B is correct.
            elif ratio > 1.1:
                error_a += penalty # Penalize A (which says Angle1=Hour)
                telemetry_log += f" | Hand 1 ({length1:.1f}px) > Hand 2 ({length2:.1f}px) -> Rewarded Hypothesis B (Ang2=Hour), Applied {penalty}deg penalty to A."
        
        # 4. Compile all candidates
        # Create structured array or list of all candidates from both hypotheses
        candidates = []
        for i in range(720):
            h = int(i // 60)
            if h == 0: h = 12
            m = int(i % 60)
            
            conf_a = np.exp(-0.5 * (error_a[i] / sigma)**2) * 100
            conf_b = np.exp(-0.5 * (error_b[i] / sigma)**2) * 100
            
            candidates.append({'error': error_a[i], 'confidence': conf_a, 'hour': h, 'minute': m, 'hypothesis': 'A'})
            candidates.append({'error': error_b[i], 'confidence': conf_b, 'hour': h, 'minute': m, 'hypothesis': 'B'})
            
        # Sort candidates by error ascending
        candidates.sort(key=lambda x: x['error'])
        
        # 5. Filter for top N distinct times (avoid returning the same minute twice)
        top_candidates = []
        seen_times = set()
        
        for cand in candidates:
            time_key = f"{cand['hour']}:{cand['minute']}"
            if time_key not in seen_times:
                top_candidates.append(cand)
                seen_times.add(time_key)
            if len(top_candidates) >= top_n:
                break
                
        best = top_candidates[0]
        telemetry_log += f" | Search Complete. Minimum Error {best['error']:.2f}deg -> Winning Hypothesis: {best['hypothesis']} ({best['confidence']:.1f}% Confidence)."
        
        return best['hour'], best['minute'], best['error'], best['confidence'], top_candidates, telemetry_log