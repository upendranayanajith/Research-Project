import numpy as np

class ClockPhysicsEngine:
    def __init__(self):
        # We check every minute from 00:00 to 11:59
        self.POSSIBLE_MINUTES = np.arange(0, 720)
    
    def get_angle_diff(self, a1, a2):
        """Calculates smallest difference between two angles (0-360)"""
        diff = np.abs(a1 - a2)
        return np.minimum(diff, 360 - diff)

    def solve_time(self, angle1, angle2):
        """
        Input: Two raw angles from the image (we don't know which is Hour/Minute).
        Output: Best fit (Hour, Minute, Confidence)
        """
        # 1. Calculate Theoretical Angles for ALL 720 possible times
        # Hour hand moves 0.5 degrees per minute (360 deg / 720 min)
        theory_h = (self.POSSIBLE_MINUTES * 0.5) % 360
        # Minute hand moves 6 degrees per minute
        theory_m = (self.POSSIBLE_MINUTES * 6) % 360
        
        # 2. Test Hypothesis A: Angle1 is Hour, Angle2 is Minute
        # Error = distance(Angle1, TheoryH) + distance(Angle2, TheoryM)
        error_a = self.get_angle_diff(angle1, theory_h) + \
                  self.get_angle_diff(angle2, theory_m)
                  
        # 3. Test Hypothesis B: Angle1 is Minute, Angle2 is Hour (SWAPPED)
        error_b = self.get_angle_diff(angle2, theory_h) + \
                  self.get_angle_diff(angle1, theory_m)
        
        # 4. Find the global minimum error
        min_error_a = np.min(error_a)
        min_error_b = np.min(error_b)
        
        if min_error_a < min_error_b:
            best_idx = np.argmin(error_a)
            score = min_error_a
        else:
            best_idx = np.argmin(error_b)
            score = min_error_b
            
        # 5. Convert index (0-719) back to HH:MM
        best_hour = int(best_idx // 60)
        best_minute = int(best_idx % 60)
        if best_hour == 0: best_hour = 12
        
        return best_hour, best_minute, score