import sqlite3
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import math

# --- [C4] GAUGE READING LOGIC ---
def calculate_gauge_reading(keypoints: List[tuple]) -> float:
    """
    [C4] Converts 4 gauge keypoints into a percentage (0-100%).
    Expected order: [Center, Min, Max, Tip]
    """
    if len(keypoints) < 4:
        return 0.0

    center, min_pt, max_pt, tip_pt = keypoints

    def get_angle(p1, p2):
        # Returns angle in degrees (0-360) relative to X-axis
        ang = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        return (ang + 360) % 360

    # 1. Get Raw Angles
    ang_min = get_angle(center, min_pt)
    ang_max = get_angle(center, max_pt)
    ang_tip = get_angle(center, tip_pt)

    # 2. Normalize to 0 (Set Min as the 0-degree starting point)
    drift = -ang_min
    rot_max = (ang_max + drift + 360) % 360
    rot_tip = (ang_tip + drift + 360) % 360

    # 3. Physics Validation (Out of bounds check)
    if rot_tip > rot_max:
        # Check if it's closer to 0 (Underflow) or Max (Overflow)
        diff_min = min(rot_tip, 360 - rot_tip)
        diff_max = abs(rot_tip - rot_max)
        if diff_min < diff_max: 
            return 0.0    # Needle resting at/below 0
        else: 
            return 100.0  # Needle pinned past Max

    # 4. Calculate Final Percentage
    if rot_max == 0: 
        return 0.0
        
    percentage = (rot_tip / rot_max) * 100
    return round(percentage, 1)

def calculate_gauge_reading_advanced(span: float, needle: float, min_val: float, max_val: float) -> tuple:
    """
    [C4] Calculates actual physical reading from gauge span/needle and scale extremes.
    Returns (reading, units_per_degree).
    """
    if span <= 0 or min_val is None or max_val is None:
        return 0.0, 0.0
        
    try:
        units_per_degree = (max_val - min_val) / span
        reading = min_val + (needle * units_per_degree)
        return round(reading, 2), round(units_per_degree, 4)
    except:
        return 0.0, 0.0


# --- [C4] METRICS TRACKER CLASS (Member 4) ---
class MetricsTracker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False 
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        
        self.db_path = "analytics.db"
        self._init_db()
        self._initialized = True

    def _init_db(self):
        """Initialize the SQLite database and table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        image_name TEXT,
                        processing_time REAL,
                        method TEXT,
                        success INTEGER,
                        confidence TEXT,
                        components_used TEXT,
                        error_msg TEXT
                    )
                """)
        except Exception as e:
            print(f"⚠️ Warning: Database initialization failed: {e}")

    def record_analysis(self, result: Dict, processing_time: float, image_name: str = ""):
        """Insert a new record into the database"""
        try:
            timestamp = datetime.now().isoformat()
            method = result.get("method", "Unknown")
            success = 1 if "error" not in result else 0
            confidence = str(result.get("confidence", "Unknown"))
            error_msg = str(result.get("error", ""))
            
            comps = []
            if "C1" in method: comps.append("C1")
            if "C2" in method: comps.append("C2")
            if "C3" in method: comps.append("C3")
            if "C4" in method: comps.append("C4")
            components_json = json.dumps(comps)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO analytics 
                    (timestamp, image_name, processing_time, method, success, confidence, components_used, error_msg)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (timestamp, image_name, processing_time, method, success, confidence, components_json, error_msg))
        except Exception as e:
            print(f"⚠️ Failed to record metric: {e}")

    def get_metrics(self) -> Dict:
        """Calculate aggregate metrics directly from SQL"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                total = cursor.execute("SELECT COUNT(*) FROM analytics").fetchone()[0]
                success = cursor.execute("SELECT COUNT(*) FROM analytics WHERE success=1").fetchone()[0]
                failed = total - success
                
                avg_time_row = cursor.execute("SELECT AVG(processing_time) FROM analytics").fetchone()
                avg_time = avg_time_row[0] if avg_time_row[0] is not None else 0.0

                methods = cursor.execute("SELECT method, COUNT(*) FROM analytics GROUP BY method").fetchall()
                method_usage = {row[0]: row[1] for row in methods}

                all_comps = cursor.execute("SELECT components_used FROM analytics").fetchall()
                comp_usage = {"C1": 0, "C2": 0, "C3": 0, "C4": 0}
                for row in all_comps:
                    try:
                        c_list = json.loads(row[0])
                        for c in c_list:
                            if c in comp_usage: comp_usage[c] += 1
                    except:
                        pass

                return {
                    "total_analyses": total,
                    "success_count": success,
                    "failure_count": failed,
                    "success_rate": (success / total * 100) if total > 0 else 0,
                    "avg_processing_time": avg_time,
                    "method_usage": method_usage,
                    "component_usage": comp_usage,
                    "uptime": 0
                }
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            return {"total_analyses": 0, "success_rate": 0}

    def get_history(self, limit=100) -> List[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM analytics ORDER BY id DESC LIMIT ?", (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except:
            return []

    def clear_metrics(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM analytics")
        except:
            pass

    def export_to_csv(self) -> str:
        import io
        import csv
        output = io.StringIO()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM analytics")
                rows = cursor.fetchall()
                if not rows: return "No Data"
                headers = [description[0] for description in cursor.description]
                writer = csv.writer(output)
                writer.writerow(headers)
                writer.writerows(rows)
        except Exception as e:
            return f"Error exporting CSV: {e}"
        return output.getvalue()

metrics_tracker = MetricsTracker()