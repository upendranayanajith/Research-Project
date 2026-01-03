import time
from datetime import datetime
from typing import List, Dict
import threading
import json

class MetricsTracker:
    """
    Singleton class to track performance metrics across all analyses
    """
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
        if self._initialized:
            return
        
        self._initialized = True
        self.analyses = []
        self.start_time = datetime.now()
    
    def record_analysis(self, result: Dict, processing_time: float, image_name: str = ""):
        """
        Record a single analysis result
        
        Args:
            result: Analysis result dict from engine
            processing_time: Time taken in seconds
            image_name: Optional name of analyzed image
        """
        analysis_record = {
            "timestamp": datetime.now().isoformat(),
            "image_name": image_name,
            "processing_time": processing_time,
            "success": "error" not in result,
            "method": result.get("method", "Unknown"),
            "confidence": result.get("confidence", "Unknown"),
            "time_detected": result.get("time", "N/A"),
            "error": result.get("error", None),
            "components_used": self._extract_components(result.get("method", "")),
            "angles": result.get("angles", {})
        }
        
        self.analyses.append(analysis_record)
    
    def _extract_components(self, method: str) -> List[str]:
        """Extract which components were used from method string"""
        components = []
        if "C1" in method:
            components.append("C1")
        if "C2" in method:
            components.append("C2")
        if "C3" in method:
            components.append("C3")
        if "C4" in method:
            components.append("C4")
        return components
    
    def get_metrics(self) -> Dict:
        """
        Calculate and return aggregated metrics
        """
        if not self.analyses:
            return {
                "total_analyses": 0,
                "success_rate": 0,
                "avg_processing_time": 0,
"component_usage": {},
                "confidence_distribution": {},
                "recent_analyses": []
            }
        
        total = len(self.analyses)
        successful = sum(1 for a in self.analyses if a["success"])
        
        # Calculate averages
        processing_times = [a["processing_time"] for a in self.analyses]
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Component usage count
        component_usage = {"C1": 0, "C2": 0, "C3": 0, "C4": 0}
        for analysis in self.analyses:
            for comp in analysis["components_used"]:
                if comp in component_usage:
                    component_usage[comp] += 1
        
        # Confidence distribution
        confidence_dist = {}
        for analysis in self.analyses:
            conf = analysis["confidence"]
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
        
        # Method usage
        method_usage = {}
        for analysis in self.analyses:
            method = analysis["method"]
            method_usage[method] = method_usage.get(method, 0) + 1
        
        # Processing time breakdown
        time_breakdown = {
            "min": min(processing_times) if processing_times else 0,
            "max": max(processing_times) if processing_times else 0,
            "avg": avg_time,
            "all_times": processing_times[-20:]  # Last 20 analyses
        }
        
        return {
            "total_analyses": total,
            "success_count": successful,
            "failure_count": total - successful,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "avg_processing_time": avg_time,
            "time_breakdown": time_breakdown,
            "component_usage": component_usage,
            "confidence_distribution": confidence_dist,
            "method_usage": method_usage,
            "recent_analyses": self.analyses[-10:],  # Last 10
            "uptime": (datetime.now() - self.start_time).total_seconds() / 3600  # hours
        }
    
    def export_to_csv(self) -> str:
        """Export analyses to CSV format"""
        if not self.analyses:
            return "No data available"
        
        import io
        import csv
        
        output = io.StringIO()
        fieldnames = ["timestamp", "image_name", "processing_time", "success", 
                     "method", "confidence", "time_detected", "error"]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for analysis in self.analyses:
            row = {k: analysis.get(k, "") for k in fieldnames}
            writer.writerow(row)
        
        return output.getvalue()
    
    def clear_metrics(self):
        """Reset all metrics"""
        self.analyses = []
        self.start_time = datetime.now()

# Global instance
metrics_tracker = MetricsTracker()
