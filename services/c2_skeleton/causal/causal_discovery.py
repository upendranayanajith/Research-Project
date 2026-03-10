"""
CausalSkeletonDiscovery
========================
Discovers directed causal relationships between clock-hand keypoints
from a sequence of frames using Granger causality.

Pipeline:
  1. Input: list of per-frame keypoints [{center, tip1, tip2}, ...]
  2. Build trajectories: extract per-keypoint (x,y) time series
  3. For each ordered pair (A, B): run GrangerCausalityTest(A → B)
  4. Build directed causal graph: edge A→B if test significant
  5. Annotate edges with causal strength (Granger score)

Clock-specific interpretation:
  For a clock: neither hand causes the other (both caused by mechanism).
  Finding "tip1 → tip2" would be spurious correlation.
  Finding "no causality" in either direction is the CORRECT result.
  Finding one-directional causality could indicate gear linkage / mechanism.

This is primarily useful for:
  - Multi-linkage mechanical systems (NOT just clocks)
  - Fault detection (if tip1 stops causing tip2 → gear broken)
"""

import numpy as np
from typing import List, Dict, Optional
from .granger import GrangerCausalityTest, GrangerResult


class CausalSkeletonDiscovery:
    """
    Discovers causal structure in keypoint trajectories via Granger tests.

    Parameters
    ----------
    lag_order : int
        VAR lag order for Granger test.
    significance : float
        p-value threshold for declaring causality.
    min_frames : int
        Minimum number of frames needed for reliable causal inference.
    """

    KEYPOINT_NAMES = ["center", "tip1", "tip2"]

    def __init__(
        self,
        lag_order: int = 3,
        significance: float = 0.05,
        min_frames: int = 20,
    ):
        self.lag_order = lag_order
        self.significance = significance
        self.min_frames = min_frames
        self.granger = GrangerCausalityTest(max_lag=lag_order, significance=significance)

    def discover(self, frame_keypoints: List[Dict]) -> Dict:
        """
        Run full causal discovery on a sequence of frame keypoints.

        Parameters
        ----------
        frame_keypoints : list of dicts, each with keys:
            "center": [x, y] or None
            "tip1":   [x, y] or None
            "tip2":   [x, y] or None

        Returns
        -------
        dict with:
          causal_graph  — adjacency matrix + edge list
          pairwise_results — full Granger test results
          summary — human-readable interpretation
          n_frames_used — effective frames (non-None entries)
        """
        T = len(frame_keypoints)
        if T < self.min_frames:
            return self._insufficient_data_result(T)

        # Extract trajectories: {keypoint_name: np.ndarray (T, 2)}
        trajectories = self._extract_trajectories(frame_keypoints)

        # Run all pairwise Granger tests
        names = list(trajectories.keys())
        pairwise = {}
        causal_edges = []

        for cause_name in names:
            for effect_name in names:
                if cause_name == effect_name:
                    continue
                key = f"{cause_name}→{effect_name}"

                result = self.granger.test_multidimensional(
                    x_traj=trajectories[cause_name],
                    y_traj=trajectories[effect_name],
                    cause_id=cause_name,
                    effect_id=effect_name,
                )
                pairwise[key] = result

                if result["combined_significant"]:
                    causal_edges.append({
                        "from": cause_name,
                        "to": effect_name,
                        "strength": result["combined_granger_score"],
                    })

        # Build adjacency matrix
        n = len(names)
        adj = np.zeros((n, n), dtype=np.float32)
        name_to_idx = {name: i for i, name in enumerate(names)}
        for edge in causal_edges:
            i = name_to_idx[edge["from"]]
            j = name_to_idx[edge["to"]]
            adj[i, j] = edge["strength"]

        summary = self._build_summary(causal_edges, names)

        return {
            "n_frames_used": T,
            "keypoints_analyzed": names,
            "causal_edges": causal_edges,
            "adjacency_matrix": adj.tolist(),
            "adjacency_labels": names,
            "pairwise_results": pairwise,
            "summary": summary,
        }

    def get_causal_order(self, causal_edges: List[Dict]) -> List[str]:
        """
        Topological sort of the causal graph (if acyclic).
        Returns keypoints ordered from cause to effect.
        """
        from collections import defaultdict, deque
        in_degree = defaultdict(int)
        graph = defaultdict(list)

        for edge in causal_edges:
            graph[edge["from"]].append(edge["to"])
            in_degree[edge["to"]] += 1
            if edge["from"] not in in_degree:
                in_degree[edge["from"]] = 0

        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for neighbor in graph[n]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_trajectories(self, frames: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Extract per-keypoint (x, y) time series from frame sequence.
        Fills missing frames by linear interpolation.
        """
        raw: Dict[str, List] = {name: [] for name in self.KEYPOINT_NAMES}

        for frame in frames:
            for name in self.KEYPOINT_NAMES:
                pos = frame.get(name)
                raw[name].append(pos if pos is not None else None)

        trajectories = {}
        for name, positions in raw.items():
            filled = self._fill_missing(positions)
            if filled is not None:
                trajectories[name] = filled

        return trajectories

    @staticmethod
    def _fill_missing(positions: List) -> Optional[np.ndarray]:
        """
        Fill None entries in trajectory via linear interpolation.
        Returns None if too few valid points exist (< 50% valid).
        """
        T = len(positions)
        valid = [(i, p) for i, p in enumerate(positions) if p is not None]
        if len(valid) < T * 0.5:
            return None

        arr = np.zeros((T, 2), dtype=np.float32)
        for i, p in valid:
            arr[i] = p

        # Forward fill, backward fill, then interpolate
        for i in range(T):
            if positions[i] is None:
                # Find nearest valid before and after
                before = [(j, p) for j, p in valid if j < i]
                after = [(j, p) for j, p in valid if j > i]
                if before and after:
                    j1, p1 = before[-1]
                    j2, p2 = after[0]
                    alpha = (i - j1) / (j2 - j1)
                    arr[i] = np.array(p1) * (1 - alpha) + np.array(p2) * alpha
                elif before:
                    arr[i] = np.array(before[-1][1])
                elif after:
                    arr[i] = np.array(after[0][1])

        return arr

    @staticmethod
    def _build_summary(edges: List[Dict], names: List[str]) -> str:
        if not edges:
            return (
                "No significant Granger causality detected between any keypoints. "
                "This is the expected result for independent pointers (driven by separate mechanisms)."
            )
        parts = [f"{e['from']} → {e['to']} (strength={e['strength']:.2f})" for e in edges]
        return f"Significant causal edges detected: {', '.join(parts)}. Interpret with caution — Granger causality is correlation-based, not mechanistic."

    @staticmethod
    def _insufficient_data_result(n_frames: int) -> Dict:
        return {
            "n_frames_used": n_frames,
            "error": f"Insufficient frames for causal inference (need >= 20, got {n_frames}).",
            "causal_edges": [],
            "pairwise_results": {},
            "summary": "Cannot determine causality from this sequence length.",
        }
