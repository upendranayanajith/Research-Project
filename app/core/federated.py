"""
app/core/federated.py
=====================
[T3.3] Federated Learning Protocol for HARP across multiple RTSP camera nodes.

Implements FedAvg (Federated Averaging) with optional Differential Privacy (ε-DP).

Architecture:
    FederatedNode        — per-camera-node client
    FederatedCoordinator — central server aggregator
    SecureAggregation    — adds Gaussian DP noise to weight deltas before sharing

Research angle:
    Privacy-preserving model improvement across surveillance cameras without
    sharing raw images. Each node trains on local frames, shares only
    compressed weight deltas with ε-DP guarantees.

References:
    McMahan et al. (2017) "Communication-Efficient Learning of Deep Networks
    from Decentralized Data" (original FedAvg paper)
    Abadi et al. (2016) "Deep Learning with Differential Privacy"
"""

from __future__ import annotations

import copy
import math
import uuid
import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Differential Privacy noise module
# ---------------------------------------------------------------------------
class SecureAggregation:
    """
    [T3.3] Adds Gaussian differential-privacy noise to weight deltas.

    Guarantees (ε, δ)-DP using the Gaussian mechanism:
        noise_scale = clip_norm * sqrt(2 * ln(1.25/δ)) / ε

    Args:
        epsilon:    Privacy budget (lower = more private). Typical: 1.0–10.0.
        delta:      Failure probability. Typical: 1e-5.
        clip_norm:  L2 gradient clipping norm. Typical: 1.0.
    """

    def __init__(self, epsilon: float = 5.0, delta: float = 1e-5, clip_norm: float = 1.0):
        self.epsilon   = epsilon
        self.delta     = delta
        self.clip_norm = clip_norm
        self.noise_scale = clip_norm * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    def clip_and_noise(self, weight_delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        1. Clip the weight delta to L2 norm ≤ clip_norm.
        2. Add Gaussian noise scaled to noise_scale.

        Args:
            weight_delta: dict of {param_name: delta_tensor}.
        Returns:
            Sanitised weight delta dict.
        """
        # Clip: compute global L2 norm across all tensors
        total_norm = sum(p.norm().item() ** 2 for p in weight_delta.values()) ** 0.5
        clip_coeff = min(1.0, self.clip_norm / (total_norm + 1e-8))

        noised = {}
        for name, delta in weight_delta.items():
            clipped = delta * clip_coeff
            noise   = torch.randn_like(clipped) * self.noise_scale
            noised[name] = clipped + noise

        return noised

    def info(self) -> dict:
        return {
            "epsilon":     self.epsilon,
            "delta":       self.delta,
            "clip_norm":   self.clip_norm,
            "noise_scale": round(self.noise_scale, 6),
        }


# ---------------------------------------------------------------------------
# Federated Node (per-camera client)
# ---------------------------------------------------------------------------
class FederatedNode:
    """
    [T3.3] Represents a single RTSP camera node in the federated network.

    Each node:
      1. Maintains a local copy of the C3 model weights
      2. Performs local fine-tuning steps on observed frames
      3. Computes a weight delta (new - old) to push to the coordinator
      4. Applies incoming global weights from coordinator

    Privacy: weight deltas are sanitised by SecureAggregation before
    being pushed to the coordinator.
    """

    def __init__(
        self,
        node_id: str,
        model: nn.Module,
        optimizer_lr: float = 1e-4,
        local_steps: int = 10,
        dp_epsilon: float = 5.0,
    ):
        self.node_id     = node_id
        self.model       = copy.deepcopy(model)
        self.optimizer   = torch.optim.Adam(self.model.parameters(), lr=optimizer_lr)
        self.local_steps = local_steps
        self.dp          = SecureAggregation(epsilon=dp_epsilon)
        self._round      = 0
        self._last_push  = None

        # Snapshot of weights before local training (for delta computation)
        self._global_weights = self._get_weights()

    def _get_weights(self) -> Dict[str, torch.Tensor]:
        """Return a deep copy of all parameter tensors."""
        return {k: v.clone().detach() for k, v in self.model.state_dict().items()}

    def apply_global_weights(self, global_weights: Dict[str, torch.Tensor]):
        """
        Receive and apply global model weights from the coordinator.
        Resets the delta baseline.
        """
        self.model.load_state_dict(global_weights)
        self._global_weights = self._get_weights()
        self._round += 1

    def local_train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module,
    ) -> float:
        """
        Run one local training batch.

        Args:
            inputs:  (N, 3, 64, 64) normalised crop tensors.
            targets: (N,) angle labels in degrees.
            loss_fn: VonMisesLoss or nn.MSELoss.

        Returns:
            Loss value for this step.
        """
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)

        # Handle both scalar and circular (sin,cos) outputs
        if outputs.shape[-1] == 2:
            # Circular head
            loss = loss_fn(outputs, targets)
        else:
            # Scalar sigmoid head
            targets_norm = targets / 360.0
            loss = loss_fn(outputs.squeeze(1), targets_norm)

        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def compute_weight_delta(self, use_dp: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute the difference between current local weights and the last
        received global weights. Optionally apply DP noise.

        Returns:
            Sanitised weight delta dict ready to push to coordinator.
        """
        current = self._get_weights()
        delta   = {k: current[k] - self._global_weights[k] for k in current}

        if use_dp:
            delta = self.dp.clip_and_noise(delta)

        self._last_push = time.time()
        return delta

    def status(self) -> dict:
        return {
            "node_id":    self.node_id,
            "round":      self._round,
            "last_push":  self._last_push,
            "dp_info":    self.dp.info(),
        }


# ---------------------------------------------------------------------------
# Federated Coordinator (server-side aggregator)
# ---------------------------------------------------------------------------
class FederatedCoordinator:
    """
    [T3.3] Central coordinator implementing FedAvg aggregation.

    Maintains a global C3 model and aggregates weight deltas from
    registered nodes using weighted averaging.

    Typical federated round:
        1. broadcast_weights(nodes)  — push current global model to all nodes
        2. [each node trains locally and computes delta]
        3. aggregate(deltas)         — FedAvg update to global model
        4. Repeat
    """

    def __init__(self, global_model: nn.Module):
        self.global_model   = copy.deepcopy(global_model)
        self.nodes: Dict[str, dict] = {}    # node_id → {n_samples, ...}
        self._round         = 0
        self._history: List[dict] = []

    def register_node(self, node_id: str, n_samples: int = 100) -> str:
        """
        Register a new camera node.

        Args:
            node_id:   Unique node identifier (e.g. "rtsp_cam_01").
            n_samples: Representative number of local samples (used for FedAvg weighting).
        Returns:
            Confirmation message.
        """
        self.nodes[node_id] = {"n_samples": n_samples, "registered_at": time.time()}
        return f"Node '{node_id}' registered. Total nodes: {len(self.nodes)}"

    def aggregate(self, weight_deltas: Dict[str, Tuple[Dict[str, torch.Tensor], int]]) -> dict:
        """
        FedAvg: apply weighted average of weight deltas to the global model.

        Args:
            weight_deltas: dict of {node_id: (delta_dict, n_samples)}

        Returns:
            Aggregation summary dict.
        """
        if not weight_deltas:
            return {"status": "skipped", "reason": "no deltas received"}

        total_samples = sum(n for _, n in weight_deltas.values())
        if total_samples == 0:
            return {"status": "error", "reason": "zero total samples"}

        # Compute weighted average delta
        avg_delta: Dict[str, torch.Tensor] = {}
        for node_id, (delta, n) in weight_deltas.items():
            weight = n / total_samples
            for name, tensor in delta.items():
                if name not in avg_delta:
                    avg_delta[name] = torch.zeros_like(tensor)
                avg_delta[name] += weight * tensor.float()

        # Apply weighted delta to global model
        global_weights = {k: v.clone() for k, v in self.global_model.state_dict().items()}
        for name, delta in avg_delta.items():
            if name in global_weights:
                global_weights[name] = global_weights[name].float() + delta

        self.global_model.load_state_dict(global_weights)
        self._round += 1

        summary = {
            "round":         self._round,
            "nodes_in_round": len(weight_deltas),
            "total_samples":  total_samples,
            "status":         "ok",
        }
        self._history.append(summary)
        return summary

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Return current global model weights (for broadcasting to nodes)."""
        return {k: v.clone().detach() for k, v in self.global_model.state_dict().items()}

    def broadcast_weights(self, nodes: List[FederatedNode]):
        """Push current global weights to all registered nodes."""
        weights = self.get_global_weights()
        for node in nodes:
            node.apply_global_weights(weights)

    def status(self) -> dict:
        return {
            "round":            self._round,
            "registered_nodes": list(self.nodes.keys()),
            "history_rounds":   len(self._history),
        }
