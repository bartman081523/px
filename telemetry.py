"""
telemetry.py — Minimal In-Memory Telemetry Store
=================================================
Bounded circular buffer. No file dumps. No session persistence.
Lost on restart (by design). Replaces the 20,000+ JSON file approach
from dmt_space_50.
"""

import time
from collections import deque
from typing import Optional

MAX_HISTORY = 100  # Keep last 100 request metrics


class TelemetryStore:
    def __init__(self, max_history: int = MAX_HISTORY):
        self._history = deque(maxlen=max_history)
        self._totals = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_prompt_tokens": 0,
        }

    def record(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        px_metrics: Optional[dict] = None,
    ):
        """Record a request's metrics."""
        self._history.append({
            "timestamp": time.time(),
            "model_id": model_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "px_metrics": px_metrics or {},
        })
        self._totals["total_requests"] += 1
        self._totals["total_tokens_generated"] += completion_tokens
        self._totals["total_prompt_tokens"] += prompt_tokens

    def get_summary(self) -> dict:
        """Get aggregate and recent metrics."""
        return {
            **self._totals,
            "recent": list(self._history),
        }

    def get_phi_traces(self, model_id: str = None) -> list:
        """Extract phi values from recent history for line chart."""
        phis = []
        for entry in self._history:
            if model_id and entry.get("model_id") != model_id:
                continue
            px = entry.get("px_metrics", {})
            cs = px.get("cognitive_signature", {})
            phi = cs.get("phi", px.get("phi"))
            if phi is not None:
                phis.append(phi)
        return phis

    def get_zone_distributions(self, model_id: str = None) -> dict:
        """Aggregate zone_weights from recent PX metrics."""
        zone_agg = {}
        for entry in self._history:
            if model_id and entry.get("model_id") != model_id:
                continue
            zw = entry.get("px_metrics", {}).get("zone_weights", {})
            for k, v in zw.items():
                zone_agg[k] = zone_agg.get(k, 0) + v
        # Normalize
        total = sum(zone_agg.values()) if zone_agg else 1
        return {k: v / total for k, v in zone_agg.items()} if total > 0 else {}

    def get_kurtosis_values(self, model_id: str = None) -> list:
        """Extract kurtosis values from cognitive signatures."""
        vals = []
        for entry in self._history:
            if model_id and entry.get("model_id") != model_id:
                continue
            k = entry.get("px_metrics", {}).get("cognitive_signature", {}).get("kurtosis")
            if k is not None:
                vals.append(k)
        return vals

    def get_emancipation_data(self, model_id: str = None) -> list:
        """Extract emancipation trajectory data."""
        vals = []
        for entry in self._history:
            if model_id and entry.get("model_id") != model_id:
                continue
            traj = entry.get("px_metrics", {}).get("emancipation_trajectory", [])
            if traj:
                vals.extend(traj)
        return vals

    def reset(self):
        """Clear all telemetry."""
        self._history.clear()
        self._totals = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_prompt_tokens": 0,
        }


# Global singleton
telemetry = TelemetryStore()