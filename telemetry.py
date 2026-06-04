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