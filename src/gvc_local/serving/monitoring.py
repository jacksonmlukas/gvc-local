"""Lightweight, zero-dependency request monitoring for the GVC-Local API.

All metrics live in-process memory (no Prometheus/StatsD required).  The
``RequestMonitor`` is thread-safe and designed to be instantiated once at
app startup, then shared via FastAPI dependency injection.

Typical flow::

    monitor = RequestMonitor(buffer_size=2000)

    # inside a request handler:
    record = monitor.start()
    ...
    monitor.finish(record, tokens_in=512, tokens_out=128, success=True)

    # on GET /metrics:
    return monitor.summary()
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestRecord:
    """Mutable record for a single in-flight or completed request."""

    start_time: float = field(default_factory=time.monotonic)
    end_time: float = 0.0
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    solver: str = ""
    model: str = ""
    success: bool = True
    error: str = ""


class RequestMonitor:
    """Thread-safe in-memory metrics collector.

    Parameters
    ----------
    buffer_size:
        Maximum number of completed ``RequestRecord`` objects to retain.
        Older entries are evicted in FIFO order.  A larger buffer yields
        more accurate percentile estimates at the cost of memory.
    """

    def __init__(self, buffer_size: int = 2000) -> None:
        self._buffer: deque[RequestRecord] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._created_at: float = time.monotonic()

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def start(self, *, solver: str = "", model: str = "") -> RequestRecord:
        """Create a new record when a request arrives.

        The caller keeps the returned object and passes it back to
        :meth:`finish` once the request completes.
        """
        return RequestRecord(solver=solver, model=model)

    def finish(
        self,
        record: RequestRecord,
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        success: bool = True,
        error: str = "",
    ) -> None:
        """Finalize a record and push it into the ring buffer."""
        record.end_time = time.monotonic()
        record.latency_ms = (record.end_time - record.start_time) * 1_000
        record.tokens_in = tokens_in
        record.tokens_out = tokens_out
        record.success = success
        record.error = error

        with self._lock:
            self._buffer.append(record)
            self._total_requests += 1
            if not success:
                self._total_errors += 1

    # ------------------------------------------------------------------
    # Percentile helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile(sorted_values: list[float], p: float) -> float:
        """Linear-interpolation percentile on a *pre-sorted* list."""
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * p
        f = int(k)
        c = f + 1
        if c >= len(sorted_values):
            return sorted_values[-1]
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])

    # ------------------------------------------------------------------
    # Summary export (for /metrics)
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return an aggregate snapshot suitable for JSON serialisation.

        The returned dict is intentionally flat so it renders nicely as a
        JSON response and is easy to ingest into dashboards.
        """
        with self._lock:
            records = list(self._buffer)
            total = self._total_requests
            errors = self._total_errors

        uptime_s = time.monotonic() - self._created_at

        if not records:
            return {
                "total_requests": total,
                "error_count": errors,
                "error_rate": 0.0,
                "uptime_s": round(uptime_s, 2),
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
                "tokens_in_total": 0,
                "tokens_out_total": 0,
                "tokens_per_minute": 0.0,
                "requests_by_solver": {},
                "requests_by_model": {},
            }

        latencies = sorted(r.latency_ms for r in records)
        tokens_in_total = sum(r.tokens_in for r in records)
        tokens_out_total = sum(r.tokens_out for r in records)
        total_tokens = tokens_in_total + tokens_out_total

        # Tokens-per-minute based on the time span covered by the buffer.
        earliest = min(r.start_time for r in records)
        latest = max(r.end_time for r in records) if any(r.end_time for r in records) else earliest
        span_minutes = max((latest - earliest) / 60.0, 1 / 60.0)  # floor at 1 second
        tokens_per_minute = round(total_tokens / span_minutes, 1)

        # Break-downs by solver and model.
        by_solver: dict[str, int] = {}
        by_model: dict[str, int] = {}
        for r in records:
            by_solver[r.solver] = by_solver.get(r.solver, 0) + 1
            by_model[r.model] = by_model.get(r.model, 0) + 1

        return {
            "total_requests": total,
            "error_count": errors,
            "error_rate": round(errors / max(total, 1), 4),
            "uptime_s": round(uptime_s, 2),
            "latency_p50_ms": round(self._percentile(latencies, 0.50), 2),
            "latency_p95_ms": round(self._percentile(latencies, 0.95), 2),
            "latency_p99_ms": round(self._percentile(latencies, 0.99), 2),
            "tokens_in_total": tokens_in_total,
            "tokens_out_total": tokens_out_total,
            "tokens_per_minute": tokens_per_minute,
            "requests_by_solver": by_solver,
            "requests_by_model": by_model,
        }
