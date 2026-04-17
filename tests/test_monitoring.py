"""Unit tests for the in-memory request monitoring module."""

import threading
import time

from gvc_local.serving.monitoring import RequestMonitor


class TestRequestMonitor:
    def test_empty_summary(self) -> None:
        mon = RequestMonitor(buffer_size=100)
        s = mon.summary()
        assert s["total_requests"] == 0
        assert s["error_rate"] == 0.0
        assert s["latency_p50_ms"] == 0.0

    def test_single_request(self) -> None:
        mon = RequestMonitor()
        rec = mon.start(solver="gvc", model="llama-8b")
        time.sleep(0.01)  # ~10 ms
        mon.finish(rec, tokens_in=100, tokens_out=50, success=True)

        s = mon.summary()
        assert s["total_requests"] == 1
        assert s["error_count"] == 0
        assert s["tokens_in_total"] == 100
        assert s["tokens_out_total"] == 50
        assert s["latency_p50_ms"] > 0
        assert s["requests_by_solver"] == {"gvc": 1}
        assert s["requests_by_model"] == {"llama-8b": 1}

    def test_error_tracking(self) -> None:
        mon = RequestMonitor()
        rec = mon.start(solver="snap_gvc", model="qwen-7b")
        mon.finish(rec, success=False, error="timeout")

        s = mon.summary()
        assert s["total_requests"] == 1
        assert s["error_count"] == 1
        assert s["error_rate"] == 1.0

    def test_buffer_eviction(self) -> None:
        """Old records are evicted when buffer_size is exceeded."""
        mon = RequestMonitor(buffer_size=5)
        for i in range(10):
            rec = mon.start(solver="gvc", model="llama-8b")
            mon.finish(rec, tokens_in=i, tokens_out=0)

        s = mon.summary()
        # total_requests counts ALL requests, not just buffered ones.
        assert s["total_requests"] == 10
        # But token totals only reflect the 5 most recent (indices 5..9).
        assert s["tokens_in_total"] == sum(range(5, 10))

    def test_percentile_calculation(self) -> None:
        assert RequestMonitor._percentile([], 0.5) == 0.0
        assert RequestMonitor._percentile([10.0], 0.5) == 10.0
        assert RequestMonitor._percentile([1.0, 2.0, 3.0], 0.5) == 2.0
        # p99 of a 3-element list should be close to the max.
        p99 = RequestMonitor._percentile([1.0, 2.0, 3.0], 0.99)
        assert p99 >= 2.9

    def test_thread_safety(self) -> None:
        """Concurrent writers should not corrupt the monitor."""
        mon = RequestMonitor(buffer_size=500)
        errors: list[Exception] = []

        def writer(n: int) -> None:
            try:
                for _ in range(n):
                    rec = mon.start(solver="gvc", model="llama-8b")
                    mon.finish(rec, tokens_in=1, tokens_out=1)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(50,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        s = mon.summary()
        assert s["total_requests"] == 500

    def test_tokens_per_minute_positive(self) -> None:
        mon = RequestMonitor()
        rec = mon.start()
        time.sleep(0.01)
        mon.finish(rec, tokens_in=500, tokens_out=200)

        s = mon.summary()
        assert s["tokens_per_minute"] > 0
