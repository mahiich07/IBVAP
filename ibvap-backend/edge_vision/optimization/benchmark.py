"""
Edge Model Performance Profiler & Benchmark Utility
Measures inference latency breakdown (preprocess, inference, NMS, ByteTrack), P95/P99 latency, and FPS.
"""

import time
import numpy as np
from typing import Dict, Any, List


class EdgeModelBenchmark:
    """
    Profiles end-to-end latency and FPS throughput for edge computer vision pipelines.
    """

    def __init__(self, warmup_runs: int = 10, benchmark_runs: int = 50):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

    def profile_pipeline(self, pipeline_callable, sample_frame: np.ndarray) -> Dict[str, Any]:
        """
        Executes warmup and benchmark iterations, collecting high-precision execution timings.
        """
        # Warmup
        for _ in range(self.warmup_runs):
            _ = pipeline_callable(sample_frame)

        latencies_ms: List[float] = []

        # Benchmark
        for _ in range(self.benchmark_runs):
            t0 = time.perf_counter()
            _ = pipeline_callable(sample_frame)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        latencies_arr = np.array(latencies_ms)
        mean_lat = float(np.mean(latencies_arr))
        median_lat = float(np.median(latencies_arr))
        p95_lat = float(np.percentile(latencies_arr, 95))
        p99_lat = float(np.percentile(latencies_arr, 99))
        min_lat = float(np.min(latencies_arr))
        max_lat = float(np.max(latencies_arr))
        fps = round(1000.0 / max(1e-5, mean_lat), 1)

        return {
            "total_runs": self.benchmark_runs,
            "mean_latency_ms": round(mean_lat, 2),
            "median_latency_ms": round(median_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "p99_latency_ms": round(p99_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "throughput_fps": fps
        }
