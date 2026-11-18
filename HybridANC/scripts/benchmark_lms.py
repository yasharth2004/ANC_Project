"""Benchmark Python LMS against the C++ CMake benchmark using identical data."""

from __future__ import annotations

import time

import numpy as np

from prototype.python.lms import LMSConfig, LMSFilter


def main() -> int:
    frames = 48_000
    runs = 200
    time_axis = np.arange(frames, dtype=np.float64) / 48_000
    reference = np.sin(2.0 * np.pi * 100.0 * time_axis)
    desired = 0.8 * reference

    start = time.perf_counter()
    for _ in range(runs):
        LMSFilter(
            LMSConfig(
                filter_length=64,
                learning_rate=0.001,
            )
        ).process(reference, desired)
    elapsed = time.perf_counter() - start
    simulated_seconds_per_second = runs / elapsed

    print(
        f"Python LMS: {elapsed:.6f} seconds for "
        f"{runs} x {frames} samples, "
        f"{simulated_seconds_per_second:.3f} "
        "simulated-seconds/second"
    )
    print(
        "Run the C++ benchmark with:"
        "\n  ./build/cpp/benchmark/hybridanc_dsp_benchmark"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
