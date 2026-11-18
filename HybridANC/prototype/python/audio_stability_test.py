"""Measured real-time capture stability matrix.

No assumptions are promoted to results. Each row reports only this machine,
the selected device and the observed callback/dropout/CPU values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import logging
import os
import time

import numpy as np

try:
    from .audio_capture import AudioCapture, AudioCaptureConfig
except ImportError:
    from audio_capture import AudioCapture, AudioCaptureConfig


LOGGER = logging.getLogger("hybridanc.audio_stability")


@dataclass(frozen=True)
class StabilityReport:
    sample_rate: int
    block_size: int
    duration_seconds: float
    capture_blocks: int
    capture_overflows: int
    callback_count: int
    callback_average_ms: float | None
    callback_jitter_ms: float | None
    callback_min_ms: float | None
    callback_max_ms: float | None
    cpu_load_percent: float
    status: str


def run_stability(
    input_device: int | str | None,
    sample_rate: int,
    block_size: int,
    duration_seconds: float,
) -> StabilityReport:
    capture = AudioCapture(
        AudioCaptureConfig(
            device=input_device,
            sample_rate=sample_rate,
            block_size=block_size,
            channels=1,
            queue_depth=32,
            buffer_count=64,
        )
    )
    cpu_before = os.times()
    wall_before = time.monotonic()
    deadline = wall_before + duration_seconds
    block_count = 0
    status = "completed"

    try:
        capture.start()
        while time.monotonic() < deadline:
            try:
                block = capture.read(timeout=0.25)
            except TimeoutError:
                capture.raise_if_failed()
                continue
            block_count += 1
            block.release()
    except Exception as exc:
        LOGGER.error(
            "Capture stability run failed at %d Hz / %d: %s",
            sample_rate,
            block_size,
            exc,
        )
        status = f"failed: {type(exc).__name__}"
    finally:
        wall_after = time.monotonic()
        cpu_after = os.times()
        capture.stop()

    cpu_seconds = (
        (cpu_after.user - cpu_before.user)
        + (cpu_after.system - cpu_before.system)
    )
    wall_seconds = wall_after - wall_before
    cpu_load = (
        100.0 * cpu_seconds / wall_seconds
        if wall_seconds > 0
        else 0.0
    )
    intervals = capture.callback_interval_samples
    if intervals.size:
        callback_average = float(np.mean(intervals))
        callback_jitter = float(np.std(intervals))
        callback_min = float(np.min(intervals))
        callback_max = float(np.max(intervals))
    else:
        callback_average = None
        callback_jitter = None
        callback_min = None
        callback_max = None

    return StabilityReport(
        sample_rate=sample_rate,
        block_size=block_size,
        duration_seconds=duration_seconds,
        capture_blocks=block_count,
        capture_overflows=capture.overflow_count,
        callback_count=int(intervals.size),
        callback_average_ms=callback_average,
        callback_jitter_ms=callback_jitter,
        callback_min_ms=callback_min,
        callback_max_ms=callback_max,
        cpu_load_percent=cpu_load,
        status=status,
    )


def write_json(path: str, reports: list[StabilityReport]) -> None:
    with open(path, "w", encoding="utf-8") as output:
        json.dump(
            [asdict(report) for report in reports],
            output,
            indent=2,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure real-time capture stability"
    )
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--sample-rates",
        nargs="+",
        type=int,
        choices=[44100, 48000],
        default=[44100, 48000],
    )
    parser.add_argument(
        "--block-sizes",
        nargs="+",
        type=int,
        choices=[64, 128, 256, 512],
        default=[64, 128, 256, 512],
    )
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    device: int | str | None
    if args.input_device is None:
        device = None
    else:
        try:
            device = int(args.input_device)
        except ValueError:
            device = args.input_device

    reports: list[StabilityReport] = []
    for sample_rate in args.sample_rates:
        for block_size in args.block_sizes:
            print(
                f"Measuring {sample_rate} Hz / "
                f"{block_size} frames",
            )
            report = run_stability(
                device,
                sample_rate,
                block_size,
                args.duration,
            )
            reports.append(report)
            print(
                f"  status={report.status}, "
                f"blocks={report.capture_blocks}, "
                f"overflows={report.capture_overflows}, "
                f"jitter={report.callback_jitter_ms}",
            )

    if args.json_output:
        write_json(args.json_output, reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
