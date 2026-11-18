"""Audio pipeline latency and dropout measurement for HybridANC.

This module measures callback and bridge timing. It does not claim that these
timings equal acoustic round-trip latency. Acoustic loopback is reported only
when a controlled signal produces a measurable correlation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import signal

try:
    from .audio_capture import AudioCapture, AudioCaptureConfig
    from .audio_playback import AudioPlayback, AudioPlaybackConfig
except ImportError:
    from audio_capture import AudioCapture, AudioCaptureConfig
    from audio_playback import AudioPlayback, AudioPlaybackConfig


LOGGER = logging.getLogger("hybridanc.latency")
TEST_SAMPLE_RATES = (44_100, 48_000)
TEST_BLOCK_SIZES = (64, 128, 256, 512)


@dataclass(frozen=True)
class LatencyConfig:
    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: int = 48_000
    block_size: int = 512
    duration_seconds: float = 5.0
    playback_gain: float = 0.05
    loopback_chirp_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.sample_rate not in TEST_SAMPLE_RATES:
            raise ValueError("sample_rate must be 44100 or 48000 Hz")
        if self.block_size not in TEST_BLOCK_SIZES:
            raise ValueError("block_size must be 64, 128, 256 or 512")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if not 0.0 <= self.playback_gain <= 0.25:
            raise ValueError("playback_gain must be between 0.0 and 0.25")


@dataclass(frozen=True)
class LatencyReport:
    sample_rate: int
    block_size: int
    average_latency_ms: float
    minimum_latency_ms: float
    maximum_latency_ms: float
    jitter_ms: float
    average_processing_ms: float
    maximum_processing_ms: float
    cpu_load_percent: float
    dropout_count: int
    input_overflow_count: int
    output_underrun_count: int
    input_buffer_ms: float
    output_buffer_ms: float
    callback_count: int
    latency_sample_count: int
    processing_sample_count: int
    average_callback_interval_ms: float | None
    acoustic_round_trip_ms: float | None
    acoustic_loopback_confidence: float | None
    loopback_status: str


class LatencyProbe:
    """Measure capture-to-playback bridge timing for one format."""

    def __init__(self, config: LatencyConfig) -> None:
        self.config = config
        self.capture = AudioCapture(
            AudioCaptureConfig(
                device=config.input_device,
                sample_rate=config.sample_rate,
                block_size=config.block_size,
                channels=1,
                queue_depth=32,
            )
        )
        self.playback = AudioPlayback(
            AudioPlaybackConfig(
                device=config.output_device,
                sample_rate=config.sample_rate,
                block_size=config.block_size,
                channels=1,
                gain=config.playback_gain,
                queue_depth=32,
            )
        )
        self._routing_latency_ms: list[float] = []
        self._processing_ms: list[float] = []
        self._callback_intervals_ms: list[float] = []

    def run(self) -> LatencyReport:
        routing_latency = self._routing_latency_ms
        processing_times = self._processing_ms
        deadline = time.monotonic() + self.config.duration_seconds
        cpu_before = os.times()
        wall_before = time.monotonic()

        try:
            self.playback.start()
            self.capture.start()
            while time.monotonic() < deadline:
                processing_start_ns = time.monotonic_ns()
                try:
                    block = self.capture.read(timeout=0.05)
                except TimeoutError:
                    self.capture.raise_if_failed()
                    self.playback.raise_if_failed()
                    continue

                routing_start_ns = getattr(
                    block,
                    "capture_time",
                    time.monotonic(),
                )
                if isinstance(routing_start_ns, float):
                    routing_start_ns = int(routing_start_ns * 1_000_000_000)

                try:
                    self.playback.write(block.samples, timeout=0.05)
                finally:
                    block.release()

                processing_end_ns = time.monotonic_ns()
                processing_times.append(
                    (processing_end_ns - processing_start_ns) / 1_000_000.0
                )

                if hasattr(block, "capture_time"):
                    captured_ns = int(block.capture_time * 1_000_000_000)
                    routing_latency.append(
                        max(
                            0.0,
                            (processing_end_ns - captured_ns) / 1_000_000.0,
                        )
                    )

                capture_interval = self.capture.callback_interval_ms
                playback_interval = self.playback.callback_interval_ms
                if capture_interval is not None:
                    self._callback_intervals_ms.append(capture_interval)
                if playback_interval is not None:
                    self._callback_intervals_ms.append(playback_interval)
        finally:
            wall_after = time.monotonic()
            cpu_after = os.times()
            self.capture.stop()
            self.playback.stop()

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

        return build_latency_report(
            sample_rate=self.config.sample_rate,
            block_size=self.config.block_size,
            routing_latency=routing_latency,
            processing_times=processing_times,
            callback_intervals=self._callback_intervals_ms,
            cpu_load_percent=cpu_load,
            input_overflow_count=self.capture.overflow_count,
            output_underrun_count=self.playback.underrun_count,
        )


def build_latency_report(
    *,
    sample_rate: int,
    block_size: int,
    routing_latency: Iterable[float],
    processing_times: Iterable[float],
    callback_intervals: Iterable[float],
    cpu_load_percent: float,
    input_overflow_count: int,
    output_underrun_count: int,
    acoustic_round_trip_ms: float | None = None,
    acoustic_loopback_confidence: float | None = None,
    loopback_status: str = "unavailable",
) -> LatencyReport:
    """Construct a report from observed values only."""
    latency = np.asarray(list(routing_latency), dtype=np.float64)
    processing = np.asarray(list(processing_times), dtype=np.float64)
    callbacks = np.asarray(list(callback_intervals), dtype=np.float64)

    if latency.size:
        average_latency = float(np.mean(latency))
        minimum_latency = float(np.min(latency))
        maximum_latency = float(np.max(latency))
        jitter = float(np.std(latency))
    else:
        average_latency = 0.0
        minimum_latency = 0.0
        maximum_latency = 0.0
        jitter = 0.0

    if processing.size:
        average_processing = float(np.mean(processing))
        maximum_processing = float(np.max(processing))
    else:
        average_processing = 0.0
        maximum_processing = 0.0

    callback_average = (
        float(np.mean(callbacks)) if callbacks.size else None
    )
    dropout_count = input_overflow_count + output_underrun_count

    return LatencyReport(
        sample_rate=sample_rate,
        block_size=block_size,
        average_latency_ms=average_latency,
        minimum_latency_ms=minimum_latency,
        maximum_latency_ms=maximum_latency,
        jitter_ms=jitter,
        average_processing_ms=average_processing,
        maximum_processing_ms=maximum_processing,
        cpu_load_percent=float(cpu_load_percent),
        dropout_count=dropout_count,
        input_overflow_count=input_overflow_count,
        output_underrun_count=output_underrun_count,
        input_buffer_ms=block_size * 1000.0 / sample_rate,
        output_buffer_ms=block_size * 1000.0 / sample_rate,
        callback_count=int(callbacks.size),
        latency_sample_count=int(latency.size),
        processing_sample_count=int(processing.size),
        average_callback_interval_ms=callback_average,
        acoustic_round_trip_ms=acoustic_round_trip_ms,
        acoustic_loopback_confidence=acoustic_loopback_confidence,
        loopback_status=loopback_status,
    )


def estimate_loopback_delay(
    reference: np.ndarray,
    recording: np.ndarray,
    sample_rate: int,
    minimum_confidence: float = 0.35,
) -> tuple[float | None, float]:
    """Estimate delay using cross-correlation of a controlled chirp."""
    reference_mono = np.asarray(reference, dtype=np.float64).reshape(-1)
    recording_mono = np.asarray(recording, dtype=np.float64).reshape(-1)
    if reference_mono.size == 0 or recording_mono.size == 0:
        return None, 0.0

    correlation = signal.correlate(recording_mono, reference_mono, mode="full")
    lag = signal.correlation_lags(recording_mono.size, reference_mono.size)
    nonnegative = lag >= 0
    lag = lag[nonnegative]
    correlation = correlation[nonnegative]
    if correlation.size == 0:
        return None, 0.0

    peak_index = int(np.argmax(np.abs(correlation)))
    peak = float(np.abs(correlation[peak_index]))
    reference_power = float(np.linalg.norm(reference_mono))
    recording_power = float(np.linalg.norm(recording_mono))
    denominator = reference_power * recording_power
    confidence = peak / denominator if denominator > 0 else 0.0
    if confidence < minimum_confidence:
        return None, confidence

    delay_seconds = float(lag[peak_index] / sample_rate)
    return delay_seconds * 1000.0, confidence


def generate_chirp(
    sample_rate: int,
    duration_seconds: float,
    start_hz: float = 250.0,
    end_hz: float = 8_000.0,
) -> np.ndarray:
    frames = int(sample_rate * duration_seconds)
    if frames <= 0:
        raise ValueError("loopback chirp duration must be positive")
    times = np.arange(frames, dtype=np.float64) / sample_rate
    chirp = signal.chirp(
        times,
        f0=start_hz,
        f1=end_hz,
        t1=duration_seconds,
        method="logarithmic",
    )
    return np.asarray(chirp * 0.05, dtype=np.float32)


def run_controlled_loopback(config: LatencyConfig) -> tuple[float | None, float, str]:
    """Emit and capture a controlled chirp for optional correlation."""
    capture = AudioCapture(
        AudioCaptureConfig(
            device=config.input_device,
            sample_rate=config.sample_rate,
            block_size=config.block_size,
            channels=1,
            queue_depth=64,
        )
    )
    playback = AudioPlayback(
        AudioPlaybackConfig(
            device=config.output_device,
            sample_rate=config.sample_rate,
            block_size=config.block_size,
            channels=1,
            gain=config.playback_gain,
            queue_depth=64,
        )
    )
    chirp = generate_chirp(
        config.sample_rate,
        config.loopback_chirp_seconds,
    )
    recording: list[np.ndarray] = []

    try:
        playback.start()
        playback.clear()
        capture.start()
        playback.clear()

        for offset in range(0, len(chirp), config.block_size):
            block = chirp[offset:offset + config.block_size]
            if len(block) < config.block_size:
                block = np.pad(
                    block,
                    (0, config.block_size - len(block)),
                    mode="constant",
                )
            playback.write(block, timeout=0.25)

        deadline = time.monotonic() + config.loopback_chirp_seconds + 1.0
        while time.monotonic() < deadline:
            try:
                captured = capture.read(timeout=0.1)
            except TimeoutError:
                continue
            recording.append(captured.samples.copy())
            captured.release()
    except Exception as exc:
        return None, 0.0, f"failed: {type(exc).__name__}: {exc}"
    finally:
        playback.stop()
        capture.stop()

    if not recording:
        return None, 0.0, "no input samples captured"

    recorded_signal = np.concatenate(recording)
    delay_ms, confidence = estimate_loopback_delay(
        chirp,
        recorded_signal,
        config.sample_rate,
    )
    if delay_ms is None:
        return None, confidence, "no reliable chirp correlation"
    return delay_ms, confidence, "controlled chirp correlation"


def write_json_report(path: str | Path, reports: Iterable[LatencyReport]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(report) for report in reports]
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv_report(path: str | Path, reports: Iterable[LatencyReport]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(reports)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure HybridANC digital audio pipeline latency"
    )
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--gain", type=float, default=0.05)
    parser.add_argument(
        "--loopback",
        action="store_true",
        help="Attempt controlled chirp correlation; acoustic measurement requires wired/controlled hardware",
    )
    parser.add_argument("--json-report", default=None)
    parser.add_argument("--csv-report", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    def parse_device(value: str | None) -> int | str | None:
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return value

    input_device = parse_device(args.input_device)
    output_device = parse_device(args.output_device)
    reports: list[LatencyReport] = []

    for sample_rate in TEST_SAMPLE_RATES:
        for block_size in TEST_BLOCK_SIZES:
            print(
                f"\nMeasuring {sample_rate} Hz / "
                f"{block_size} samples",
            )
            config = LatencyConfig(
                input_device=input_device,
                output_device=output_device,
                sample_rate=sample_rate,
                block_size=block_size,
                duration_seconds=args.duration,
                playback_gain=args.gain,
            )
            try:
                report = LatencyProbe(config).run()
            except Exception as exc:
                LOGGER.error(
                    "Unable to measure %d Hz / %d blocks: %s",
                    sample_rate,
                    block_size,
                    exc,
                )
                report = build_latency_report(
                    sample_rate=sample_rate,
                    block_size=block_size,
                    routing_latency=[],
                    processing_times=[],
                    callback_intervals=[],
                    cpu_load_percent=0.0,
                    input_overflow_count=0,
                    output_underrun_count=0,
                    loopback_status=(
                        "failed: "
                        f"{type(exc).__name__}"
                    ),
                )
            reports.append(report)

    print("\nLatency report")
    print(
        "SR   BLOCK  AVG MS  MIN MS  MAX MS  JITTER MS  "
        "PROCESS MS  CPU %  DROPOUTS"
    )
    for report in reports:
        if report.latency_sample_count:
            average_latency = f"{report.average_latency_ms:.3f}"
            minimum_latency = f"{report.minimum_latency_ms:.3f}"
            maximum_latency = f"{report.maximum_latency_ms:.3f}"
            jitter = f"{report.jitter_ms:.3f}"
        else:
            average_latency = "N/A"
            minimum_latency = "N/A"
            maximum_latency = "N/A"
            jitter = "N/A"
        processing = (
            f"{report.average_processing_ms:.3f}"
            if report.processing_sample_count
            else "N/A"
        )
        print(
            f"{report.sample_rate:<5} "
            f"{report.block_size:<6} "
            f"{average_latency:<7} "
            f"{minimum_latency:<6} "
            f"{maximum_latency:<6} "
            f"{jitter:<10} "
            f"{processing:<11} "
            f"{report.cpu_load_percent:<6.1f} "
            f"{report.dropout_count}"
        )

    if args.loopback:
        print("\nRunning controlled loopback chirps")
        loopback_reports: list[LatencyReport] = []
        for report in reports:
            config = LatencyConfig(
                input_device=input_device,
                output_device=output_device,
                sample_rate=report.sample_rate,
                block_size=report.block_size,
                duration_seconds=args.duration,
                playback_gain=args.gain,
            )
            try:
                delay_ms, confidence, status = run_controlled_loopback(config)
            except Exception as exc:
                delay_ms = None
                confidence = 0.0
                status = f"failed: {type(exc).__name__}: {exc}"
            report_dict = asdict(report)
            report_dict["acoustic_round_trip_ms"] = delay_ms
            report_dict["acoustic_loopback_confidence"] = confidence
            report_dict["loopback_status"] = status
            loopback_reports.append(LatencyReport(**report_dict))
        reports = loopback_reports

    if args.json_report:
        write_json_report(args.json_report, reports)
    if args.csv_report:
        write_csv_report(args.csv_report, reports)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
