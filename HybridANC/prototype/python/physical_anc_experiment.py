"""Offline workflow for controlled physical-ANC experiment preparation.

This module never opens or starts audio playback. It validates recorded files,
estimates a secondary FIR/delay, computes residual metrics and writes a repeatable
offline report. Any future physical test must use a separate, explicit, calibrated
and safety-approved activation path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

try:
    from .secondary_path import estimate_secondary_path
except ImportError:
    from secondary_path import estimate_secondary_path


LOGGER = logging.getLogger("hybridanc.physical_anc")


@dataclass(frozen=True)
class PhysicalExperimentConfig:
    configuration: str
    sample_rate: int = 48_000
    fir_length: int = 256
    delay_search_limit: int = 512
    measurement_location: str = "unspecified"

    def __post_init__(self) -> None:
        if self.configuration not in ("A_existing_hardware", "B_future_measurement"):
            raise ValueError("configuration must identify A or B")
        if self.sample_rate not in (44_100, 48_000):
            raise ValueError("sample_rate must be 44100 or 48000 Hz")
        if not 32 <= self.fir_length <= 4096:
            raise ValueError("fir_length must be between 32 and 4096")
        if not 0 < self.delay_search_limit <= 4096:
            raise ValueError("delay_search_limit must be between 1 and 4096")


@dataclass(frozen=True)
class RecordedMetrics:
    rms_before: float
    rms_after: float
    reduction_db: float
    delay_ms: float | None
    delay_samples: int | None
    correlation_confidence: float
    normalized_mse: float | None
    sample_rate: int
    frame_count: int


def validate_mono_float(filename: str | Path, name: str) -> tuple[np.ndarray, int]:
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"{name} WAV file not found: {path}")
    samples, sample_rate = sf.read(
        str(path),
        dtype="float64",
        always_2d=True,
    )
    if samples.size == 0:
        raise ValueError(f"{name} recording is empty")
    if sample_rate not in (44_100, 48_000):
        raise ValueError(
            f"{name} sample rate must be 44100 or 48000 Hz"
        )
    mono = np.mean(samples, axis=1)
    if not np.all(np.isfinite(mono)):
        raise ValueError(f"{name} contains non-finite samples")
    return mono, int(sample_rate)


def align_length(reference: np.ndarray, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if reference.size != measurement.size:
        count = min(reference.size, measurement.size)
        return reference[:count], measurement[:count]
    return reference, measurement


def estimate_impulse_and_delay(
    excitation: np.ndarray,
    measurement: np.ndarray,
    sample_rate: int,
    fir_length: int = 256,
    delay_search_limit: int = 512,
) -> tuple[np.ndarray, int | None, float, float]:
    """Estimate FIR and peak delay from offline controlled recordings."""
    excitation = np.asarray(excitation, dtype=np.float64).reshape(-1)
    measurement = np.asarray(measurement, dtype=np.float64).reshape(-1)
    if excitation.size != measurement.size:
        excitation, measurement = align_length(excitation, measurement)
    if excitation.size < fir_length:
        raise ValueError("recording is shorter than the requested FIR length")

    impulse = estimate_secondary_path(
        excitation,
        measurement,
        fir_length,
    )
    peak_index = int(
        np.argmax(
            np.abs(
                impulse[:min(delay_search_limit, impulse.size)]
            )
        )
    )
    peak_absolute = float(np.abs(impulse[peak_index]))
    if peak_absolute <= np.finfo(np.float64).eps:
        return impulse, None, 0.0, 0.0

    predicted = signal.lfilter(impulse, [1.0], excitation)
    residual = measurement - predicted
    error_mse = float(np.mean(residual * residual))
    measurement_power = float(np.mean(measurement * measurement))
    normalized_mse = (
        error_mse / measurement_power
        if measurement_power > np.finfo(np.float64).eps
        else 0.0
    )

    correlation = signal.correlate(measurement, excitation, mode="full")
    lags = signal.correlation_lags(measurement.size, excitation.size)
    center = int(np.argmax(np.abs(correlation)))
    correlation_peak = float(np.abs(correlation[center]))
    denominator = (
        float(np.linalg.norm(measurement))
        * float(np.linalg.norm(excitation))
    )
    confidence = (
        correlation_peak / denominator
        if denominator > np.finfo(np.float64).eps
        else 0.0
    )
    lag = int(lags[center])
    if lag < 0:
        peak_index = peak_index
    return impulse, peak_index, confidence, normalized_mse


def calculate_residual_metrics(
    before: np.ndarray,
    after: np.ndarray,
    sample_rate: int,
    delay_samples: int | None = None,
) -> RecordedMetrics:
    before, after = align_length(
        np.asarray(before, dtype=np.float64).reshape(-1),
        np.asarray(after, dtype=np.float64).reshape(-1),
    )
    rms_before = float(np.sqrt(np.mean(before * before)))
    rms_after = float(np.sqrt(np.mean(after * after)))
    reduction_db = float(
        20.0 * np.log10(
            max(rms_after, np.finfo(np.float64).tiny)
            / max(rms_before, np.finfo(np.float64).tiny)
        )
    )
    return RecordedMetrics(
        rms_before=rms_before,
        rms_after=rms_after,
        reduction_db=reduction_db,
        delay_ms=(
            1000.0 * delay_samples / sample_rate
            if delay_samples is not None
            else None
        ),
        delay_samples=delay_samples,
        correlation_confidence=0.0,
        normalized_mse=None,
        sample_rate=sample_rate,
        frame_count=int(before.size),
    )


def load_calibration(
    excitation_path: str | Path,
    measurement_path: str | Path,
    config: PhysicalExperimentConfig,
) -> dict[str, object]:
    excitation, excitation_rate = validate_mono_float(
        excitation_path,
        "excitation",
    )
    measurement, measurement_rate = validate_mono_float(
        measurement_path,
        "measurement",
    )
    if excitation_rate != measurement_rate:
        raise ValueError("excitation and measurement sample rates do not match")
    if excitation_rate != config.sample_rate:
        raise ValueError("recording sample rate does not match experiment config")

    impulse, delay, confidence, normalized_mse = estimate_impulse_and_delay(
        excitation,
        measurement,
        config.sample_rate,
        config.fir_length,
        config.delay_search_limit,
    )
    peak_delay_ms = (
        1000.0 * delay / config.sample_rate
        if delay is not None
        else None
    )
    return {
        "configuration": config.configuration,
        "measurement_location": config.measurement_location,
        "sample_rate": config.sample_rate,
        "fir_length": config.fir_length,
        "impulse_response": impulse.tolist(),
        "estimated_delay_samples": delay,
        "estimated_delay_ms": peak_delay_ms,
        "correlation_confidence": confidence,
        "normalized_mse": normalized_mse,
        "physical_output_enabled": False,
        "hardware_cancellation_claim": False,
    }


def load_residual_metrics(
    before_path: str | Path,
    after_path: str | Path,
    config: PhysicalExperimentConfig,
) -> RecordedMetrics:
    before, before_rate = validate_mono_float(before_path, "before")
    after, after_rate = validate_mono_float(after_path, "after")
    if before_rate != after_rate or before_rate != config.sample_rate:
        raise ValueError("before/after recordings must share the experiment sample rate")
    return calculate_residual_metrics(
        before,
        after,
        config.sample_rate,
    )


TEMPLATE_MARKDOWN = """# Controlled Physical ANC Experiment Report

**Status: no measured hardware result is assumed.**

## Safety Gate
- [ ] Earphones are not worn on a person's ears.
- [ ] Output is low-level and controlled.
- [ ] Emergency stop is available.
- [ ] Secondary measurement path is validated.
- [ ] physical output activation requires separate explicit authorization.

## Configuration
- Configuration A: existing laptop microphone, wired-earphone speakers,
  no independent validated residual-error microphone.
- Configuration B: external reference microphone, earphone speaker secondary
  source, separate measurement microphone at the listening location.
- Experiment ID:
- Date/time:
- Operator:
- Device IDs:
- Sample rate:
- Block size:
- Calibration ID:

## Hardware
- Reference microphone:
- Earphone speaker location:
- Measurement microphone:
- Inline microphone role:
- Noise source:

## Calibration
- Excitation WAV:
- Measurement WAV:
- FIR length:
- Estimated delay:
- Confidence:
- Normalized MSE:
- Repeatability:

## Residual Measurements
| Trial | Noise | Before WAV | After WAV | RMS before | RMS after | Reduction dB |
|---|---|---|---|---:|---:|---:|

## Measured Results

## Simulation Results

## Limitations and Stop Conditions

## Final Verdict
- [ ] Insufficient hardware
- [ ] Calibration incomplete
- [ ] Simulation only
- [ ] Repeatable benefit measured
- [ ] No repeatable benefit
- [ ] Unsafe to continue
"""


def write_template(path: str | Path) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(TEMPLATE_MARKDOWN, encoding="utf-8")
    return report_path


def write_calibration_csv(
    path: str | Path,
    result: dict[str, object],
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "configuration",
        "measurement_location",
        "sample_rate",
        "fir_length",
        "estimated_delay_samples",
        "estimated_delay_ms",
        "correlation_confidence",
        "normalized_mse",
        "physical_output_enabled",
        "hardware_cancellation_claim",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                key: result[key]
                for key in fields
            }
        )
    return report_path


def write_calibration_json(
    path: str | Path,
    result: dict[str, object],
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare/offline-analyze a controlled physical ANC experiment"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    template_parser = subparsers.add_parser(
        "template",
        help="Create a detailed physical experiment report template",
    )
    template_parser.add_argument("--output", default="docs/PHYSICAL_ANC_EXPERIMENT_REPORT.md")

    calibration_parser = subparsers.add_parser(
        "calibrate",
        help="Analyze prerecorded excitation and measurement WAV files",
    )
    calibration_parser.add_argument("--configuration", choices=["A_existing_hardware", "B_future_measurement"], required=True)
    calibration_parser.add_argument("--excitation", required=True)
    calibration_parser.add_argument("--measurement", required=True)
    calibration_parser.add_argument("--sample-rate", type=int, choices=[44100, 48000], default=48000)
    calibration_parser.add_argument("--fir-length", type=int, default=256)
    calibration_parser.add_argument("--delay-limit", type=int, default=512)
    calibration_parser.add_argument("--measurement-location", default="unspecified")
    calibration_parser.add_argument("--json-output", default=None)
    calibration_parser.add_argument("--csv-output", default=None)

    residual_parser = subparsers.add_parser(
        "residual",
        help="Compare prerecorded baseline and residual WAV files",
    )
    residual_parser.add_argument("--before", required=True)
    residual_parser.add_argument("--after", required=True)
    residual_parser.add_argument("--sample-rate", type=int, choices=[44100, 48000], default=48000)
    residual_parser.add_argument("--json-output", default=None)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    if args.command == "template":
        path = write_template(args.output)
        print(f"Created physical ANC report template: {path}")
        return 0

    if args.command == "calibrate":
        config = PhysicalExperimentConfig(
            configuration=args.configuration,
            sample_rate=args.sample_rate,
            fir_length=args.fir_length,
            delay_search_limit=args.delay_limit,
            measurement_location=args.measurement_location,
        )
        result = load_calibration(
            args.excitation,
            args.measurement,
            config,
        )
        print(
            f"Estimated delay: "
            f"{result['estimated_delay_samples']} samples "
            f"({result['estimated_delay_ms']:.3f} ms)"
            if result["estimated_delay_samples"] is not None
            else "Estimated delay: unavailable"
        )
        print(
            f"Correlation confidence: "
            f"{result['correlation_confidence']:.6f}"
        )
        print(f"Normalized MSE: {result['normalized_mse']:.6f}")
        print("Physical output enabled: False")
        if args.json_output:
            write_calibration_json(args.json_output, result)
        if args.csv_output:
            write_calibration_csv(args.csv_output, result)
        return 0

    if args.command == "residual":
        config = PhysicalExperimentConfig(
            configuration="B_future_measurement",
            sample_rate=args.sample_rate,
        )
        metrics = load_residual_metrics(
            args.before,
            args.after,
            config,
        )
        print(f"RMS before: {metrics.rms_before:.8f}")
        print(f"RMS after: {metrics.rms_after:.8f}")
        print(f"Reduction: {metrics.reduction_db:.3f} dB")
        if args.json_output:
            Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_output).write_text(
                json.dumps(asdict(metrics), indent=2),
                encoding="utf-8",
            )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
