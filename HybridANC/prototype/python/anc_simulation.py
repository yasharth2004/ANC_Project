"""Offline mathematical laboratory for ideal and mismatched cancellation.

This module operates on synthetic or recorded arrays only. It never opens an
output device and never sends generated anti-noise to headphones.

Ideal mathematical cancellation means that a negative sample is added to the
same sample of the original signal. Physical acoustic cancellation additionally
requires a secondary path, calibrated error sensor, phase/path measurement,
latency and acoustic environment validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


LOGGER = logging.getLogger("hybridanc.anc_simulation")


@dataclass(frozen=True)
class ANCMismatchConfig:
    """Parameters controlling an offline anti-noise experiment."""

    amplitude_ratio: float = 1.0
    phase_shift_degrees: float = 0.0
    delay_samples: int = 0
    invert: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.amplitude_ratio) or self.amplitude_ratio < 0:
            raise ValueError("amplitude_ratio must be finite and non-negative")
        if not np.isfinite(self.phase_shift_degrees):
            raise ValueError("phase_shift_degrees must be finite")
        if not isinstance(self.delay_samples, int) or isinstance(
            self.delay_samples,
            bool,
        ):
            raise TypeError("delay_samples must be an integer")
        if abs(self.delay_samples) > 100_000:
            raise ValueError("delay_samples magnitude is unexpectedly large")


@dataclass(frozen=True)
class CancellationResult:
    original: np.ndarray
    antinoise: np.ndarray
    residual: np.ndarray
    rms_before: float
    rms_after: float
    attenuation_db: float


@dataclass(frozen=True)
class ExperimentReport:
    name: str
    sample_rate: int
    duration_seconds: float
    rms_before: float
    rms_after: float
    attenuation_db: float
    amplitude_ratio: float
    phase_shift_degrees: float
    delay_samples: int


def validate_signal(
    signal: object,
    *,
    name: str = "signal",
) -> np.ndarray:
    """Return a finite mono float64 signal."""
    samples = np.asarray(signal)
    if samples.size == 0:
        raise ValueError(f"{name} must not be empty")
    if samples.ndim == 2:
        if samples.shape[1] not in (1, 2):
            raise ValueError(f"{name} must be mono or stereo")
        samples = np.mean(samples, axis=1)
    elif samples.ndim != 1:
        raise ValueError(f"{name} must be one- or two-dimensional")
    if not np.issubdtype(samples.dtype, np.floating):
        raise TypeError(f"{name} must contain floating-point samples")
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(samples, dtype=np.float64)


def generate_sine(
    frequency_hz: float,
    sample_rate: int = 48_000,
    duration_seconds: float = 1.0,
    amplitude: float = 1.0,
    phase_degrees: float = 0.0,
) -> np.ndarray:
    """Generate x[n] = A sin(2*pi*f*n/fs + phase)."""
    if not np.isfinite(frequency_hz) or frequency_hz < 0:
        raise ValueError("frequency_hz must be finite and non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if frequency_hz >= sample_rate / 2.0:
        raise ValueError("frequency_hz must be below Nyquist")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    if not np.isfinite(phase_degrees):
        raise ValueError("phase_degrees must be finite")

    frame_count = int(sample_rate * duration_seconds)
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    phase = np.deg2rad(phase_degrees)
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * time + phase)


def generate_multi_frequency_noise(
    frequencies_hz: Iterable[float],
    amplitudes: Iterable[float] | None = None,
    sample_rate: int = 48_000,
    duration_seconds: float = 1.0,
) -> np.ndarray:
    """Combine multiple sine components at the requested frequencies."""
    frequencies = np.asarray(tuple(frequencies_hz), dtype=np.float64)
    if frequencies.size == 0:
        raise ValueError("at least one frequency is required")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies must be finite")
    if np.any(frequencies < 0) or np.any(frequencies >= sample_rate / 2.0):
        raise ValueError("all frequencies must be between zero and Nyquist")

    if amplitudes is None:
        amplitude_values = np.ones(frequencies.size, dtype=np.float64)
    else:
        amplitude_values = np.asarray(tuple(amplitudes), dtype=np.float64)
        if amplitude_values.shape != frequencies.shape:
            raise ValueError("amplitudes must match frequencies")
        if not np.all(np.isfinite(amplitude_values)):
            raise ValueError("amplitudes must be finite")

    result = np.zeros(
        int(sample_rate * duration_seconds),
        dtype=np.float64,
    )
    for frequency, amplitude in zip(frequencies, amplitude_values):
        result += generate_sine(
            frequency_hz=float(frequency),
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            amplitude=float(amplitude),
        )
    return result


def generate_white_noise(
    sample_rate: int = 48_000,
    duration_seconds: float = 1.0,
    amplitude: float = 0.2,
    seed: int | None = None,
) -> np.ndarray:
    """Generate reproducible white noise from a normal distribution."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("amplitude must be finite and non-negative")
    frame_count = int(sample_rate * duration_seconds)
    rng = np.random.default_rng(seed)
    return amplitude * rng.normal(size=frame_count)


def load_recorded_noise(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an offline recording and return mono samples plus sample rate."""
    recording, sample_rate = sf.read(str(path), dtype="float64", always_2d=True)
    if recording.size == 0:
        raise ValueError("recording is empty")
    mono = np.mean(recording, axis=1)
    if not np.all(np.isfinite(mono)):
        raise ValueError("recording contains non-finite samples")
    return mono, int(sample_rate)


def apply_phase_shift(
    signal: np.ndarray,
    sample_rate: int,
    phase_degrees: float,
) -> np.ndarray:
    """Shift a real sinusoidal signal in frequency domain.

    This helper is an idealized phase-shift experiment. It is not a claim
    that a physical signal can be phase-shifted independently of path response.
    """
    samples = validate_signal(signal)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not np.isfinite(phase_degrees):
        raise ValueError("phase_degrees must be finite")
    spectrum = np.fft.rfft(samples)
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    shift_radians = np.deg2rad(phase_degrees)
    shifted = spectrum * np.exp(-1j * shift_radians)
    return np.fft.irfft(shifted, n=samples.size)


def adjust_amplitude(signal: np.ndarray, amplitude_ratio: float) -> np.ndarray:
    samples = validate_signal(signal)
    if not np.isfinite(amplitude_ratio) or amplitude_ratio < 0:
        raise ValueError("amplitude_ratio must be finite and non-negative")
    return samples * amplitude_ratio


def apply_time_delay(
    signal: np.ndarray,
    delay_samples: int,
) -> np.ndarray:
    """Shift samples by delay; positive delay means later output."""
    samples = validate_signal(signal)
    if not isinstance(delay_samples, int) or isinstance(delay_samples, bool):
        raise TypeError("delay_samples must be an integer")
    if abs(delay_samples) > len(samples):
        raise ValueError("delay_samples magnitude exceeds signal length")

    shifted = np.zeros_like(samples)
    if delay_samples >= 0:
        shifted[delay_samples:] = samples[:-delay_samples] if delay_samples else samples
    else:
        shifted[:delay_samples] = samples[-delay_samples:]
    return shifted


def generate_antinoise(
    noise: np.ndarray,
    mismatch: ANCMismatchConfig,
    sample_rate: int,
) -> np.ndarray:
    """Generate a mismatched candidate anti-noise array."""
    samples = validate_signal(noise)
    antinoise = adjust_amplitude(samples, mismatch.amplitude_ratio)
    if mismatch.phase_shift_degrees:
        antinoise = apply_phase_shift(
            antinoise,
            sample_rate,
            mismatch.phase_shift_degrees,
        )
    if mismatch.delay_samples:
        antinoise = apply_time_delay(antinoise, mismatch.delay_samples)
    if mismatch.invert:
        antinoise = -antinoise
    return antinoise


def combine_signals(*signals: np.ndarray) -> np.ndarray:
    """Combine equal-length mono signals sample by sample."""
    if not signals:
        raise ValueError("at least one signal is required")
    validated = [validate_signal(item) for item in signals]
    lengths = {item.size for item in validated}
    if len(lengths) != 1:
        raise ValueError("all signals must have the same length")
    return np.sum(validated, axis=0)


def calculate_residual(
    original_noise: np.ndarray,
    antinoise: np.ndarray,
) -> np.ndarray:
    """Return original + candidate anti-noise."""
    return combine_signals(original_noise, antinoise)


def rms_amplitude(signal: np.ndarray) -> float:
    samples = validate_signal(signal)
    return float(np.sqrt(np.mean(np.square(samples))))


def attenuation_db(before: np.ndarray, after: np.ndarray) -> float:
    before_rms = rms_amplitude(before)
    after_rms = rms_amplitude(after)
    return float(
        20.0 * np.log10(
            max(after_rms, np.finfo(np.float64).tiny)
            / max(before_rms, np.finfo(np.float64).tiny)
        )
    )


def run_cancellation_experiment(
    noise: np.ndarray,
    sample_rate: int,
    mismatch: ANCMismatchConfig,
) -> CancellationResult:
    original = validate_signal(noise)
    antinoise = generate_antinoise(original, mismatch, sample_rate)
    residual = calculate_residual(original, antinoise)
    before = rms_amplitude(original)
    after = rms_amplitude(residual)
    attenuation = float(
        20.0 * np.log10(
            max(after, np.finfo(np.float64).tiny)
            / max(before, np.finfo(np.float64).tiny)
        )
    )
    return CancellationResult(
        original=original,
        antinoise=antinoise,
        residual=residual,
        rms_before=before,
        rms_after=after,
        attenuation_db=attenuation,
    )


def experiment_suite(
    sample_rate: int = 48_000,
    duration_seconds: float = 0.5,
) -> dict[str, CancellationResult]:
    """Run experiments A-F entirely on synthetic offline signals."""
    sine_100 = generate_sine(
        100.0,
        sample_rate,
        duration_seconds,
    )
    ideal = run_cancellation_experiment(
        sine_100,
        sample_rate,
        ANCMismatchConfig(),
    )
    amplitude_mismatch = run_cancellation_experiment(
        sine_100,
        sample_rate,
        ANCMismatchConfig(amplitude_ratio=0.5),
    )
    phase_mismatch = run_cancellation_experiment(
        sine_100,
        sample_rate,
        ANCMismatchConfig(phase_shift_degrees=45.0),
    )
    delay_mismatch = run_cancellation_experiment(
        sine_100,
        sample_rate,
        ANCMismatchConfig(delay_samples=max(1, int(sample_rate * 0.001))),
    )
    multi_noise = generate_multi_frequency_noise(
        (100.0, 300.0, 1000.0),
        (1.0, 0.5, 0.25),
        sample_rate,
        duration_seconds,
    )
    multi_ideal = run_cancellation_experiment(
        multi_noise,
        sample_rate,
        ANCMismatchConfig(),
    )

    return {
        "A_100Hz_original": run_cancellation_experiment(
            sine_100,
            sample_rate,
            ANCMismatchConfig(invert=False),
        ),
        "B_theoretical_inverted": ideal,
        "C_amplitude_mismatch": amplitude_mismatch,
        "D_phase_mismatch": phase_mismatch,
        "E_time_delay": delay_mismatch,
        "F_multiple_frequencies": multi_ideal,
    }


def run_visualization(
    sample_rate: int,
    duration_seconds: float,
    experiment_name: str = "B_theoretical_inverted",
) -> None:
    """Plot one offline experiment and its spectrum."""
    import matplotlib.pyplot as plt

    experiments = experiment_suite(sample_rate, duration_seconds)
    result = experiments[experiment_name]
    time_axis = np.arange(result.original.size) / sample_rate
    frequencies = np.fft.rfftfreq(result.original.size, d=1.0 / sample_rate)
    original_spectrum = np.abs(np.fft.rfft(result.original))
    residual_spectrum = np.abs(np.fft.rfft(result.residual))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(time_axis, result.original, lw=0.8, label="Original")
    axes[0, 0].plot(time_axis, result.antinoise, lw=0.8, label="Anti-noise")
    axes[0, 0].set_title("Original noise and anti-noise")
    axes[0, 0].set_xlabel("Time (seconds)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.2)

    axes[0, 1].plot(time_axis, result.residual, lw=0.8)
    axes[0, 1].set_title(
        f"Combined residual | RMS {result.rms_after:.6f} "
        f"| attenuation {result.attenuation_db:.2f} dB"
    )
    axes[0, 1].set_xlabel("Time (seconds)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.2)

    axes[1, 0].plot(
        frequencies,
        20.0 * np.log10(
            np.maximum(original_spectrum, np.finfo(np.float64).tiny)
        ),
        lw=0.8,
    )
    axes[1, 0].set_title("Spectrum before combination")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Relative magnitude")
    axes[1, 0].set_xlim(0.0, min(3000.0, sample_rate / 2.0))

    axes[1, 1].plot(
        frequencies,
        20.0 * np.log10(
            np.maximum(residual_spectrum, np.finfo(np.float64).tiny)
        ),
        lw=0.8,
    )
    axes[1, 1].set_title("Spectrum after combination")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Relative magnitude")
    axes[1, 1].set_xlim(0.0, min(3000.0, sample_rate / 2.0))
    fig.suptitle(
        f"{experiment_name}: ideal mathematical result only",
        fontweight="bold",
    )
    plt.show()


def build_experiment_reports(
    sample_rate: int,
    duration_seconds: float,
) -> list[ExperimentReport]:
    """Build experiment metadata and measured synthetic RMS values."""
    experiments = experiment_suite(sample_rate, duration_seconds)
    mismatches = {
        "A_100Hz_original": ANCMismatchConfig(invert=False),
        "B_theoretical_inverted": ANCMismatchConfig(),
        "C_amplitude_mismatch": ANCMismatchConfig(amplitude_ratio=0.5),
        "D_phase_mismatch": ANCMismatchConfig(phase_shift_degrees=45.0),
        "E_time_delay": ANCMismatchConfig(
            delay_samples=max(1, int(sample_rate * 0.001))
        ),
        "F_multiple_frequencies": ANCMismatchConfig(),
    }
    reports: list[ExperimentReport] = []
    for name, result in experiments.items():
        mismatch = mismatches[name]
        reports.append(
            ExperimentReport(
                name=name,
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
                rms_before=result.rms_before,
                rms_after=result.rms_after,
                attenuation_db=result.attenuation_db,
                amplitude_ratio=mismatch.amplitude_ratio,
                phase_shift_degrees=mismatch.phase_shift_degrees,
                delay_samples=mismatch.delay_samples,
            )
        )
    return reports


def write_report(path: str | Path, reports: Iterable[ExperimentReport]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for report in reports:
        payload.append(
            {
                "name": report.name,
                "sample_rate": report.sample_rate,
                "duration_seconds": report.duration_seconds,
                "rms_before": report.rms_before,
                "rms_after": report.rms_after,
                "attenuation_db": report.attenuation_db,
                "amplitude_ratio": report.amplitude_ratio,
                "phase_shift_degrees": report.phase_shift_degrees,
                "delay_samples": report.delay_samples,
                "measurement_type": "ideal_mathematical_offline",
            }
        )
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline ideal and mismatched ANC mathematical laboratory"
    )
    parser.add_argument("--sample-rate", type=int, choices=[44100, 48000], default=48000)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument(
        "--experiment",
        choices=[
            "A_100Hz_original",
            "B_theoretical_inverted",
            "C_amplitude_mismatch",
            "D_phase_mismatch",
            "E_time_delay",
            "F_multiple_frequencies",
        ],
        default="B_theoretical_inverted",
    )
    parser.add_argument("--json-report", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    if args.duration <= 0:
        parser.error("duration must be positive")

    experiments = experiment_suite(args.sample_rate, args.duration)
    result = experiments[args.experiment]
    print("Ideal mathematical experiment; no audio output is active.")
    print(f"Experiment: {args.experiment}")
    print(f"RMS before: {result.rms_before:.8f}")
    print(f"RMS after: {result.rms_after:.8f}")
    print(f"Attenuation: {result.attenuation_db:.3f} dB")

    if args.json_report:
        write_report(
            args.json_report,
            build_experiment_reports(args.sample_rate, args.duration),
        )

    run_visualization(args.sample_rate, args.duration, args.experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
