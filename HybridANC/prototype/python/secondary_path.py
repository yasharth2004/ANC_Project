"""Secondary acoustic-path modeling for HybridANC ANC simulations.

The secondary path is modeled as a causal finite impulse response:

    y[n] = sum(k=0..M-1) h[k] * x[n-k]

Variables:
* x[n]: excitation sent to the simulated output/driver.
* h[k]: FIR impulse-response coefficient at tap k.
* y[n]: signal at the modeled measurement location.
* M: number of FIR taps.
* fs: sample rate.

This model is mathematical. It does not represent an actual earphone,
acoustic environment or measurement microphone until real controlled hardware
measurements have been validated.

It includes no live output and no playback call.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import signal


LOGGER = logging.getLogger("hybridanc.secondary_path")
SUPPORTED_SAMPLE_RATES = (44_100, 48_000)


@dataclass(frozen=True)
class FIRSecondaryPathConfig:
    sample_rate: int = 48_000
    fir_length: int = 128
    delay_samples: int = 16
    scale: float = 1.0
    resonance_hz: float | None = 1_000.0
    resonance_q: float = 8.0
    noise_gain: float = 0.0
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError("sample_rate must be 44100 or 48000 Hz")
        if not isinstance(self.fir_length, int) or not 1 <= self.fir_length <= 4096:
            raise ValueError("fir_length must be between 1 and 4096")
        if not isinstance(self.delay_samples, int):
            raise TypeError("delay_samples must be an integer")
        if self.delay_samples < 0 or self.delay_samples >= self.fir_length:
            raise ValueError("delay_samples must be within the FIR length")
        if not np.isfinite(self.scale):
            raise ValueError("scale must be finite")
        if not np.isfinite(self.resonance_q) or self.resonance_q <= 0:
            raise ValueError("resonance_q must be finite and positive")
        if self.resonance_hz is not None:
            if not np.isfinite(self.resonance_hz):
                raise ValueError("resonance_hz must be finite")
            if not 0.0 < self.resonance_hz < self.sample_rate / 2.0:
                raise ValueError("resonance_hz must be below Nyquist")
        if not np.isfinite(self.noise_gain) or self.noise_gain < 0:
            raise ValueError("noise_gain must be finite and non-negative")


@dataclass(frozen=True)
class EstimationResult:
    impulse_response: np.ndarray
    true_impulse_response: np.ndarray
    measurement: np.ndarray
    excitation: np.ndarray
    normalized_mse: float
    impulse_peak_error: float
    delay_error_samples: int


@dataclass(frozen=True)
class SecondaryPathReport:
    name: str
    sample_rate: int
    fir_length: int
    configured_delay_samples: int
    estimated_delay_samples: int
    normalized_mse: float
    impulse_peak_error: float
    delay_error_samples: int


def validate_mono(signal_samples: object, name: str = "signal") -> np.ndarray:
    values = np.asarray(signal_samples)
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if values.ndim == 2:
        if values.shape[1] not in (1, 2):
            raise ValueError(f"{name} must be mono or two-channel")
        values = np.mean(values, axis=1)
    elif values.ndim != 1:
        raise ValueError(f"{name} must be mono or one-dimensional")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"{name} must contain floating-point samples")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(values, dtype=np.float64)


class FIRSecondaryPath:
    """Causal FIR representation of a modeled secondary path."""

    def __init__(
        self,
        impulse_response: np.ndarray,
        sample_rate: int = 48_000,
        *,
        source: str = "synthetic",
    ) -> None:
        impulse = validate_mono(impulse_response, "impulse_response")
        if impulse.ndim != 1:
            raise ValueError("impulse_response must be one-dimensional")
        if impulse.size == 0:
            raise ValueError("impulse_response must not be empty")
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError("sample_rate must be 44100 or 48000 Hz")
        if source not in ("synthetic", "controlled_measurement", "unknown"):
            raise ValueError("source must describe synthetic or measurement origin")
        self.impulse_response = impulse.copy()
        self.sample_rate = sample_rate
        self.source = source

    @property
    def fir_length(self) -> int:
        return self.impulse_response.size

    def convolve(self, excitation: np.ndarray) -> np.ndarray:
        """Apply y[n] = sum h[k]x[n-k] using causal zero padding."""
        samples = validate_mono(excitation, "excitation")
        output = signal.lfilter(
            self.impulse_response,
            [1.0],
            samples,
        )
        return np.asarray(output, dtype=np.float64)

    def apply(self, excitation: np.ndarray) -> np.ndarray:
        return self.convolve(excitation)

    def frequency_response(
        self,
        fft_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return frequency Hz, magnitude dB and phase radians."""
        if not isinstance(fft_size, int) or fft_size <= 0:
            raise ValueError("fft_size must be positive")
        frequencies, response = signal.freqz(
            self.impulse_response,
            worN=fft_size,
            fs=self.sample_rate,
        )
        magnitude_db = 20.0 * np.log10(
            np.maximum(
                np.abs(response),
                np.finfo(np.float64).tiny,
            )
        )
        phase_radians = np.angle(response)
        return frequencies, magnitude_db, phase_radians

    def peak_delay_samples(self) -> int:
        return int(np.argmax(np.abs(self.impulse_response)))

    def copy(self) -> "FIRSecondaryPath":
        return FIRSecondaryPath(
            self.impulse_response,
            self.sample_rate,
            source=self.source,
        )


def generate_synthetic_secondary_path(
    config: FIRSecondaryPathConfig,
) -> FIRSecondaryPath:
    """Generate a synthetic multi-tap secondary path."""
    impulse = np.zeros(config.fir_length, dtype=np.float64)

    main_index = config.delay_samples
    impulse[main_index] = config.scale

    if main_index + 1 < config.fir_length:
        impulse[main_index + 1] = -0.20 * config.scale
    if main_index + 4 < config.fir_length:
        impulse[main_index + 4] = 0.12 * config.scale
    if main_index + 9 < config.fir_length:
        impulse[main_index + 9] = -0.08 * config.scale

    if config.resonance_hz is not None:
        fraction = np.arange(config.fir_length, dtype=np.float64)
        time_seconds = fraction / config.sample_rate
        resonance = np.exp(
            -np.pi * config.resonance_hz / config.resonance_q * time_seconds
        ) * np.sin(
            2.0 * np.pi * config.resonance_hz * time_seconds
        )
        resonance *= 0.15 * config.scale
        impulse += resonance

    impulse /= max(float(np.max(np.abs(impulse))), np.finfo(np.float64).tiny)
    impulse *= config.scale
    return FIRSecondaryPath(
        impulse,
        config.sample_rate,
        source="synthetic",
    )


def generate_controlled_excitation(
    sample_rate: int,
    duration_seconds: float,
    seed: int = 2026,
    frequencies_hz: Iterable[float] | None = None,
) -> np.ndarray:
    """Generate low-level deterministic offline excitation."""
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError("sample_rate must be 44100 or 48000 Hz")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    frame_count = int(sample_rate * duration_seconds)
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    if frequencies_hz is None:
        rng = np.random.default_rng(seed)
        excitation = rng.normal(0.0, 1.0, frame_count)
    else:
        excitation = np.zeros(frame_count, dtype=np.float64)
        amplitudes = np.asarray(tuple(frequencies_hz), dtype=np.float64)
        if amplitudes.size == 0:
            raise ValueError("frequencies_hz must not be empty")
        for index, frequency in enumerate(amplitudes):
            if not np.isfinite(frequency) or not 0.0 < frequency < sample_rate / 2.0:
                raise ValueError("all excitation frequencies must be below Nyquist")
            excitation += np.sin(
                2.0 * np.pi * frequency * time + index * 0.37
            )
        excitation /= len(amplitudes)

    peak = float(np.max(np.abs(excitation)))
    if peak > 0:
        excitation *= 0.05 / peak
    return excitation


def add_measurement_noise(
    measurement: np.ndarray,
    noise_gain: float,
    seed: int = 2026,
) -> np.ndarray:
    """Add deterministic additive Gaussian noise to a synthetic measurement."""
    clean = validate_mono(measurement, "measurement")
    if not np.isfinite(noise_gain) or noise_gain < 0:
        raise ValueError("noise_gain must be finite and non-negative")
    rng = np.random.default_rng(seed)
    noise = noise_gain * rng.normal(size=clean.size)
    result = clean + noise
    return np.asarray(result, dtype=np.float64)


def estimate_secondary_path(
    excitation: np.ndarray,
    measurement: np.ndarray,
    fir_length: int,
) -> np.ndarray:
    """Estimate h from controlled excitation and measured output via least squares.

    The measurement is y[n] = Xh + noise.  X is causal convolution matrix,
    and h = argmin ||Xh-y||^2.

    This only describes an offline estimate. It is not a validated hardware
    impulse response until controlled real measurements are performed.
    """
    excitation_values = validate_mono(excitation, "excitation")
    measurement_values = validate_mono(measurement, "measurement")
    if excitation_values.size != measurement_values.size:
        raise ValueError("excitation and measurement lengths must match")
    if not isinstance(fir_length, int) or not 1 <= fir_length <= 4096:
        raise ValueError("fir_length must be between 1 and 4096")
    if fir_length > excitation_values.size:
        raise ValueError("fir_length cannot exceed excitation length")

    matrix = np.zeros(
        (excitation_values.size, fir_length),
        dtype=np.float64,
    )
    for tap in range(fir_length):
        matrix[tap:, tap] = excitation_values[:excitation_values.size - tap]

    impulse, *_ = np.linalg.lstsq(
        matrix,
        measurement_values,
        rcond=None,
    )
    if not np.all(np.isfinite(impulse)):
        raise RuntimeError("secondary-path estimation produced non-finite coefficients")
    return np.asarray(impulse, dtype=np.float64)


def normalized_mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference_values = validate_mono(reference, "reference")
    estimate_values = validate_mono(estimate, "estimate")
    if reference_values.size != estimate_values.size:
        raise ValueError("reference and estimate lengths must match")
    denominator = float(
        np.dot(reference_values, reference_values)
    )
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0 if np.array_equal(reference_values, estimate_values) else float("inf")
    mse = float(
        np.mean(
            np.square(reference_values - estimate_values)
        )
    )
    return mse / (
        np.mean(np.square(reference_values))
    )


def estimate_peak_delay(impulse_response: np.ndarray) -> int:
    impulse = validate_mono(impulse_response, "impulse_response")
    return int(np.argmax(np.abs(impulse)))


def run_estimation_experiment(
    config: FIRSecondaryPathConfig,
    duration_seconds: float = 1.0,
) -> EstimationResult:
    """Generate synthetic excitation, model output and a least-squares estimate."""
    true_path = generate_synthetic_secondary_path(config)
    excitation = generate_controlled_excitation(
        config.sample_rate,
        duration_seconds,
        seed=config.seed,
    )
    clean_measurement = true_path.convolve(excitation)
    measurement = add_measurement_noise(
        clean_measurement,
        config.noise_gain,
        seed=config.seed + 1,
    )
    estimated = estimate_secondary_path(
        excitation,
        measurement,
        config.fir_length,
    )
    mse = normalized_mse(
        true_path.impulse_response,
        estimated,
    )
    true_peak = estimate_peak_delay(true_path.impulse_response)
    estimated_peak = estimate_peak_delay(estimated)

    return EstimationResult(
        impulse_response=estimated,
        true_impulse_response=true_path.impulse_response,
        measurement=measurement,
        excitation=excitation,
        normalized_mse=mse,
        impulse_peak_error=float(
            np.max(
                np.abs(
                    true_path.impulse_response
                    - estimated
                )
            )
        ),
        delay_error_samples=estimated_peak - true_peak,
    )


def run_experiment_suite(
    sample_rate: int = 48_000,
) -> dict[str, EstimationResult]:
    """Run A-E synthetic secondary-path comparison experiments."""
    return {
        "A_simulated_secondary_path": run_estimation_experiment(
            FIRSecondaryPathConfig(
                sample_rate=sample_rate,
                fir_length=128,
                delay_samples=16,
                scale=1.0,
                resonance_hz=1000.0,
                noise_gain=0.0,
            )
        ),
        "B_different_delay": run_estimation_experiment(
            FIRSecondaryPathConfig(
                sample_rate=sample_rate,
                fir_length=128,
                delay_samples=32,
                scale=1.0,
                resonance_hz=1000.0,
                noise_gain=0.0,
            )
        ),
        "C_different_frequency_response": run_estimation_experiment(
            FIRSecondaryPathConfig(
                sample_rate=sample_rate,
                fir_length=128,
                delay_samples=16,
                scale=1.0,
                resonance_hz=3500.0,
                resonance_q=4.0,
                noise_gain=0.0,
            )
        ),
        "D_measurement_noise": run_estimation_experiment(
            FIRSecondaryPathConfig(
                sample_rate=sample_rate,
                fir_length=128,
                delay_samples=16,
                scale=1.0,
                resonance_hz=1000.0,
                noise_gain=0.002,
            )
        ),
        "E_model_estimation_error": run_estimation_experiment(
            FIRSecondaryPathConfig(
                sample_rate=sample_rate,
                fir_length=128,
                delay_samples=16,
                scale=1.0,
                resonance_hz=None,
                noise_gain=0.01,
                seed=909,
            )
        ),
    }


def build_reports(
    sample_rate: int = 48_000,
) -> list[SecondaryPathReport]:
    reports: list[SecondaryPathReport] = []
    experiments = run_experiment_suite(sample_rate)
    for name, result in experiments.items():
        reports.append(
            SecondaryPathReport(
                name=name,
                sample_rate=sample_rate,
                fir_length=result.true_impulse_response.size,
                configured_delay_samples=estimate_peak_delay(
                    result.true_impulse_response
                ),
                estimated_delay_samples=estimate_peak_delay(
                    result.impulse_response
                ),
                normalized_mse=result.normalized_mse,
                impulse_peak_error=result.impulse_peak_error,
                delay_error_samples=result.delay_error_samples,
            )
        )
    return reports


def write_report(
    path: str | Path,
    reports: Iterable[SecondaryPathReport],
) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for report in reports:
        payload.append(
            {
                "name": report.name,
                "sample_rate": report.sample_rate,
                "fir_length": report.fir_length,
                "configured_delay_samples": report.configured_delay_samples,
                "estimated_delay_samples": report.estimated_delay_samples,
                "normalized_mse": report.normalized_mse,
                "impulse_peak_error": report.impulse_peak_error,
                "delay_error_samples": report.delay_error_samples,
                "measurement_type": "synthetic_offline",
            }
        )
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_visualization(
    sample_rate: int = 48_000,
    experiment_name: str = "A_simulated_secondary_path",
) -> EstimationResult:
    """Plot impulse response and frequency response for one synthetic case."""
    import matplotlib.pyplot as plt

    result = run_experiment_suite(sample_rate)[experiment_name]
    true_path = FIRSecondaryPath(
        result.true_impulse_response,
        sample_rate,
        source="synthetic",
    )
    estimated_path = FIRSecondaryPath(
        result.impulse_response,
        sample_rate,
        source="controlled_measurement",
    )
    true_frequency, true_magnitude, true_phase = true_path.frequency_response()
    estimated_frequency, estimated_magnitude, estimated_phase = (
        estimated_path.frequency_response()
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].stem(
        np.arange(true_path.fir_length),
        true_path.impulse_response,
        linefmt="tab:blue",
        markerfmt=".",
        basefmt=" ",
        label="Synthetic true model",
    )
    axes[0, 0].stem(
        np.arange(estimated_path.fir_length),
        estimated_path.impulse_response,
        linefmt="tab:red",
        markerfmt=".",
        basefmt=" ",
        label="Least-squares estimate",
    )
    axes[0, 0].set_title("Impulse response")
    axes[0, 0].set_xlabel("Tap")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.2)

    axes[0, 1].plot(
        true_frequency,
        true_magnitude,
        lw=0.8,
        label="Synthetic true model",
    )
    axes[0, 1].plot(
        estimated_frequency,
        estimated_magnitude,
        lw=0.8,
        label="Least-squares estimate",
    )
    axes[0, 1].set_title("Magnitude response")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Magnitude (dB)")
    axes[0, 1].set_xlim(0.0, sample_rate / 2.0)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.2)

    axes[1, 0].plot(true_frequency, true_phase, lw=0.8, label="Synthetic")
    axes[1, 0].plot(
        estimated_frequency,
        estimated_phase,
        lw=0.8,
        label="Estimated",
    )
    axes[1, 0].set_title("Phase response")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Phase (radians)")
    axes[1, 0].set_xlim(0.0, sample_rate / 2.0)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.2)

    axes[1, 1].text(
        0.02,
        0.98,
        (
            f"Experiment: {experiment_name}\n"
            f"Normalized MSE: {result.normalized_mse:.6f}\n"
            f"Peak error: {result.impulse_peak_error:.6f}\n"
            f"Delay error: {result.delay_error_samples:+d} samples\n"
            "\nThis is synthetic/offline modeling,\n"
            "not a measured earphone response."
        ),
        transform=axes[1, 1].transAxes,
        va="top",
        family="monospace",
    )
    axes[1, 1].set_axis_off()
    fig.suptitle(
        "Secondary FIR path: synthetic model and offline estimate",
        fontweight="bold",
    )
    plt.show()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline FIR secondary acoustic-path modeling laboratory"
    )
    parser.add_argument(
        "--experiment",
        choices=[
            "A_simulated_secondary_path",
            "B_different_delay",
            "C_different_frequency_response",
            "D_measurement_noise",
            "E_model_estimation_error",
        ],
        default="A_simulated_secondary_path",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=[44100, 48000],
        default=48000,
    )
    parser.add_argument("--json-report", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    result = run_visualization(
        args.sample_rate,
        args.experiment,
    )
    print("Offline synthetic secondary-path experiment")
    print(f"Experiment: {args.experiment}")
    print(f"Normalized MSE: {result.normalized_mse:.6f}")
    print(f"Impulse peak error: {result.impulse_peak_error:.6f}")
    print(f"Delay error: {result.delay_error_samples:+d} samples")

    if args.json_report:
        write_report(
            args.json_report,
            build_reports(args.sample_rate),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
