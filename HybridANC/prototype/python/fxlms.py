"""Filtered-X Least Mean Squares (FxLMS) simulation.

Architecture:
reference x[n] -> adaptive FIR w[n] -> anti-noise y[n]
                -> modeled secondary path S(z)
                -> cancellation location residual d[n]-S*y[n]
                <- simulated error e[n]

This is an offline, synthetic algorithm. It has no audio callbacks, microphone
roles, uncalibrated playback or safety interlock for physical headphones.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import logging
from pathlib import Path
import time
from typing import Iterable

import numpy as np

try:
    from .lms import LMSFilter, LMSConfig
    from .nlms import NLMSFilter, NLMSConfig
except ImportError:
    from lms import LMSFilter, LMSConfig
    from nlms import NLMSFilter, NLMSConfig


LOGGER = logging.getLogger("hybridanc.fxlms")


@dataclass(frozen=True)
class FxLMSConfig:
    filter_length: int = 64
    learning_rate: float = 0.001
    epsilon: float = 1e-12
    weight_limit: float = 100.0
    output_limit: float = 10.0
    coefficient_history_limit: int = 10_000
    convergence_window: int = 256
    convergence_threshold: float = 1e-12

    def __post_init__(self) -> None:
        if not isinstance(self.filter_length, int) or isinstance(
            self.filter_length,
            bool,
        ):
            raise TypeError("filter_length must be an integer")
        if not 1 <= self.filter_length <= 4096:
            raise ValueError("filter_length must be between 1 and 4096")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not np.isfinite(self.weight_limit) or self.weight_limit <= 0:
            raise ValueError("weight_limit must be finite and positive")
        if not np.isfinite(self.output_limit) or self.output_limit <= 0:
            raise ValueError("output_limit must be finite and positive")
        if not isinstance(
            self.coefficient_history_limit,
            int,
        ) or self.coefficient_history_limit < 0:
            raise TypeError("coefficient_history_limit must be non-negative")
        if not 1 <= self.convergence_window <= 100_000:
            raise ValueError("convergence_window is out of range")
        if not np.isfinite(self.convergence_threshold) or (
            self.convergence_threshold < 0
        ):
            raise ValueError("convergence_threshold must be finite and non-negative")


@dataclass(frozen=True)
class FxLMSResult:
    antinoise: np.ndarray
    residual_error: np.ndarray
    weights: np.ndarray
    coefficient_history: np.ndarray
    residual_rms_history: np.ndarray
    converged: bool
    divergence_detected: bool
    residual_rms: float
    noise_reduction_db: float
    convergence_time_samples: int | None
    mean_processing_time_ms: float


@dataclass(frozen=True)
class FxLMSComparisonReport:
    name: str
    method: str
    residual_rms: float
    noise_reduction_db: float
    convergence_time_samples: int | None
    divergence_detected: bool
    mean_processing_time_ms: float


def validate_mono(signal: object, name: str = "signal") -> np.ndarray:
    values = np.asarray(signal)
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if values.ndim == 2:
        if values.shape[1] not in (1, 2):
            raise ValueError(f"{name} must be mono or two-channel")
        values = np.mean(values, axis=1)
    elif values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional or mono/stereo")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"{name} must contain floating-point samples")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(values, dtype=np.float64)


def validate_secondary_path(
    secondary_impulse_response: np.ndarray,
    name: str = "secondary_impulse_response",
) -> np.ndarray:
    impulse = validate_mono(
        secondary_impulse_response,
        name,
    )
    if impulse.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.any(np.abs(impulse) > 0):
        raise ValueError(f"{name} must contain at least one non-zero coefficient")
    return impulse


def rms(signal: np.ndarray) -> float:
    values = validate_mono(signal, "signal")
    return float(
        np.sqrt(np.mean(np.square(values, dtype=np.float64)))
    )


def noise_reduction_db(original: np.ndarray, residual: np.ndarray) -> float:
    before = rms(original)
    after = rms(residual)
    return float(
        20.0 * np.log10(
            max(after, np.finfo(np.float64).tiny)
            / max(before, np.finfo(np.float64).tiny)
        )
    )


def convolve_causal(
    impulse_response: np.ndarray,
    signal: np.ndarray,
) -> np.ndarray:
    """Apply y[n] = sum h[k]x[n-k] with causal history."""
    impulse = validate_secondary_path(impulse_response)
    values = validate_mono(signal, "signal")
    output = np.zeros(values.size + impulse.size - 1, dtype=np.float64)
    for tap, coefficient in enumerate(impulse):
        if coefficient:
            output[tap:tap + values.size] += coefficient * values
    return output[:values.size]


class FxLMSFilter:
    """Independent offline Filtered-X LMS FIR adaptive controller."""

    def __init__(
        self,
        config: FxLMSConfig,
        estimated_secondary_impulse_response: np.ndarray,
        true_secondary_impulse_response: np.ndarray | None = None,
        initial_weights: np.ndarray | None = None,
    ) -> None:
        self.config = config
        impulse = validate_secondary_path(
            estimated_secondary_impulse_response,
            "estimated_secondary_impulse_response",
        )
        self.estimated_secondary_impulse_response = impulse
        if true_secondary_impulse_response is None:
            true_impulse = impulse.copy()
        else:
            true_impulse = validate_secondary_path(
                true_secondary_impulse_response,
                "true_secondary_impulse_response",
            )
        self.true_secondary_impulse_response = true_impulse

        if initial_weights is None:
            weights = np.zeros(config.filter_length, dtype=np.float64)
        else:
            weights = validate_mono(initial_weights, "initial_weights")
            if weights.size != config.filter_length:
                raise ValueError(
                    "initial_weights length must equal filter_length"
                )
        self.weights = weights.copy()
        self._reference_buffer = np.zeros(
            config.filter_length,
            dtype=np.float64,
        )
        self._filtered_reference_buffer = np.zeros(
            config.filter_length,
            dtype=np.float64,
        )
        self._reference_history = np.empty(0, dtype=np.float64)
        self._anti_noise_history = np.empty(0, dtype=np.float64)
        self._coefficient_history: list[np.ndarray] = []
        self._residual_history: list[float] = []
        self._residual_rms_history: list[float] = []
        self._weight_norm_history: list[float] = []
        self._processing_times_ms: list[float] = []
        self._converged = False
        self._divergence_detected = False
        self._convergence_time_samples: int | None = None

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def divergence_detected(self) -> bool:
        return self._divergence_detected

    @property
    def coefficient_history(self) -> np.ndarray:
        if not self._coefficient_history:
            return np.empty(
                (0, self.config.filter_length),
                dtype=np.float64,
            )
        return np.asarray(self._coefficient_history)

    @property
    def residual_history(self) -> np.ndarray:
        return np.asarray(self._residual_history, dtype=np.float64)

    def reset(self) -> None:
        self.weights[:] = 0.0
        self._reference_buffer[:] = 0.0
        self._filtered_reference_buffer[:] = 0.0
        self._reference_history = np.empty(0, dtype=np.float64)
        self._anti_noise_history = np.empty(0, dtype=np.float64)
        self._coefficient_history = []
        self._residual_history = []
        self._residual_rms_history = []
        self._weight_norm_history = []
        self._processing_times_ms = []
        self._converged = False
        self._divergence_detected = False
        self._convergence_time_samples = None

    def update(
        self,
        reference_sample: float,
        primary_residual_sample: float,
    ) -> tuple[float, float]:
        """Process x[n], generate y[n], and update w[n+1].

        primary_residual_sample is d[n], the noise reaching the error location
        before anti-noise, represented as an offline desired residual signal.
        """
        start_ns = time.monotonic_ns()
        if not np.isfinite(reference_sample) or not np.isfinite(
            primary_residual_sample
        ):
            raise ValueError("reference and primary residual samples must be finite")

        length = self.config.filter_length

        self._reference_buffer[1:] = self._reference_buffer[:-1]
        self._reference_buffer[0] = reference_sample

        controller_output = float(
            np.dot(self.weights, self._reference_buffer)
        )
        if abs(controller_output) > self.config.output_limit:
            self._divergence_detected = True
            controller_output = float(
                np.clip(
                    controller_output,
                    -self.config.output_limit,
                    self.config.output_limit,
                )
            )

        previous_reference = np.concatenate(
            (
                np.asarray([reference_sample]),
                self._reference_history,
            )
        )
        filtered_reference = 0.0
        for tap, coefficient in enumerate(
            self.estimated_secondary_impulse_response
        ):
            if tap >= previous_reference.size:
                break
            filtered_reference += coefficient * previous_reference[tap]

        self._filtered_reference_buffer[1:] = (
            self._filtered_reference_buffer[:-1]
        )
        self._filtered_reference_buffer[0] = filtered_reference

        anti_noise_secondary = 0.0
        if hasattr(self, "_anti_noise_history"):
            history = self._anti_noise_history
        else:
            history = np.empty(0, dtype=np.float64)
        anti_noise_history = np.concatenate(
            (
                np.asarray([controller_output]),
                history,
            )
        )
        for tap, coefficient in enumerate(
            self.true_secondary_impulse_response
        ):
            if tap >= anti_noise_history.size:
                break
            anti_noise_secondary += (
                coefficient * anti_noise_history[tap]
            )
        self._anti_noise_history = anti_noise_history[
            :max(
                self.true_secondary_impulse_response.size,
                length,
            )
        ]

        residual_error = float(
            primary_residual_sample - anti_noise_secondary
        )
        reference_energy = float(
            np.dot(
                self._filtered_reference_buffer,
                self._filtered_reference_buffer,
            )
        )
        if reference_energy <= self.config.epsilon:
            update = np.zeros(length, dtype=np.float64)
        else:
            update = (
                self.config.learning_rate
                * residual_error
                * self._filtered_reference_buffer
                / (self.config.epsilon + reference_energy)
            )

        updated_weights = self.weights + update
        if not np.all(np.isfinite(updated_weights)):
            self._divergence_detected = True
            raise RuntimeError("FxLMS coefficient update produced non-finite weights")
        if float(np.max(np.abs(updated_weights))) > self.config.weight_limit:
            self._divergence_detected = True
            updated_weights = np.clip(
                updated_weights,
                -self.config.weight_limit,
                self.config.weight_limit,
            )
        self.weights = updated_weights

        self._residual_history.append(residual_error)
        window = self.config.convergence_window
        recent = np.asarray(self._residual_history[-window:])
        residual_rms = float(
            np.sqrt(np.mean(recent * recent))
        )
        self._residual_rms_history.append(residual_rms)

        weight_norm = float(np.linalg.norm(self.weights))
        self._weight_norm_history.append(weight_norm)
        if len(self._coefficient_history) < self.config.coefficient_history_limit:
            self._coefficient_history.append(self.weights.copy())

        if len(recent) >= window:
            if residual_rms <= self.config.convergence_threshold:
                self._converged = True
                if self._convergence_time_samples is None:
                    self._convergence_time_samples = len(
                        self._residual_history
                    )

        self._reference_history = np.concatenate(
            (
                np.asarray([reference_sample]),
                self._reference_history,
            )
        )
        if self._reference_history.size > (
            self.estimated_secondary_impulse_response.size + length
        ):
            self._reference_history = self._reference_history[
                :self.estimated_secondary_impulse_response.size + length
            ]

        elapsed_ms = (
            time.monotonic_ns() - start_ns
        ) / 1_000_000.0
        self._processing_times_ms.append(elapsed_ms)
        return controller_output, residual_error

    def process(
        self,
        reference: np.ndarray,
        primary_residual: np.ndarray,
    ) -> FxLMSResult:
        """Run FxLMS on matching offline reference/error arrays."""
        reference_values = validate_mono(reference, "reference")
        residual_values = validate_mono(
            primary_residual,
            "primary_residual",
        )
        if reference_values.size != residual_values.size:
            raise ValueError("reference and primary residual lengths must match")

        antinoise = np.empty(reference_values.size, dtype=np.float64)
        residual_error = np.empty(reference_values.size, dtype=np.float64)
        for index, (reference_sample, residual_sample) in enumerate(
            zip(reference_values, residual_values)
        ):
            antinoise[index], residual_error[index] = self.update(
                reference_sample,
                residual_sample,
            )

        residual_rms = rms(residual_error)
        original_rms = rms(residual_values)
        reduction_db = 20.0 * np.log10(
            max(residual_rms, np.finfo(np.float64).tiny)
            / max(original_rms, np.finfo(np.float64).tiny)
        )
        mean_processing = (
            float(np.mean(self._processing_times_ms))
            if self._processing_times_ms
            else 0.0
        )
        return FxLMSResult(
            antinoise=antinoise,
            residual_error=residual_error,
            weights=self.weights.copy(),
            coefficient_history=self.coefficient_history,
            residual_rms_history=np.asarray(
                self._residual_rms_history,
                dtype=np.float64,
            ),
            converged=self._converged,
            divergence_detected=self._divergence_detected,
            residual_rms=residual_rms,
            noise_reduction_db=reduction_db,
            convergence_time_samples=self._convergence_time_samples,
            mean_processing_time_ms=mean_processing,
        )


def generate_stationary_sine(
    sample_rate: int = 48_000,
    duration_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = int(sample_rate * duration_seconds)
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    reference = 0.50 * np.sin(2.0 * np.pi * 100.0 * time)
    primary_residual = reference.copy()
    return reference, primary_residual


def generate_multi_frequency(
    sample_rate: int = 48_000,
    duration_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = int(sample_rate * duration_seconds)
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    reference = (
        0.40 * np.sin(2.0 * np.pi * 100.0 * time)
        + 0.20 * np.sin(2.0 * np.pi * 350.0 * time)
        + 0.10 * np.sin(2.0 * np.pi * 900.0 * time)
    )
    primary_residual = reference.copy()
    return reference, primary_residual


def generate_changing_amplitude(
    sample_rate: int = 48_000,
    duration_seconds: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = int(sample_rate * duration_seconds)
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    envelope = np.linspace(0.05, 1.0, frame_count)
    reference = envelope * np.sin(
        2.0 * np.pi * 100.0 * time
    )
    primary_residual = reference.copy()
    return reference, primary_residual


def generate_changing_reference(
    sample_rate: int = 48_000,
    duration_seconds: float = 2.0,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = int(sample_rate * duration_seconds)
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(seed)
    frequency = 150.0 + 100.0 * np.sin(
        2.0 * np.pi * 0.25 * time
    )
    phase = 2.0 * np.pi * np.cumsum(frequency) / sample_rate
    reference = 0.35 * np.sin(phase) + 0.05 * rng.normal(
        size=frame_count
    )
    primary_residual = reference.copy()
    return reference, primary_residual


def fixed_phase_inversion(reference: np.ndarray) -> np.ndarray:
    """Benchmark fixed exact inversion on the synthetic reference."""
    values = validate_mono(reference, "reference")
    return -values


def compare_methods(
    reference: np.ndarray,
    primary_residual: np.ndarray,
    estimated_secondary_impulse_response: np.ndarray,
    true_secondary_impulse_response: np.ndarray | None = None,
    fxlms_config: FxLMSConfig | None = None,
) -> dict[str, FxLMSComparisonReport]:
    """Compare no cancellation, fixed inversion, LMS, NLMS and FxLMS."""
    reference_values = validate_mono(reference, "reference")
    primary_values = validate_mono(primary_residual, "primary_residual")
    if reference_values.size != primary_values.size:
        raise ValueError("reference and primary residual lengths must match")

    none_rms = rms(primary_values)
    reports: dict[str, FxLMSComparisonReport] = {}

    no_residual = primary_values.copy()
    reports["no_cancellation"] = FxLMSComparisonReport(
        name="no_cancellation",
        method="No cancellation",
        residual_rms=rms(no_residual),
        noise_reduction_db=noise_reduction_db(primary_values, no_residual),
        convergence_time_samples=None,
        divergence_detected=False,
        mean_processing_time_ms=0.0,
    )

    fixed_residual = primary_values + fixed_phase_inversion(reference_values)
    reports["fixed_phase_inversion"] = FxLMSComparisonReport(
        name="fixed_phase_inversion",
        method="Fixed phase inversion",
        residual_rms=rms(fixed_residual),
        noise_reduction_db=noise_reduction_db(primary_values, fixed_residual),
        convergence_time_samples=None,
        divergence_detected=False,
        mean_processing_time_ms=0.0,
    )

    fxlms_config = fxlms_config or FxLMSConfig()

    from .lms import LMSConfig, LMSFilter
    from .nlms import NLMSConfig, NLMSFilter
    lms_result = LMSFilter(
        LMSConfig(
            filter_length=fxlms_config.filter_length,
            learning_rate=fxlms_config.learning_rate,
        )
    ).process(reference_values, primary_values)
    reports["lms"] = FxLMSComparisonReport(
        name="lms",
        method="LMS",
        residual_rms=rms(lms_result.error),
        noise_reduction_db=noise_reduction_db(primary_values, lms_result.error),
        convergence_time_samples=None,
        divergence_detected=lms_result.divergence_detected,
        mean_processing_time_ms=0.0,
    )

    nlms_result = NLMSFilter(
        NLMSConfig(
            filter_length=fxlms_config.filter_length,
            step_size=0.5,
        )
    ).process(reference_values, primary_values)
    reports["nlms"] = FxLMSComparisonReport(
        name="nlms",
        method="NLMS",
        residual_rms=rms(nlms_result.error),
        noise_reduction_db=noise_reduction_db(primary_values, nlms_result.error),
        convergence_time_samples=None,
        divergence_detected=nlms_result.divergence_detected,
        mean_processing_time_ms=0.0,
    )

    fxlms_result = FxLMSFilter(
        fxlms_config,
        estimated_secondary_impulse_response,
        true_secondary_impulse_response,
    ).process(reference_values, primary_values)
    reports["fxlms"] = FxLMSComparisonReport(
        name="fxlms",
        method="FxLMS",
        residual_rms=fxlms_result.residual_rms,
        noise_reduction_db=fxlms_result.noise_reduction_db,
        convergence_time_samples=fxlms_result.convergence_time_samples,
        divergence_detected=fxlms_result.divergence_detected,
        mean_processing_time_ms=fxlms_result.mean_processing_time_ms,
    )
    return reports


def secondary_path_with_delay(
    sample_rate: int,
    delay_samples: int,
    fir_length: int = 64,
) -> np.ndarray:
    if sample_rate not in (44100, 48000):
        raise ValueError("sample_rate must be 44100 or 48000 Hz")
    if not 0 <= delay_samples < fir_length:
        raise ValueError("delay_samples must be within fir_length")
    impulse = np.zeros(fir_length, dtype=np.float64)
    impulse[delay_samples] = 1.0
    return impulse


def write_report(
    path: str | Path,
    reports: dict[str, FxLMSComparisonReport],
) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": report.name,
            "method": report.method,
            "residual_rms": report.residual_rms,
            "noise_reduction_db": report.noise_reduction_db,
            "convergence_time_samples": report.convergence_time_samples,
            "divergence_detected": report.divergence_detected,
            "mean_processing_time_ms": report.mean_processing_time_ms,
            "measurement_type": "offline_simulation",
        }
        for report in reports.values()
    ]
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_visualization(
    reference: np.ndarray,
    primary_residual: np.ndarray,
    estimated_secondary_impulse_response: np.ndarray,
    true_secondary_impulse_response: np.ndarray,
) -> FxLMSResult:
    import matplotlib.pyplot as plt

    result = FxLMSFilter(
        FxLMSConfig(filter_length=64, learning_rate=0.001),
        estimated_secondary_impulse_response,
        true_secondary_impulse_response,
    ).process(reference, primary_residual)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(primary_residual, lw=0.5, label="Primary residual d[n]")
    axes[0, 0].plot(result.residual_error, lw=0.5, label="FxLMS error e[n]")
    axes[0, 0].set_title("Primary residual and FxLMS residual")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.2)

    axes[0, 1].plot(result.residual_rms_history, lw=0.8)
    axes[0, 1].set_title("Residual RMS history")
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].set_ylabel("Residual RMS")
    axes[0, 1].grid(True, alpha=0.2)

    if result.coefficient_history.size:
        axes[1, 0].imshow(
            result.coefficient_history.T,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
        )
    axes[1, 0].set_title("Controller coefficient history")
    axes[1, 0].set_xlabel("Update")
    axes[1, 0].set_ylabel("Coefficient index")

    axes[1, 1].text(
        0.02,
        0.98,
        (
            f"Residual RMS: {result.residual_rms:.6f}\n"
            f"Noise reduction: {result.noise_reduction_db:.2f} dB\n"
            f"Convergence sample: "
            f"{result.convergence_time_samples if result.convergence_time_samples is not None else 'not reached'}\n"
            f"Divergence: {result.divergence_detected}\n"
            f"Mean processing: {result.mean_processing_time_ms:.4f} ms\n"
            "\nOffline simulation only.\n"
            "No anti-noise is played."
        ),
        transform=axes[1, 1].transAxes,
        va="top",
        family="monospace",
    )
    axes[1, 1].set_axis_off()
    fig.suptitle(
        "Filtered-X LMS offline simulation",
        fontweight="bold",
    )
    import matplotlib.pyplot as plt
    plt.show()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run offline Filtered-X LMS ANC simulation"
    )
    parser.add_argument(
        "--experiment",
        choices=[
            "stationary_100hz",
            "multi_frequency",
            "changing_amplitude",
            "changing_reference",
        ],
        default="stationary_100hz",
    )
    parser.add_argument("--filter-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--secondary-delay", type=int, default=16)
    parser.add_argument("--json-report", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    generators = {
        "stationary_100hz": generate_stationary_sine,
        "multi_frequency": generate_multi_frequency,
        "changing_amplitude": generate_changing_amplitude,
        "changing_reference": generate_changing_reference,
    }
    reference, primary_residual = generators[args.experiment]()
    estimated_secondary = secondary_path_with_delay(
        48000,
        args.secondary_delay,
        args.filter_length,
    )
    true_secondary = secondary_path_with_delay(
        48000,
        args.secondary_delay,
        args.filter_length,
    )

    try:
        result = run_visualization(
            reference,
            primary_residual,
            estimated_secondary,
            true_secondary,
        )
    except Exception as exc:
        LOGGER.error("FxLMS simulation failed: %s", exc)
        return 1

    print("Offline FxLMS simulation; no anti-noise output is active.")
    print(f"Residual RMS: {result.residual_rms:.8f}")
    print(f"Noise reduction: {result.noise_reduction_db:.3f} dB")
    print(
        "Convergence sample:",
        result.convergence_time_samples
        if result.convergence_time_samples is not None
        else "not reached",
    )

    if args.json_report:
        write_report(
            args.json_report,
            compare_methods(
                reference,
                primary_residual,
                estimated_secondary,
                true_secondary,
                FxLMSConfig(
                    filter_length=args.filter_length,
                    learning_rate=args.learning_rate,
                ),
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
