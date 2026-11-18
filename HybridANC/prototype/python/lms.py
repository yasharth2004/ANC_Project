"""Least Mean Squares (LMS) FIR adaptive filtering for offline experiments.

This is a basic simulation tool, not a physical headphone ANC implementation.
It does not open an output device or play anti-noise.

Definitions:
* x[n]: newest-to-oldest reference vector for filter length N.
* d[n]: desired signal at sample n.
* w[n]: FIR coefficient vector.
* y[n]: w[n]^T x[n], filter output.
* e[n]: d[n] - y[n], residual error.
* mu: learning-rate step size.
* w[n+1]: w[n] + mu * e[n] * x[n].

The learning rate controls convergence. Too small is slow; too large can
overshoot, amplify coefficient drift and become numerically unstable.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np


LOGGER = logging.getLogger("hybridanc.lms")


@dataclass(frozen=True)
class LMSConfig:
    filter_length: int = 64
    learning_rate: float = 0.01
    weight_limit: float = 10_000.0
    output_limit: float = 10_000.0
    record_coefficient_history: bool = True
    coefficient_history_limit: int = 10_000

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
        if not np.isfinite(self.weight_limit) or self.weight_limit <= 0:
            raise ValueError("weight_limit must be finite and positive")
        if not np.isfinite(self.output_limit) or self.output_limit <= 0:
            raise ValueError("output_limit must be finite and positive")
        if not isinstance(
            self.coefficient_history_limit,
            int,
        ) or self.coefficient_history_limit < 0:
            raise TypeError("coefficient_history_limit must be non-negative")


@dataclass(frozen=True)
class LMSResult:
    output: np.ndarray
    error: np.ndarray
    weights: np.ndarray
    coefficient_history: np.ndarray
    converged: bool
    divergence_detected: bool
    mean_error_power: float
    final_error_power: float
    mean_weight_norm: float


@dataclass(frozen=True)
class LMSExperimentReport:
    name: str
    reference_type: str
    filter_length: int
    learning_rate: float
    mean_adaptive_error_power: float
    mean_fixed_error_power: float
    adaptive_rms_error: float
    fixed_rms_error: float
    converged: bool
    divergence_detected: bool


def validate_mono(signal: object, name: str) -> np.ndarray:
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


class LMSFilter:
    """Stateful FIR LMS adaptive filter."""

    def __init__(
        self,
        config: LMSConfig | None = None,
        initial_weights: np.ndarray | None = None,
    ) -> None:
        self.config = config or LMSConfig()
        length = self.config.filter_length
        if initial_weights is None:
            weights = np.zeros(length, dtype=np.float64)
        else:
            weights = validate_mono(initial_weights, "initial_weights")
            if weights.size != length:
                raise ValueError(
                    "initial_weights length must equal filter_length"
                )
        self.weights = weights.copy()
        self._reference_buffer = np.zeros(length, dtype=np.float64)
        self._coefficient_history: list[np.ndarray] = []
        self._error_history: list[float] = []
        self._weight_norm_history: list[float] = []
        self._converged = False
        self._divergence_detected = False

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def divergence_detected(self) -> bool:
        return self._divergence_detected

    @property
    def coefficient_history(self) -> np.ndarray:
        if not self._coefficient_history:
            return np.empty((0, self.config.filter_length), dtype=np.float64)
        return np.asarray(self._coefficient_history)

    @property
    def error_history(self) -> np.ndarray:
        return np.asarray(self._error_history, dtype=np.float64)

    def reset(self) -> None:
        self.weights[:] = 0.0
        self._reference_buffer[:] = 0.0
        self._coefficient_history = []
        self._error_history = []
        self._weight_norm_history = []
        self._converged = False
        self._divergence_detected = False

    def update(self, reference_sample: float, desired_sample: float) -> tuple[float, float]:
        """Process one sample using y[n], e[n], and w[n+1] LMS update."""
        if not np.isfinite(reference_sample) or not np.isfinite(desired_sample):
            raise ValueError("reference and desired samples must be finite")

        length = self.config.filter_length
        self._reference_buffer[1:] = self._reference_buffer[:-1]
        self._reference_buffer[0] = reference_sample

        output = float(np.dot(self.weights, self._reference_buffer))
        if abs(output) > self.config.output_limit:
            self._divergence_detected = True
            output = float(
                np.clip(
                    output,
                    -self.config.output_limit,
                    self.config.output_limit,
                )
            )
        error = float(desired_sample - output)

        step = self.config.learning_rate * error
        updated_weights = self.weights + step * self._reference_buffer
        if not np.all(np.isfinite(updated_weights)):
            self._divergence_detected = True
            raise RuntimeError("LMS coefficient update produced non-finite weights")

        weight_peak = float(np.max(np.abs(updated_weights)))
        if weight_peak > self.config.weight_limit:
            self._divergence_detected = True
            updated_weights = np.clip(
                updated_weights,
                -self.config.weight_limit,
                self.config.weight_limit,
            )
        self.weights = updated_weights

        self._error_history.append(error)
        weight_norm = float(np.linalg.norm(self.weights))
        self._weight_norm_history.append(weight_norm)
        if self.config.record_coefficient_history:
            limit = self.config.coefficient_history_limit
            if limit > 0:
                self._coefficient_history.append(self.weights.copy())
                if len(self._coefficient_history) > limit:
                    del self._coefficient_history[:len(self._coefficient_history) - limit]

        if len(self._error_history) >= 256:
            recent = np.asarray(self._error_history[-256:])
            recent_power = float(np.mean(recent * recent))
            if recent_power < 1e-12:
                self._converged = True
        return output, error

    def process(
        self,
        reference: np.ndarray,
        desired: np.ndarray,
    ) -> LMSResult:
        reference_values = validate_mono(reference, "reference")
        desired_values = validate_mono(desired, "desired")
        if reference_values.size != desired_values.size:
            raise ValueError("reference and desired lengths must match")

        output = np.empty(reference_values.size, dtype=np.float64)
        error = np.empty(reference_values.size, dtype=np.float64)
        for index, (reference_sample, desired_sample) in enumerate(
            zip(reference_values, desired_values)
        ):
            output[index], error[index] = self.update(
                reference_sample,
                desired_sample,
            )

        error_power = error * error
        return LMSResult(
            output=output,
            error=error,
            weights=self.weights.copy(),
            coefficient_history=self.coefficient_history,
            converged=self._converged,
            divergence_detected=self._divergence_detected,
            mean_error_power=float(np.mean(error_power)),
            final_error_power=float(np.mean(error_power[-256:])),
            mean_weight_norm=float(np.mean(self._weight_norm_history)),
        )


def fixed_fir_filter(
    reference: np.ndarray,
    desired: np.ndarray,
    filter_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fixed filter fit for offline benchmark comparison."""
    reference_values = validate_mono(reference, "reference")
    desired_values = validate_mono(desired, "desired")
    if reference_values.size != desired_values.size:
        raise ValueError("reference and desired lengths must match")
    if not isinstance(filter_length, int) or not 1 <= filter_length <= 4096:
        raise ValueError("filter_length must be between 1 and 4096")

    matrix = np.zeros(
        (reference_values.size, filter_length),
        dtype=np.float64,
    )
    for index in range(filter_length):
        matrix[index:, index] = reference_values[:reference_values.size - index]
    weights, *_ = np.linalg.lstsq(matrix, desired_values, rcond=None)
    output = matrix @ weights
    error = desired_values - output
    return error, weights


def rms_error(error: np.ndarray) -> float:
    values = np.asarray(error, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("error must not be empty")
    return float(np.sqrt(np.mean(values * values)))


def generate_constant_sine(
    sample_rate: int = 48_000,
    duration_seconds: float = 1.0,
    frequency_hz: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = int(sample_rate * duration_seconds)
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    reference = np.sin(2.0 * np.pi * frequency_hz * time)
    desired = 0.8 * np.sin(
        2.0 * np.pi * frequency_hz * time - np.deg2rad(20.0)
    )
    return reference, desired


def generate_changing_amplitude(
    sample_rate: int = 48_000,
    duration_seconds: float = 2.0,
    frequency_hz: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = int(sample_rate * duration_seconds)
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    reference = np.sin(2.0 * np.pi * frequency_hz * time)
    amplitude = 0.4 + 0.4 * np.linspace(0.0, 1.0, frame_count)
    desired = amplitude * np.sin(
        2.0 * np.pi * frequency_hz * time - np.deg2rad(25.0)
    )
    return reference, desired


def generate_multi_frequency(
    sample_rate: int = 48_000,
    duration_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = int(sample_rate * duration_seconds)
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    reference = (
        0.7 * np.sin(2.0 * np.pi * 100 * time)
        + 0.3 * np.sin(2.0 * np.pi * 350 * time)
    )
    desired = (
        0.5 * np.sin(2.0 * np.pi * 100 * time - np.deg2rad(15))
        + 0.22 * np.sin(2.0 * np.pi * 350 * time - np.deg2rad(35))
    )
    return reference, desired


def generate_white_noise_reference(
    sample_rate: int = 48_000,
    duration_seconds: float = 1.0,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    del sample_rate
    frame_count = int(48_000 * duration_seconds)
    rng = np.random.default_rng(seed)
    reference = rng.normal(0.0, 0.25, frame_count)
    weights = np.zeros(32, dtype=np.float64)
    weights[0] = 0.6
    weights[7] = -0.25
    weights[15] = 0.12
    desired = np.zeros(frame_count, dtype=np.float64)
    for index in range(32, frame_count):
        desired[index] = np.dot(
            weights,
            reference[index - 32:index][::-1],
        )
    return reference, desired


def run_lms_experiments(
    filter_length: int = 64,
    learning_rate: float = 0.01,
) -> list[LMSExperimentReport]:
    experiments = {
        "constant_frequency": (
            "constant sine",
            generate_constant_sine,
        ),
        "changing_amplitude": (
            "changing-amplitude sine",
            generate_changing_amplitude,
        ),
        "multi_frequency": (
            "multiple sine frequencies",
            generate_multi_frequency,
        ),
        "white_noise_reference": (
            "seeded white noise",
            generate_white_noise_reference,
        ),
    }
    reports: list[LMSExperimentReport] = []
    for name, (reference_type, generator) in experiments.items():
        reference, desired = generator()
        adaptive = LMSFilter(
            LMSConfig(
                filter_length=filter_length,
                learning_rate=learning_rate,
            )
        ).process(reference, desired)
        fixed_error, _ = fixed_fir_filter(
            reference,
            desired,
            filter_length,
        )
        reports.append(
            LMSExperimentReport(
                name=name,
                reference_type=reference_type,
                filter_length=filter_length,
                learning_rate=learning_rate,
                mean_adaptive_error_power=adaptive.mean_error_power,
                mean_fixed_error_power=float(
                    np.mean(fixed_error * fixed_error)
                ),
                adaptive_rms_error=rms_error(adaptive.error),
                fixed_rms_error=rms_error(fixed_error),
                converged=adaptive.converged,
                divergence_detected=adaptive.divergence_detected,
            )
        )
    return reports


def write_report(path: str | Path, reports: Iterable[LMSExperimentReport]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for report in reports:
        payload.append(
            {
                "name": report.name,
                "reference_type": report.reference_type,
                "filter_length": report.filter_length,
                "learning_rate": report.learning_rate,
                "mean_adaptive_error_power": report.mean_adaptive_error_power,
                "mean_fixed_error_power": report.mean_fixed_error_power,
                "adaptive_rms_error": report.adaptive_rms_error,
                "fixed_rms_error": report.fixed_rms_error,
                "converged": report.converged,
                "divergence_detected": report.divergence_detected,
                "measurement_type": "offline_adaptive_benchmark",
            }
        )
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_visualization(
    reference: np.ndarray,
    desired: np.ndarray,
    config: LMSConfig,
) -> LMSResult:
    """Plot adaptive error and coefficient norm for a single offline experiment."""
    import matplotlib.pyplot as plt

    result = LMSFilter(config).process(reference, desired)
    time = np.arange(desired.size)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(time, desired, lw=0.5, label="Desired d[n]", alpha=0.8)
    axes[0].plot(time, result.output, lw=0.5, label="LMS output y[n]", alpha=0.8)
    axes[0].set_title("Desired signal and LMS filter output")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(time, result.error, lw=0.5, color="tab:red")
    axes[1].set_title(
        f"LMS residual error e[n] | RMS {rms_error(result.error):.5f}"
    )
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Error")
    axes[1].grid(True, alpha=0.2)
    fig.suptitle(
        "Offline LMS simulation; no physical anti-noise is generated",
        fontweight="bold",
    )
    plt.show()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run offline LMS FIR adaptive-filter experiments"
    )
    parser.add_argument(
        "--experiment",
        choices=[
            "constant_frequency",
            "changing_amplitude",
            "multi_frequency",
            "white_noise_reference",
        ],
        default="constant_frequency",
    )
    parser.add_argument("--filter-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--json-report", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    try:
        config = LMSConfig(
            filter_length=args.filter_length,
            learning_rate=args.learning_rate,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    generators = {
        "constant_frequency": generate_constant_sine,
        "changing_amplitude": generate_changing_amplitude,
        "multi_frequency": generate_multi_frequency,
        "white_noise_reference": generate_white_noise_reference,
    }
    reference, desired = generators[args.experiment]()
    result = run_visualization(reference, desired, config)

    print("Offline LMS result")
    print(f"Mean error power: {result.mean_error_power:.8f}")
    print(f"Final error RMS: {rms_error(result.error[-256:]):.8f}")
    print(f"Converged: {result.converged}")
    print(f"Divergence detected: {result.divergence_detected}")

    fixed_error, _ = fixed_fir_filter(
        reference,
        desired,
        args.filter_length,
    )
    print(
        "Fixed-filter RMS error:",
        f"{rms_error(fixed_error):.8f}",
    )

    if args.json_report:
        write_report(
            args.json_report,
            run_lms_experiments(args.filter_length, args.learning_rate),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
