"""Normalized Least Mean Squares (NLMS) FIR adaptive filtering.

NLMS normalizes the LMS coefficient update by the current reference-vector
energy. It is a separate implementation and does not use LMS internals.

Variables:
* x[n]: newest-to-oldest FIR reference vector.
* d[n]: desired signal.
* w[n]: current filter coefficients.
* y[n] = w[n]^T x[n]: filter output.
* e[n] = d[n] - y[n]: residual error.
* P[n] = x[n]^T x[n]: reference energy.
* epsilon: positive stabilization floor for silent or low-energy input.
* mu: normalized step size, normally in (0, 1].
* w[n+1] = w[n] + mu * e[n] * x[n] / (epsilon + P[n]).

Without normalization, LMS update magnitude scales with x[n]^T x[n]. NLMS
partially removes that amplitude sensitivity, improving adaptation when input
power changes. It still can diverge with a large mu or rapidly changing desired
path.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .lms import LMSConfig, LMSFilter
except ImportError:
    from lms import LMSConfig, LMSFilter


LOGGER = logging.getLogger("hybridanc.nlms")


@dataclass(frozen=True)
class NLMSConfig:
    filter_length: int = 64
    step_size: float = 0.5
    epsilon: float = 1e-6
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
        if not np.isfinite(self.step_size) or not 0.0 < self.step_size <= 1.0:
            raise ValueError("step_size must be finite and in (0, 1]")
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


@dataclass(frozen=True)
class NLMSResult:
    output: np.ndarray
    error: np.ndarray
    weights: np.ndarray
    coefficient_history: np.ndarray
    residual_rms_history: np.ndarray
    converged: bool
    divergence_detected: bool
    mean_error_power: float
    final_error_power: float
    mean_weight_norm: float


@dataclass(frozen=True)
class ComparisonReport:
    name: str
    filter_length: int
    lms_step_size: float
    nlms_step_size: float
    lms_mean_error_power: float
    nlms_mean_error_power: float
    lms_final_rms_error: float
    nlms_final_rms_error: float
    lms_converged: bool
    nlms_converged: bool
    lms_divergence_detected: bool
    nlms_divergence_detected: bool


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


class NLMSFilter:
    """Stateful normalized FIR adaptive filter."""

    def __init__(
        self,
        config: NLMSConfig | None = None,
        initial_weights: np.ndarray | None = None,
    ) -> None:
        self.config = config or NLMSConfig()
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
        self._residual_rms_history: list[float] = []
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
        self._residual_rms_history = []
        self._weight_norm_history = []
        self._converged = False
        self._divergence_detected = False

    def update(
        self,
        reference_sample: float,
        desired_sample: float,
    ) -> tuple[float, float]:
        """Process one sample using normalized coefficient update."""
        if not np.isfinite(reference_sample) or not np.isfinite(desired_sample):
            raise ValueError("reference and desired samples must be finite")

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

        reference_energy = float(
            np.dot(
                self._reference_buffer,
                self._reference_buffer,
            )
        )
        denominator = self.config.epsilon + reference_energy
        if reference_energy <= self.config.epsilon:
            coefficient_update = np.zeros_like(self.weights)
        else:
            coefficient_update = (
                self.config.step_size
                * error
                * self._reference_buffer
                / denominator
            )

        updated_weights = self.weights + coefficient_update
        if not np.all(np.isfinite(updated_weights)):
            self._divergence_detected = True
            raise RuntimeError("NLMS coefficient update produced non-finite weights")

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
        recent_error = np.asarray(self._error_history[-256:])
        residual_rms = float(np.sqrt(np.mean(recent_error * recent_error)))
        self._residual_rms_history.append(residual_rms)

        weight_norm = float(np.linalg.norm(self.weights))
        self._weight_norm_history.append(weight_norm)

        if self.config.record_coefficient_history:
            limit = self.config.coefficient_history_limit
            if limit > 0:
                self._coefficient_history.append(self.weights.copy())
                if len(self._coefficient_history) > limit:
                    del self._coefficient_history[
                        :len(self._coefficient_history) - limit
                    ]

        if len(recent_error) >= 256:
            if residual_rms < 1e-9:
                self._converged = True
        return output, error

    def process(
        self,
        reference: np.ndarray,
        desired: np.ndarray,
    ) -> NLMSResult:
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
        residual_window = min(256, error.size)
        return NLMSResult(
            output=output,
            error=error,
            weights=self.weights.copy(),
            coefficient_history=self.coefficient_history,
            residual_rms_history=np.asarray(
                self._residual_rms_history,
                dtype=np.float64,
            ),
            converged=self._converged,
            divergence_detected=self._divergence_detected,
            mean_error_power=float(np.mean(error_power)),
            final_error_power=float(
                np.mean(error[-residual_window:] * error[-residual_window:])
            ),
            mean_weight_norm=float(np.mean(self._weight_norm_history)),
        )

    def process_block(
        self,
        reference_block: np.ndarray,
        desired_block: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process a streaming block and preserve state for the next block."""
        output, error = [], []
        reference_values = validate_mono(reference_block, "reference_block")
        desired_values = validate_mono(desired_block, "desired_block")
        if reference_values.size != desired_values.size:
            raise ValueError("reference and desired block lengths must match")
        for reference_sample, desired_sample in zip(
            reference_values,
            desired_values,
        ):
            block_output, block_error = self.update(
                reference_sample,
                desired_sample,
            )
            output.append(block_output)
            error.append(block_error)
        return (
            np.asarray(output, dtype=np.float64),
            np.asarray(error, dtype=np.float64),
        )


def rms_error(error: np.ndarray) -> float:
    values = np.asarray(error, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("error must not be empty")
    return float(np.sqrt(np.mean(values * values)))


def generate_changing_amplitude_reference(
    sample_rate: int = 48_000,
    duration_seconds: float = 2.0,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate identical low/high-amplitude reference and known desired path."""
    frame_count = int(sample_rate * duration_seconds)
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(seed)
    reference = rng.normal(0.0, 1.0, frame_count)
    envelope = np.concatenate(
        (
            np.linspace(0.05, 1.0, frame_count // 3),
            np.ones(frame_count // 3),
            np.linspace(1.0, 0.05, frame_count - 2 * (frame_count // 3)),
        )
    )
    reference *= envelope

    delayed = np.zeros(frame_count, dtype=np.float64)
    delayed[8:] = reference[:-8]
    delayed_gain = 0.75 * delayed
    desired = delayed_gain + 0.10 * rng.normal(
        0.0,
        1.0,
        frame_count,
    )
    return reference, desired


def generate_constant_reference(
    sample_rate: int = 48_000,
    duration_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = int(sample_rate * duration_seconds)
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    reference = np.sin(2.0 * np.pi * 100.0 * time)
    desired = 0.65 * np.sin(
        2.0 * np.pi * 100.0 * time - np.deg2rad(20.0)
    )
    return reference, desired


def compare_lms_and_nlms(
    reference: np.ndarray,
    desired: np.ndarray,
    filter_length: int = 64,
    lms_learning_rate: float = 0.01,
    nlms_step_size: float = 0.5,
    epsilon: float = 1e-6,
) -> dict[str, object]:
    """Run LMS and NLMS on exactly the same reference and desired arrays."""
    lms_result = LMSFilter(
        LMSConfig(
            filter_length=filter_length,
            learning_rate=lms_learning_rate,
        )
    ).process(reference, desired)
    nlms_result = NLMSFilter(
        NLMSConfig(
            filter_length=filter_length,
            step_size=nlms_step_size,
            epsilon=epsilon,
        )
    ).process(reference, desired)

    lms_final = rms_error(lms_result.error[-256:])
    nlms_final = rms_error(nlms_result.error[-256:])
    return {
        "lms": lms_result,
        "nlms": nlms_result,
        "lms_final_rms": lms_final,
        "nlms_final_rms": nlms_final,
    }


def run_comparison_experiments(
    filter_length: int = 64,
) -> list[ComparisonReport]:
    experiments = {
        "constant_reference": generate_constant_reference(),
        "changing_amplitude": generate_changing_amplitude_reference(),
    }
    reports: list[ComparisonReport] = []
    for name, (reference, desired) in experiments.items():
        comparison = compare_lms_and_nlms(
            reference,
            desired,
            filter_length=filter_length,
            lms_learning_rate=0.001,
            nlms_step_size=0.5,
        )
        lms_result = comparison["lms"]
        nlms_result = comparison["nlms"]
        reports.append(
            ComparisonReport(
                name=name,
                filter_length=filter_length,
                lms_step_size=0.001,
                nlms_step_size=0.5,
                lms_mean_error_power=lms_result.mean_error_power,
                nlms_mean_error_power=nlms_result.mean_error_power,
                lms_final_rms_error=comparison["lms_final_rms"],
                nlms_final_rms_error=comparison["nlms_final_rms"],
                lms_converged=lms_result.converged,
                nlms_converged=nlms_result.converged,
                lms_divergence_detected=lms_result.divergence_detected,
                nlms_divergence_detected=nlms_result.divergence_detected,
            )
        )
    return reports


def write_report(
    path: str | Path,
    reports: Iterable[ComparisonReport],
) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for report in reports:
        payload.append(
            {
                "name": report.name,
                "filter_length": report.filter_length,
                "lms_step_size": report.lms_step_size,
                "nlms_step_size": report.nlms_step_size,
                "lms_mean_error_power": report.lms_mean_error_power,
                "nlms_mean_error_power": report.nlms_mean_error_power,
                "lms_final_rms_error": report.lms_final_rms_error,
                "nlms_final_rms_error": report.nlms_final_rms_error,
                "lms_converged": report.lms_converged,
                "nlms_converged": report.nlms_converged,
                "lms_divergence_detected": report.lms_divergence_detected,
                "nlms_divergence_detected": report.nlms_divergence_detected,
                "measurement_type": "offline_identical_reference_comparison",
            }
        )
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_visualization(
    reference: np.ndarray,
    desired: np.ndarray,
    config: NLMSConfig,
) -> tuple[LMSResult | object, NLMSResult]:
    """Plot coefficients, errors and residual RMS for LMS/NLMS comparison."""
    import matplotlib.pyplot as plt

    comparison = compare_lms_and_nlms(
        reference,
        desired,
        filter_length=config.filter_length,
        lms_learning_rate=0.001,
        nlms_step_size=config.step_size,
        epsilon=config.epsilon,
    )
    lms_result = comparison["lms"]
    nlms_result = comparison["nlms"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(lms_result.error, lw=0.5, label="LMS")
    axes[0, 0].plot(nlms_result.error, lw=0.5, label="NLMS")
    axes[0, 0].set_title("Error signal")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Error")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.2)

    axes[0, 1].plot(nlms_result.residual_rms_history, lw=0.8)
    axes[0, 1].set_title("NLMS residual RMS history")
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].set_ylabel("Residual RMS")
    axes[0, 1].grid(True, alpha=0.2)

    if nlms_result.coefficient_history.size:
        axes[1, 0].imshow(
            nlms_result.coefficient_history.T,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
        )
    axes[1, 0].set_title("NLMS coefficient history")
    axes[1, 0].set_xlabel("Update")
    axes[1, 0].set_ylabel("Coefficient index")

    indices = np.arange(1, len(nlms_result.error) + 1)
    lms_cumulative = np.sqrt(
        np.cumsum(lms_result.error * lms_result.error) / indices
    )
    nlms_cumulative = np.sqrt(
        np.cumsum(nlms_result.error * nlms_result.error) / indices
    )
    axes[1, 1].plot(lms_cumulative, lw=0.8, label="LMS")
    axes[1, 1].plot(nlms_cumulative, lw=0.8, label="NLMS")
    axes[1, 1].set_title("Convergence speed: cumulative RMS")
    axes[1, 1].set_xlabel("Sample")
    axes[1, 1].set_ylabel("RMS error")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.2)

    fig.suptitle(
        "Offline LMS/NLMS comparison; no physical anti-noise is generated",
        fontweight="bold",
    )
    plt.show()
    return lms_result, nlms_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run offline NLMS adaptive-filter experiments"
    )
    parser.add_argument(
        "--experiment",
        choices=["constant_reference", "changing_amplitude"],
        default="changing_amplitude",
    )
    parser.add_argument("--filter-length", type=int, default=64)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--json-report", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    try:
        config = NLMSConfig(
            filter_length=args.filter_length,
            step_size=args.step_size,
            epsilon=args.epsilon,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.experiment == "constant_reference":
        reference, desired = generate_constant_reference()
    else:
        reference, desired = generate_changing_amplitude_reference()

    lms_result, nlms_result = run_visualization(
        reference,
        desired,
        config,
    )
    print("Offline LMS/NLMS comparison")
    print(
        f"LMS mean error power: {lms_result.mean_error_power:.8f}"
    )
    print(
        f"NLMS mean error power: {nlms_result.mean_error_power:.8f}"
    )
    print(
        f"LMS final RMS: {rms_error(lms_result.error[-256:]):.8f}"
    )
    print(
        f"NLMS final RMS: {rms_error(nlms_result.error[-256:]):.8f}"
    )

    if args.json_report:
        write_report(
            args.json_report,
            run_comparison_experiments(args.filter_length),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
