"""Experimental offline ANC algorithm comparison dashboard.

This module is separate from the desktop UI and audio I/O. It generates one
shared synthetic reference for all algorithms, then reports only outputs
produced by the simulation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import csv
import json
import logging
from pathlib import Path
import time

import numpy as np

try:
    from .lms import LMSConfig, LMSFilter
    from .nlms import NLMSConfig, NLMSFilter
    from .fxlms import (
        FxLMSConfig,
        FxLMSFilter,
        fixed_phase_inversion,
        noise_reduction_db,
        rms,
        secondary_path_with_delay,
    )
except ImportError:
    from lms import LMSConfig, LMSFilter
    from nlms import NLMSConfig, NLMSFilter
    from fxlms import (
        FxLMSConfig,
        FxLMSFilter,
        fixed_phase_inversion,
        noise_reduction_db,
        rms,
        secondary_path_with_delay,
    )


LOGGER = logging.getLogger("hybridanc.anc_dashboard")
ALGORITHMS = ("fixed_phase_inversion", "lms", "nlms", "fxlms")
NOISE_TYPES = ("sine", "multi_sine", "white", "changing_sine")


@dataclass(frozen=True)
class DashboardConfig:
    noise_frequency_hz: float = 100.0
    noise_amplitude: float = 0.50
    noise_type: str = "sine"
    sample_rate: int = 48_000
    duration_seconds: float = 1.0
    filter_length: int = 64
    learning_rate: float = 0.001
    secondary_delay_samples: int = 16
    secondary_path_model: str = "unit_delay"
    seed: int = 2026

    def __post_init__(self) -> None:
        if not np.isfinite(self.noise_frequency_hz) or not 0.0 < self.noise_frequency_hz < self.sample_rate / 2.0:
            raise ValueError("noise_frequency_hz must be below Nyquist")
        if not np.isfinite(self.noise_amplitude) or self.noise_amplitude < 0:
            raise ValueError("noise_amplitude must be finite and non-negative")
        if self.noise_type not in NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {NOISE_TYPES}")
        if self.sample_rate not in (44_100, 48_000):
            raise ValueError("sample_rate must be 44100 or 48000 Hz")
        if not np.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if not isinstance(self.filter_length, int) or not 1 <= self.filter_length <= 4096:
            raise ValueError("filter_length must be between 1 and 4096")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not isinstance(self.secondary_delay_samples, int) or not 0 <= self.secondary_delay_samples < self.filter_length:
            raise ValueError("secondary_delay_samples must fit the filter length")
        if self.secondary_path_model not in ("unit_delay", "delayed_lead", "delayed_resonant"):
            raise ValueError("unsupported secondary_path_model")


@dataclass(frozen=True)
class AlgorithmResult:
    name: str
    anti_noise: np.ndarray
    residual: np.ndarray
    residual_rms: float
    noise_reduction_db: float
    convergence_rms: np.ndarray
    coefficient_history: np.ndarray
    convergence_time_samples: int | None
    mean_processing_time_ms: float
    divergence_detected: bool


@dataclass(frozen=True)
class ExperimentSummary:
    name: str
    noise_frequency_hz: float
    noise_amplitude: float
    noise_type: str
    sample_rate: int
    duration_seconds: float
    filter_length: int
    learning_rate: float
    secondary_delay_samples: int
    secondary_path_model: str
    residual_rms: float
    noise_reduction_db: float
    convergence_time_samples: int | None
    mean_processing_time_ms: float
    divergence_detected: bool


def generate_noise(config: DashboardConfig) -> np.ndarray:
    """Generate the single shared reference signal for the experiment."""
    frame_count = int(config.sample_rate * config.duration_seconds)
    time_axis = np.arange(frame_count, dtype=np.float64) / config.sample_rate
    amplitude = config.noise_amplitude

    if config.noise_type == "sine":
        noise = amplitude * np.sin(
            2.0 * np.pi * config.noise_frequency_hz * time_axis
        )
    elif config.noise_type == "multi_sine":
        noise = (
            0.60 * amplitude
            * np.sin(2.0 * np.pi * config.noise_frequency_hz * time_axis)
            + 0.30 * amplitude
            * np.sin(2.0 * np.pi * 2.5 * config.noise_frequency_hz * time_axis)
            + 0.10 * amplitude
            * np.sin(2.0 * np.pi * 6.0 * config.noise_frequency_hz * time_axis)
        )
    elif config.noise_type == "white":
        rng = np.random.default_rng(config.seed)
        noise = amplitude * rng.normal(0.0, 1.0, frame_count)
    elif config.noise_type == "changing_sine":
        envelope = np.linspace(0.15, 1.0, frame_count)
        noise = (
            amplitude
            * envelope
            * np.sin(2.0 * np.pi * config.noise_frequency_hz * time_axis)
        )
    else:
        raise ValueError(f"unsupported noise type: {config.noise_type}")

    if not np.all(np.isfinite(noise)):
        raise RuntimeError("generated noise contains non-finite samples")
    return noise


def generate_secondary_path(config: DashboardConfig) -> np.ndarray:
    """Generate an explicit offline secondary-path FIR model."""
    impulse = np.zeros(config.filter_length, dtype=np.float64)
    delay = config.secondary_delay_samples
    impulse[delay] = 1.0

    if config.secondary_path_model == "unit_delay":
        return impulse

    if config.secondary_path_model == "delayed_lead":
        if delay + 1 < config.filter_length:
            impulse[delay + 1] = -0.20
        if delay + 4 < config.filter_length:
            impulse[delay + 4] = 0.10
        return impulse

    if config.secondary_path_model == "delayed_resonant":
        time = np.arange(config.filter_length, dtype=np.float64) / config.sample_rate
        resonance_hz = max(
            100.0,
            min(config.noise_frequency_hz * 3.0, config.sample_rate / 4.0),
        )
        resonance = 0.15 * np.sin(
            2.0 * np.pi * resonance_hz * time
        ) * np.exp(
            -100.0 * time
        )
        impulse += resonance
        if delay == 0:
            impulse[0] = 0.8
        else:
            impulse[delay] = 1.0
        return impulse
    raise ValueError("unsupported secondary path model")


def cumulative_residual_rms(error: np.ndarray) -> np.ndarray:
    values = np.asarray(error, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("error must not be empty")
    squared = values * values
    return np.sqrt(
        np.cumsum(squared) / np.arange(1, values.size + 1)
    )


def convergence_time(
    residual_rms_history: np.ndarray,
    threshold_fraction: float = 0.10,
) -> int | None:
    if residual_rms_history.size == 0:
        return None
    threshold = (
        residual_rms_history[0]
        * threshold_fraction
    )
    matches = np.flatnonzero(
        residual_rms_history <= threshold
    )
    return int(matches[0]) if matches.size else None


def run_fixed_phase_inversion(
    config: DashboardConfig,
    original: np.ndarray,
    secondary_path: np.ndarray,
) -> AlgorithmResult:
    del secondary_path
    anti_noise = fixed_phase_inversion(original)
    residual = original + anti_noise
    residual_rms = rms(residual)
    return AlgorithmResult(
        name="Fixed phase inversion",
        anti_noise=anti_noise,
        residual=residual,
        residual_rms=residual_rms,
        noise_reduction_db=noise_reduction_db(original, residual),
        convergence_rms=cumulative_residual_rms(residual),
        coefficient_history=np.empty((0, config.filter_length)),
        convergence_time_samples=None,
        mean_processing_time_ms=0.0,
        divergence_detected=False,
    )


def run_lms(
    config: DashboardConfig,
    original: np.ndarray,
) -> AlgorithmResult:
    start_ns = time.monotonic_ns()
    adaptive = LMSFilter(
        LMSConfig(
            filter_length=config.filter_length,
            learning_rate=config.learning_rate,
        )
    )
    result = adaptive.process(original, original)
    elapsed_ms = (
        time.monotonic_ns() - start_ns
    ) / 1_000_000.0

    residual = result.error
    convergence_history = cumulative_residual_rms(residual)
    return AlgorithmResult(
        name="LMS",
        anti_noise=result.output,
        residual=residual,
        residual_rms=rms(residual),
        noise_reduction_db=noise_reduction_db(original, residual),
        convergence_rms=convergence_history,
        coefficient_history=result.coefficient_history,
        convergence_time_samples=convergence_time(convergence_history),
        mean_processing_time_ms=elapsed_ms / original.size,
        divergence_detected=result.divergence_detected,
    )


def run_nlms(
    config: DashboardConfig,
    original: np.ndarray,
) -> AlgorithmResult:
    start_ns = time.monotonic_ns()
    adaptive = NLMSFilter(
        NLMSConfig(
            filter_length=config.filter_length,
            step_size=min(1.0, max(0.001, config.learning_rate)),
        )
    )
    result = adaptive.process(original, original)
    elapsed_ms = (
        time.monotonic_ns() - start_ns
    ) / 1_000_000.0

    residual = result.error
    convergence_history = cumulative_residual_rms(residual)
    return AlgorithmResult(
        name="NLMS",
        anti_noise=result.output,
        residual=residual,
        residual_rms=rms(residual),
        noise_reduction_db=noise_reduction_db(original, residual),
        convergence_rms=convergence_history,
        coefficient_history=result.coefficient_history,
        convergence_time_samples=convergence_time(convergence_history),
        mean_processing_time_ms=elapsed_ms / original.size,
        divergence_detected=result.divergence_detected,
    )


def run_fxlms(
    config: DashboardConfig,
    original: np.ndarray,
    secondary_path: np.ndarray,
) -> AlgorithmResult:
    adaptive = FxLMSFilter(
        FxLMSConfig(
            filter_length=config.filter_length,
            learning_rate=config.learning_rate,
            convergence_threshold=1e-12,
        ),
        secondary_path,
        secondary_path,
    )
    result = adaptive.process(original, original)
    convergence_history = cumulative_residual_rms(result.residual_error)
    return AlgorithmResult(
        name="FxLMS",
        anti_noise=result.antinoise,
        residual=result.residual_error,
        residual_rms=rms(result.residual_error),
        noise_reduction_db=noise_reduction_db(
            original,
            result.residual_error,
        ),
        convergence_rms=convergence_history,
        coefficient_history=result.coefficient_history,
        convergence_time_samples=convergence_time(convergence_history),
        mean_processing_time_ms=result.mean_processing_time_ms,
        divergence_detected=result.divergence_detected,
    )


def run_comparison(
    config: DashboardConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, AlgorithmResult]]:
    """Run all algorithms against exactly the same generated noise."""
    original = generate_noise(config)
    secondary_path = generate_secondary_path(config)
    fixed = run_fixed_phase_inversion(
        config,
        original,
        secondary_path,
    )
    lms = run_lms(config, original)
    nlms = run_nlms(config, original)
    fxlms = run_fxlms(
        config,
        original,
        secondary_path,
    )
    results = {
        fixed.name: fixed,
        lms.name: lms,
        nlms.name: nlms,
        fxlms.name: fxlms,
    }
    return original, secondary_path, results


def build_summaries(
    config: DashboardConfig,
    results: dict[str, AlgorithmResult],
) -> list[ExperimentSummary]:
    summaries: list[ExperimentSummary] = []
    for name, result in results.items():
        summaries.append(
            ExperimentSummary(
                name=name,
                noise_frequency_hz=config.noise_frequency_hz,
                noise_amplitude=config.noise_amplitude,
                noise_type=config.noise_type,
                sample_rate=config.sample_rate,
                duration_seconds=config.duration_seconds,
                filter_length=config.filter_length,
                learning_rate=config.learning_rate,
                secondary_delay_samples=config.secondary_delay_samples,
                secondary_path_model=config.secondary_path_model,
                residual_rms=result.residual_rms,
                noise_reduction_db=result.noise_reduction_db,
                convergence_time_samples=result.convergence_time_samples,
                mean_processing_time_ms=result.mean_processing_time_ms,
                divergence_detected=result.divergence_detected,
            )
        )
    return summaries


def export_csv(
    path: str | Path,
    config: DashboardConfig,
    original: np.ndarray,
    results: dict[str, AlgorithmResult],
) -> None:
    """Export one summary row per algorithm and full length waveform rows."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["record_type", "algorithm", "sample_index", "sample"])
        for index, value in enumerate(original):
            writer.writerow(
                ["original_noise", "shared_input", index, float(value)]
            )
        for name, result in results.items():
            for index, value in enumerate(result.anti_noise):
                writer.writerow(
                    ["anti_noise", name, index, float(value)]
                )
            for index, value in enumerate(result.residual):
                writer.writerow(
                    ["residual", name, index, float(value)]
                )
        for summary in build_summaries(config, results):
            writer.writerow(
                [
                    "summary",
                    summary.name,
                    "residual_rms",
                    summary.residual_rms,
                ]
            )
            writer.writerow(
                [
                    "summary",
                    summary.name,
                    "noise_reduction_db",
                    summary.noise_reduction_db,
                ]
            )
            writer.writerow(
                [
                    "summary",
                    summary.name,
                    "convergence_time_samples",
                    summary.convergence_time_samples
                    if summary.convergence_time_samples is not None
                    else "",
                ]
            )
            writer.writerow(
                [
                    "summary",
                    summary.name,
                    "mean_processing_time_ms",
                    summary.mean_processing_time_ms,
                ]
            )


def export_json(
    path: str | Path,
    config: DashboardConfig,
    results: dict[str, AlgorithmResult],
) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "configuration": asdict(config),
        "summaries": [
            asdict(summary)
            for summary in build_summaries(config, results)
        ],
    }
    report_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def run_dashboard(
    config: DashboardConfig,
    algorithm: str = "all",
) -> tuple[np.ndarray, dict[str, AlgorithmResult]]:
    """Run the selected algorithm or all algorithms against shared noise."""
    if algorithm != "all" and algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be 'all' or one of {ALGORITHMS}")
    original, secondary_path, all_results = run_comparison(config)
    if algorithm == "all":
        return original, all_results
    key_map = {
        "fixed_phase_inversion": "Fixed phase inversion",
        "lms": "LMS",
        "nlms": "NLMS",
        "fxlms": "FxLMS",
    }
    selected = key_map[algorithm]
    return original, {selected: all_results[selected]}


def run_visualization(
    config: DashboardConfig,
    algorithm: str = "all",
) -> tuple[np.ndarray, dict[str, AlgorithmResult]]:
    """Display the offline comparison dashboard."""
    import matplotlib.pyplot as plt

    original, results = run_dashboard(config, algorithm)
    time_axis = np.arange(original.size) / config.sample_rate
    frequencies = np.fft.rfftfreq(
        original.size,
        d=1.0 / config.sample_rate,
    )
    original_spectrum = np.abs(np.fft.rfft(original))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9),
        constrained_layout=True,
    )
    waveform_axis = axes[0, 0]
    residual_axis = axes[0, 1]
    spectrum_axis = axes[1, 0]
    convergence_axis = axes[1, 1]

    waveform_axis.plot(
        time_axis,
        original,
        lw=0.7,
        color="black",
        label="Original noise",
    )
    waveform_axis.set_title("Original noise waveform")
    waveform_axis.set_xlabel("Time (seconds)")
    waveform_axis.set_ylabel("Amplitude")
    waveform_axis.grid(True, alpha=0.2)

    colors = {
        "Fixed phase inversion": "tab:orange",
        "LMS": "tab:blue",
        "NLMS": "tab:green",
        "FxLMS": "tab:red",
    }
    for name, result in results.items():
        waveform_axis.plot(
            time_axis,
            result.anti_noise,
            lw=0.5,
            alpha=0.7,
            color=colors.get(name),
            label=f"{name} anti-noise",
        )
        residual_axis.plot(
            time_axis,
            result.residual,
            lw=0.5,
            alpha=0.8,
            color=colors.get(name),
            label=name,
        )
        convergence_axis.plot(
            result.convergence_rms,
            lw=0.8,
            color=colors.get(name),
            label=name,
        )
        residual_spectrum = np.abs(np.fft.rfft(result.residual))
        spectrum_axis.plot(
            frequencies,
            20.0 * np.log10(
                np.maximum(residual_spectrum, np.finfo(np.float64).tiny)
            ),
            lw=0.7,
            alpha=0.8,
            color=colors.get(name),
            label=name,
        )

    waveform_axis.legend(fontsize=8)
    residual_axis.set_title("Residual waveform")
    residual_axis.set_xlabel("Time (seconds)")
    residual_axis.set_ylabel("Amplitude")
    residual_axis.grid(True, alpha=0.2)
    residual_axis.legend(fontsize=8)

    spectrum_axis.plot(
        frequencies,
        20.0 * np.log10(
            np.maximum(original_spectrum, np.finfo(np.float64).tiny)
        ),
        lw=0.8,
        color="black",
        label="Original",
    )
    spectrum_axis.set_title("Frequency spectrum")
    spectrum_axis.set_xlabel("Frequency (Hz)")
    spectrum_axis.set_ylabel("Magnitude (dB)")
    spectrum_axis.set_xlim(0.0, config.sample_rate / 2.0)
    spectrum_axis.legend(fontsize=8)
    spectrum_axis.grid(True, alpha=0.2)

    convergence_axis.set_title("Residual RMS convergence")
    convergence_axis.set_xlabel("Sample")
    convergence_axis.set_ylabel("Residual RMS")
    convergence_axis.legend(fontsize=8)
    convergence_axis.grid(True, alpha=0.2)

    lines = ["Measured experiment output:", ""]
    for name, result in results.items():
        lines.append(
            f"{name:<22} RMS={result.residual_rms:.6f} "
            f"NR={result.noise_reduction_db:.2f} dB "
            f"cost={result.mean_processing_time_ms:.4f} ms/sample"
        )
    residual_axis.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=residual_axis.transAxes,
        va="top",
        family="monospace",
        fontsize=8,
    )

    fig.suptitle(
        "HybridANC offline ANC comparison dashboard "
        "(no physical anti-noise playback)",
        fontweight="bold",
    )
    return_figure = plt.show
    fig._hybridanc_dashboard = True
    return_figure()
    return original, results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experimental offline ANC algorithm comparison dashboard"
    )
    parser.add_argument(
        "--algorithm",
        choices=["all", *ALGORITHMS],
        default="all",
    )
    parser.add_argument(
        "--noise-type",
        choices=list(NOISE_TYPES),
        default="sine",
    )
    parser.add_argument("--noise-frequency", type=float, default=100.0)
    parser.add_argument("--noise-amplitude", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=int, choices=[44100, 48000], default=48000)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--filter-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--secondary-delay", type=int, default=16)
    parser.add_argument(
        "--secondary-path",
        choices=["unit_delay", "delayed_lead", "delayed_resonant"],
        default="delayed_lead",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    try:
        config = DashboardConfig(
            noise_frequency_hz=args.noise_frequency,
            noise_amplitude=args.noise_amplitude,
            noise_type=args.noise_type,
            sample_rate=args.sample_rate,
            duration_seconds=args.duration,
            filter_length=args.filter_length,
            learning_rate=args.learning_rate,
            secondary_delay_samples=args.secondary_delay,
            secondary_path_model=args.secondary_path,
            seed=args.seed,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    original, results = run_visualization(
        config,
        args.algorithm,
    )

    if args.csv_output:
        export_csv(
            args.csv_output,
            config,
            original,
            results,
        )
    if args.json_output:
        export_json(
            args.json_output,
            config,
            results,
        )

    for name, result in results.items():
        print(
            f"{name}: RMS={result.residual_rms:.8f}, "
            f"noise reduction={result.noise_reduction_db:.3f} dB, "
            f"convergence={result.convergence_time_samples}, "
            f"processing={result.mean_processing_time_ms:.4f} ms/sample"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
