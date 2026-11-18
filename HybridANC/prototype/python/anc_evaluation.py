"""Offline acoustic-performance evaluation for before/after ANC recordings.

This module evaluates recorded electrical microphone samples. It does not
convert RMS to absolute SPL unless a traceable microphone calibration factor
and reference are supplied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy import signal


LOGGER = logging.getLogger("hybridanc.anc_evaluation")
SUPPORTED_SAMPLE_RATES = (44_100, 48_000)

FREQUENCY_BANDS = (
    ("Sub-bass", 20.0, 60.0),
    ("Bass", 60.0, 250.0),
    ("Low-mids", 250.0, 500.0),
    ("Mids", 500.0, 2_000.0),
    ("High-mids", 2_000.0, 4_000.0),
    ("Presence", 4_000.0, 8_000.0),
    ("Brilliance", 8_000.0, 20_000.0),
)


@dataclass(frozen=True)
class EvaluationConfig:
    sample_rate: int = 48_000
    fft_size: int = 4096
    microphone_calibration_factor_db: float | None = None
    measurement_location: str = "unspecified"
    gain_setting: str = "unchanged"
    before_trial_count: int = 1
    after_trial_count: int = 1

    def __post_init__(self) -> None:
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError("sample_rate must be 44100 or 48000 Hz")
        if not isinstance(self.fft_size, int) or not 256 <= self.fft_size <= 65536:
            raise ValueError("fft_size must be between 256 and 65536")
        if self.microphone_calibration_factor_db is not None:
            if not np.isfinite(self.microphone_calibration_factor_db):
                raise ValueError("microphone_calibration_factor_db must be finite")
        if not isinstance(self.before_trial_count, int) or self.before_trial_count <= 0:
            raise ValueError("before_trial_count must be positive")
        if not isinstance(self.after_trial_count, int) or self.after_trial_count <= 0:
            raise ValueError("after_trial_count must be positive")


@dataclass(frozen=True)
class BandAttenuation:
    name: str
    low_hz: float
    high_hz: float
    rms_before: float
    rms_after: float
    attenuation_db: float


@dataclass(frozen=True)
class StabilityMetrics:
    sample_std_before: float
    sample_std_after: float
    before_rms_history: tuple[float, ...]
    after_rms_history: tuple[float, ...]
    relative_rms_drift: float
    stability_status: str


@dataclass(frozen=True)
class EvaluationSummary:
    trial_name: str
    sample_rate: int
    frame_count: int
    rms_before: float
    rms_after: float
    noise_reduction_db: float
    convergence_time_samples: int | None
    measurement_location: str
    gain_setting: str
    microphone_calibration_factor_db: float | None
    absolute_spl_status: str
    repeatability_status: str
    confidence_note: str


@dataclass(frozen=True)
class MeasurementHistoryEntry:
    trial_name: str
    rms_before: float
    rms_after: float
    noise_reduction_db: float
    source_before: str
    source_after: str


def validate_mono_audio(
    audio: object,
    *,
    name: str = "audio",
) -> np.ndarray:
    values = np.asarray(audio)
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if values.ndim == 2:
        if values.shape[1] not in (1, 2):
            raise ValueError(f"{name} must be mono or stereo")
        values = np.mean(values, axis=1)
    elif values.ndim != 1:
        raise ValueError(f"{name} must be mono or one-dimensional")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"{name} must contain floating-point samples")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite samples")
    return np.asarray(values, dtype=np.float64)


def load_mono_wav(
    path: str | Path,
    *,
    name: str = "recording",
    sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    recording_path = Path(path)
    if not recording_path.is_file():
        raise FileNotFoundError(f"{name} WAV not found: {recording_path}")
    samples, sample_rate_from_file = sf.read(
        str(recording_path),
        dtype="float64",
        always_2d=True,
    )
    if samples.size == 0:
        raise ValueError(f"{name} recording is empty")
    if sample_rate is not None and sample_rate_from_file != sample_rate:
        raise ValueError(
            f"{name} sample rate mismatch: "
            f"{sample_rate_from_file} != {sample_rate}"
        )
    mono = np.mean(samples, axis=1)
    if not np.all(np.isfinite(mono)):
        raise ValueError(f"{name} contains non-finite samples")
    return mono, int(sample_rate_from_file)


def align_before_after(
    before: np.ndarray,
    after: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    before_values = validate_mono_audio(before, name="before")
    after_values = validate_mono_audio(after, name="after")
    if before_values.size != after_values.size:
        frame_count = min(before_values.size, after_values.size)
        LOGGER.warning(
            "Before/after lengths differ; comparing the first %d frames",
            frame_count,
        )
        before_values = before_values[:frame_count]
        after_values = after_values[:frame_count]
    return before_values, after_values


def calculate_rms(audio: np.ndarray) -> float:
    values = validate_mono_audio(audio)
    return float(
        np.sqrt(
            np.mean(
                np.square(values, dtype=np.float64)
            )
        )
    )


def calculate_noise_reduction_db(
    rms_before: float,
    rms_after: float,
) -> float:
    if not np.isfinite(rms_before) or not np.isfinite(rms_after):
        raise ValueError("RMS values must be finite")
    if rms_before < 0 or rms_after < 0:
        raise ValueError("RMS values must be non-negative")
    return float(
        20.0 * np.log10(
            max(rms_after, np.finfo(np.float64).tiny)
            / max(rms_before, np.finfo(np.float64).tiny)
        )
    )


def band_attenuation(
    before: np.ndarray,
    after: np.ndarray,
    sample_rate: int,
    fft_size: int = 4096,
) -> list[BandAttenuation]:
    before_values = validate_mono_audio(before, name="before")
    after_values = validate_mono_audio(after, name="after")
    before_values, after_values = align_before_after(
        before_values,
        after_values,
    )
    if fft_size <= 0:
        raise ValueError("fft_size must be positive")

    fft_count = min(
        fft_size,
        before_values.size,
        after_values.size,
    )
    fft_count = max(1, fft_count)
    window = np.hanning(fft_count).astype(np.float64)
    before_segment = before_values[-window.size:] * window
    after_segment = after_values[-window.size:] * window
    before_spectrum = np.abs(np.fft.rfft(before_segment))
    after_spectrum = np.abs(np.fft.rfft(after_segment))
    frequencies = np.fft.rfftfreq(
        window.size,
        d=1.0 / sample_rate,
    )
    nyquist = sample_rate / 2.0

    results: list[BandAttenuation] = []
    for name, low_hz, high_hz in FREQUENCY_BANDS:
        high_hz = min(high_hz, nyquist)
        mask = (
            (frequencies >= low_hz)
            & (frequencies <= high_hz)
        )
        if not np.any(mask):
            continue
        before_rms = float(
            np.sqrt(
                np.mean(
                    np.square(before_spectrum[mask])
                )
            )
        )
        after_rms = float(
            np.sqrt(
                np.mean(
                    np.square(after_spectrum[mask])
                )
            )
        )
        attenuation = calculate_noise_reduction_db(
            before_rms,
            after_rms,
        )
        results.append(
            BandAttenuation(
                name=name,
                low_hz=low_hz,
                high_hz=high_hz,
                rms_before=before_rms,
                rms_after=after_rms,
                attenuation_db=attenuation,
            )
        )
    return results


def spectral_comparison(
    before: np.ndarray,
    after: np.ndarray,
    sample_rate: int,
    fft_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before_values, after_values = align_before_after(before, after)
    available = min(before_values.size, after_values.size)
    if available <= 0:
        raise ValueError("before and after must contain samples")
    fft_count = min(fft_size, available)
    if fft_count < 256:
        fft_count = fft_count
    fft_count = max(1, fft_count)
    window = np.hanning(fft_count).astype(np.float64)
    before_segment = before_values[-fft_count:] * window
    after_segment = after_values[-fft_count:] * window
    frequencies = np.fft.rfftfreq(
        fft_count,
        d=1.0 / sample_rate,
    )
    before_magnitude = np.abs(np.fft.rfft(before_segment))
    after_magnitude = np.abs(np.fft.rfft(after_segment))
    return frequencies, before_magnitude, after_magnitude


def calculate_convergence_time(
    residual: np.ndarray,
    *,
    block_size: int = 512,
    threshold_fraction: float = 0.10,
) -> int | None:
    values = validate_mono_audio(residual, name="residual")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0.0 < threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must be in (0, 1]")
    block_rms = np.asarray(
        [
            calculate_rms(values[index:index + block_size])
            for index in range(0, values.size, block_size)
        ]
    )
    if block_rms.size == 0:
        return None
    initial = float(block_rms[0])
    threshold = initial * threshold_fraction
    matches = np.flatnonzero(block_rms <= threshold)
    return int(matches[0] * block_size) if matches.size else None


def stability_metrics(
    before: np.ndarray,
    after: np.ndarray,
    block_size: int = 512,
) -> StabilityMetrics:
    before_values, after_values = align_before_after(before, after)
    before_history = tuple(
        calculate_rms(before_values[index:index + block_size])
        for index in range(0, before_values.size, block_size)
    )
    after_history = tuple(
        calculate_rms(after_values[index:index + block_size])
        for index in range(0, after_values.size, block_size)
    )
    before_array = np.asarray(before_history)
    after_array = np.asarray(after_history)
    combined = np.concatenate((before_array, after_array))
    mean_level = float(np.mean(combined))
    drift = (
        float(np.max(after_array) - np.min(after_array)) / max(mean_level, np.finfo(np.float64).tiny)
        if after_array.size
        else 0.0
    )
    status = (
        "stable"
        if drift <= 0.25
        else "unstable_or_variable"
    )
    return StabilityMetrics(
        sample_std_before=float(np.std(before_values)),
        sample_std_after=float(np.std(after_values)),
        before_rms_history=before_history,
        after_rms_history=after_history,
        relative_rms_drift=drift,
        stability_status=status,
    )


def evaluate(
    before: np.ndarray,
    after: np.ndarray,
    config: EvaluationConfig,
    *,
    trial_name: str = "unnamed_trial",
) -> tuple[EvaluationSummary, list[BandAttenuation], StabilityMetrics, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    before_values, after_values = align_before_after(before, after)
    rms_before = calculate_rms(before_values)
    rms_after = calculate_rms(after_values)
    reduction_db = calculate_noise_reduction_db(
        rms_before,
        rms_after,
    )
    convergence = calculate_convergence_time(
        after_values,
        block_size=max(128, config.fft_size // 8),
    )
    bands = band_attenuation(
        before_values,
        after_values,
        config.sample_rate,
        config.fft_size,
    )
    stability = stability_metrics(
        before_values,
        after_values,
        block_size=max(128, config.fft_size // 8),
    )
    spectrum = spectral_comparison(
        before_values,
        after_values,
        config.sample_rate,
        config.fft_size,
    )

    repeatable = (
        config.before_trial_count > 1
        and config.after_trial_count > 1
    )
    repeatability_status = (
        "repeat_trials_available"
        if repeatable
        else "single_trial_or_insufficient_repeatability"
    )
    absolute_status = (
        "calibrated_spl_requires_confirmed_reference"
        if config.microphone_calibration_factor_db is not None
        else "uncalibrated_electrical_rms"
    )
    confidence_note = (
        "Microphone placement, gain and environment must be documented. "
        "One before/after pair is not a repeatability estimate."
        if not repeatable
        else "Repeat counts exist, but standard deviation or trial-to-trial "
             "variation must still be reported."
    )
    summary = EvaluationSummary(
        trial_name=trial_name,
        sample_rate=config.sample_rate,
        frame_count=int(before_values.size),
        rms_before=rms_before,
        rms_after=rms_after,
        noise_reduction_db=reduction_db,
        convergence_time_samples=convergence,
        measurement_location=config.measurement_location,
        gain_setting=config.gain_setting,
        microphone_calibration_factor_db=config.microphone_calibration_factor_db,
        absolute_spl_status=absolute_status,
        repeatability_status=repeatability_status,
        confidence_note=confidence_note,
    )
    return summary, bands, stability, spectrum


def load_evaluation_wavs(
    before_path: str | Path,
    after_path: str | Path,
    config: EvaluationConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    before, before_rate = load_mono_wav(
        before_path,
        name="before",
        sample_rate=config.sample_rate,
    )
    after, after_rate = load_mono_wav(
        after_path,
        name="after",
        sample_rate=config.sample_rate,
    )
    if before_rate != after_rate:
        raise ValueError("before and after recordings use different sample rates")
    return before, after, before_rate


def export_csv(
    path: str | Path,
    summary: EvaluationSummary,
    bands: Iterable[BandAttenuation],
    stability: StabilityMetrics,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["record_type", "name", "value"])
        for key, value in asdict(summary).items():
            writer.writerow(["summary", key, value])
        for band in bands:
            writer.writerow(
                ["band", band.name, band.attenuation_db]
            )
            writer.writerow(
                ["band_rms_before", band.name, band.rms_before]
            )
            writer.writerow(
                ["band_rms_after", band.name, band.rms_after]
            )
        writer.writerow(
            ["stability", "relative_rms_drift", stability.relative_rms_drift]
        )
        writer.writerow(
            ["stability", "status", stability.stability_status]
        )
        writer.writerow(
            ["stability", "sample_std_before", stability.sample_std_before]
        )
        writer.writerow(
            ["stability", "sample_std_after", stability.sample_std_after]
        )
    return report_path


def export_json(
    path: str | Path,
    summary: EvaluationSummary,
    bands: Iterable[BandAttenuation],
    stability: StabilityMetrics,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": asdict(summary),
        "frequency_bands": [
            asdict(band)
            for band in bands
        ],
        "stability": asdict(stability),
    }
    report_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return report_path


def export_history_csv(
    path: str | Path,
    history: Iterable[MeasurementHistoryEntry],
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_name",
        "rms_before",
        "rms_after",
        "noise_reduction_db",
        "source_before",
        "source_after",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in history:
            writer.writerow(asdict(entry))
    return report_path


def run_visualization(
    before: np.ndarray,
    after: np.ndarray,
    summary: EvaluationSummary,
    bands: Iterable[BandAttenuation],
    stability: StabilityMetrics,
    spectrum: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    before_values, after_values = align_before_after(before, after)
    frequencies, before_magnitude, after_magnitude = spectrum
    band_values = list(bands)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9),
        constrained_layout=True,
    )
    time_axis = np.arange(before_values.size) / summary.sample_rate
    axes[0, 0].plot(
        time_axis,
        before_values,
        lw=0.5,
        label="Before",
        alpha=0.8,
    )
    axes[0, 0].plot(
        time_axis,
        after_values,
        lw=0.5,
        label="After",
        alpha=0.8,
    )
    axes[0, 0].set_title("Before/after time-domain measurements")
    axes[0, 0].set_xlabel("Time (seconds)")
    axes[0, 0].set_ylabel("Microphone amplitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.2)

    axes[0, 1].plot(
        frequencies,
        20.0 * np.log10(
            np.maximum(before_magnitude, np.finfo(np.float64).tiny)
        ),
        lw=0.8,
        label="Before",
    )
    axes[0, 1].plot(
        frequencies,
        20.0 * np.log10(
            np.maximum(after_magnitude, np.finfo(np.float64).tiny)
        ),
        lw=0.8,
        label="After",
    )
    axes[0, 1].set_title("Before/after spectrum")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Magnitude (dB)")
    axes[0, 1].set_xlim(0.0, summary.sample_rate / 2.0)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.2)

    names = [band.name for band in band_values]
    attenuations = [band.attenuation_db for band in band_values]
    axes[1, 0].bar(names, attenuations)
    axes[1, 0].axhline(0.0, color="black", lw=0.8)
    axes[1, 0].set_title("Frequency-band attenuation")
    axes[1, 0].set_ylabel("Attenuation (dB)")
    axes[1, 0].tick_params(axis="x", rotation=35)

    axes[1, 1].text(
        0.02,
        0.98,
        (
            f"Trial: {summary.trial_name}\n"
            f"RMS before: {summary.rms_before:.8f}\n"
            f"RMS after: {summary.rms_after:.8f}\n"
            f"Noise reduction: {summary.noise_reduction_db:.3f} dB\n"
            f"Convergence: "
            f"{summary.convergence_time_samples if summary.convergence_time_samples is not None else 'not reached'}\n"
            f"Stability: {stability.stability_status}\n"
            f"Repeatability: {summary.repeatability_status}\n"
            f"SPL status: {summary.absolute_spl_status}\n"
            "\n"
            f"{summary.confidence_note}"
        ),
        transform=axes[1, 1].transAxes,
        va="top",
        family="monospace",
        fontsize=8,
    )
    axes[1, 1].set_axis_off()
    fig.suptitle(
        "HybridANC acoustic-performance evaluation",
        fontweight="bold",
    )
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline before/after acoustic-ANC performance evaluation"
    )
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--trial-name", default="physical_or_simulated_trial")
    parser.add_argument("--sample-rate", type=int, choices=[44100, 48000], default=48000)
    parser.add_argument("--fft-size", type=int, default=4096)
    parser.add_argument("--measurement-location", default="unspecified")
    parser.add_argument("--gain-setting", default="unchanged")
    parser.add_argument("--microphone-calibration-db", type=float, default=None)
    parser.add_argument("--before-trials", type=int, default=1)
    parser.add_argument("--after-trials", type=int, default=1)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    try:
        config = EvaluationConfig(
            sample_rate=args.sample_rate,
            fft_size=args.fft_size,
            microphone_calibration_factor_db=args.microphone_calibration_db,
            measurement_location=args.measurement_location,
            gain_setting=args.gain_setting,
            before_trial_count=args.before_trials,
            after_trial_count=args.after_trials,
        )
        before, after, _ = load_evaluation_wavs(
            args.before,
            args.after,
            config,
        )
        summary, bands, stability, spectrum = evaluate(
            before,
            after,
            config,
            trial_name=args.trial_name,
        )
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.error("Acoustic evaluation failed: %s", exc)
        return 1

    run_visualization(
        before,
        after,
        summary,
        bands,
        stability,
        spectrum,
    )
    if args.csv_output:
        export_csv(
            args.csv_output,
            summary,
            bands,
            stability,
        )
    if args.json_output:
        export_json(
            args.json_output,
            summary,
            bands,
            stability,
        )

    print(f"RMS before: {summary.rms_before:.8f}")
    print(f"RMS after: {summary.rms_after:.8f}")
    print(f"Noise reduction: {summary.noise_reduction_db:.3f} dB")
    print(f"Stability: {stability.stability_status}")
    print(f"Repeatability: {summary.repeatability_status}")
    print(f"SPL status: {summary.absolute_spl_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
