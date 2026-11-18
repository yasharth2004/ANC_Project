"""Causal software microphone-noise suppression.

This module improves captured microphone audio. It does not generate
headphone anti-noise or make any claim of physical active noise cancellation.

Streaming processing uses only current and past samples. Spectral analysis
uses overlap-plus-carry; the overlap consists of previous samples, never
future samples.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import argparse
import logging
import time
from typing import Iterable

import numpy as np
from scipy import signal

try:
    from .digital_filters import highpass
except ImportError:
    from digital_filters import highpass


LOGGER = logging.getLogger("hybridanc.noise_suppression")
SUPPORTED_SAMPLE_RATES = (44_100, 48_000)
SUPPORTED_FFT_SIZES = (256, 512, 1024, 2048)


@dataclass(frozen=True)
class NoiseSuppressionConfig:
    sample_rate: int = 48_000
    fft_size: int = 512
    hop_size: int = 128
    high_pass_hz: float = 80.0
    high_pass_order: int = 2
    suppression_strength: float = 1.0
    noise_floor_alpha: float = 0.05
    spectral_floor: float = 0.05
    temporal_smoothing: float = 0.80
    output_gain: float = 1.0
    bypass: bool = False

    def __post_init__(self) -> None:
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError("sample_rate must be 44100 or 48000 Hz")
        if self.fft_size not in SUPPORTED_FFT_SIZES:
            raise ValueError("fft_size must be 256, 512, 1024 or 2048")
        if not 0 < self.hop_size <= self.fft_size:
            raise ValueError("hop_size must be greater than zero and not exceed fft_size")
        if not 0.0 < self.high_pass_hz < self.sample_rate / 2.0:
            raise ValueError("high_pass_hz must be below Nyquist")
        if not 1 <= self.high_pass_order <= 10:
            raise ValueError("high_pass_order must be between 1 and 10")
        if not 0.0 <= self.suppression_strength <= 2.0:
            raise ValueError("suppression_strength must be between 0.0 and 2.0")
        if not 0.0 < self.noise_floor_alpha <= 1.0:
            raise ValueError("noise_floor_alpha must be in (0, 1]")
        if not 0.0 <= self.spectral_floor <= 1.0:
            raise ValueError("spectral_floor must be between 0.0 and 1.0")
        if not 0.0 <= self.temporal_smoothing < 1.0:
            raise ValueError("temporal_smoothing must be between 0.0 and below 1.0")
        if not 0.0 <= self.output_gain <= 2.0:
            raise ValueError("output_gain must be between 0.0 and 2.0")


@dataclass(frozen=True)
class SuppressionMetrics:
    input_rms: float
    output_rms: float
    input_peak: float
    output_peak: float
    input_snr_db: float
    output_snr_db: float
    processing_latency_ms: float
    speech_distortion_db: float
    noise_profile_ready: bool


def validate_audio(audio: object) -> np.ndarray:
    """Return mono/stereo float64 audio or reject invalid input."""
    samples = np.asarray(audio)
    if samples.size == 0:
        raise ValueError("audio block must not be empty")
    if samples.ndim not in (1, 2):
        raise ValueError("audio must be mono (N,) or stereo (N, C)")
    if samples.ndim == 2 and samples.shape[1] not in (1, 2):
        raise ValueError("only mono or two-channel audio is supported")
    if not np.issubdtype(samples.dtype, np.floating):
        raise TypeError("audio must contain floating-point samples")
    if not np.all(np.isfinite(samples)):
        raise ValueError("audio must contain only finite samples")
    return samples.astype(np.float64)


def calculate_rms_db(
    signal_samples: np.ndarray,
) -> float:
    rms = float(np.sqrt(np.mean(np.square(signal_samples, dtype=np.float64))))
    return float(
        20.0 * np.log10(max(rms, np.finfo(np.float64).tiny))
    )


def estimate_snr_db(
    signal_samples: np.ndarray,
    noise_samples: np.ndarray,
) -> float:
    """Estimate broadband segment SNR from independent signal and noise arrays."""
    signal_rms = float(
        np.sqrt(np.mean(np.square(signal_samples, dtype=np.float64)))
    )
    noise_rms = float(
        np.sqrt(np.mean(np.square(noise_samples, dtype=np.float64)))
    )
    return float(
        20.0 * np.log10(
            max(signal_rms, np.finfo(np.float64).tiny)
            / max(noise_rms, np.finfo(np.float64).tiny)
        )
    )


def measure_speech_distortion_db(
    original: np.ndarray,
    processed: np.ndarray,
) -> float:
    """Measure broadband log-energy difference as a simple distortion indicator."""
    original_flat = np.asarray(original).reshape(-1)
    processed_flat = np.asarray(processed).reshape(-1)
    if original_flat.size != processed_flat.size:
        count = min(original_flat.size, processed_flat.size)
        original_flat = original_flat[:count]
        processed_flat = processed_flat[:count]
    difference_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    original_flat - processed_flat,
                    dtype=np.float64,
                )
            )
        )
    )
    original_rms = float(
        np.sqrt(np.mean(np.square(original_flat, dtype=np.float64)))
    )
    return float(
        20.0 * np.log10(
            max(difference_rms, np.finfo(np.float64).tiny)
            / max(original_rms, np.finfo(np.float64).tiny)
        )
    )


class MicrophoneNoiseSuppressor:
    """Causal stateful spectral microphone-noise suppression."""

    def __init__(self, config: NoiseSuppressionConfig | None = None) -> None:
        self.config = config or NoiseSuppressionConfig()
        self._channels: int | None = None
        self._noise_profile: np.ndarray | None = None
        self._previous_mask: np.ndarray | None = None
        self._previous_output: np.ndarray | None = None
        self._previous_input: np.ndarray | None = None
        self._calibration_samples: list[np.ndarray] = []
        self._calibration_complete = False
        self._high_pass_filters: dict[int, object] = {}
        self._last_latency_ms = 0.0
        self.reset()

    @property
    def noise_profile_ready(self) -> bool:
        return self._noise_profile is not None

    @property
    def processing_latency_ms(self) -> float:
        return self._last_latency_ms

    def reset(self) -> None:
        """Clear learned noise profile and causal spectral state."""
        self._channels = None
        self._noise_profile = None
        self._previous_mask = None
        self._previous_output = None
        self._previous_input = None
        self._calibration_samples = []
        self._calibration_complete = False
        self._high_pass_filters = {}
        self._last_latency_ms = 0.0

    def set_bypass(self, bypass: bool) -> None:
        if not isinstance(bypass, bool):
            raise TypeError("bypass must be a boolean")
        self.config = replace(self.config, bypass=bypass)

    def set_suppression_strength(self, strength: float) -> None:
        if not 0.0 <= strength <= 2.0:
            raise ValueError("strength must be between 0.0 and 2.0")
        self.config = replace(self.config, suppression_strength=strength)

    def calibrate_noise_profile(
        self,
        calibration_audio: np.ndarray,
    ) -> np.ndarray:
        """Learn a noise spectrum from supplied noise-only audio."""
        samples = validate_audio(calibration_audio)
        if samples.ndim == 2:
            mono = np.mean(samples, axis=1)
        else:
            mono = samples
        self._fit_noise_profile(mono)
        self._calibration_complete = True
        return self._noise_profile.copy()

    def collect_noise_sample(self, audio: np.ndarray) -> None:
        """Collect short noise-only samples during calibration."""
        samples = validate_audio(audio)
        mono = np.mean(samples, axis=1) if samples.ndim == 2 else samples
        if len(self._calibration_samples) >= 128:
            self._calibration_samples.pop(0)
        self._calibration_samples.append(np.array(mono, copy=True))

    def finish_noise_calibration(self) -> np.ndarray:
        """Estimate noise profile from collected noise-only blocks."""
        if not self._calibration_samples:
            raise RuntimeError("no noise samples have been collected")
        combined = np.concatenate(self._calibration_samples)
        self._fit_noise_profile(combined)
        self._calibration_complete = True
        self._calibration_samples = []
        return self._noise_profile.copy()

    def cancel_noise_calibration(self) -> None:
        self._calibration_samples = []
        self._calibration_complete = False

    def _fit_noise_profile(self, mono: np.ndarray) -> None:
        fft_size = self.config.fft_size
        if mono.size < fft_size:
            mono = np.pad(mono, (fft_size - mono.size, 0), mode="constant")

        frame_count = 1 + max(0, (mono.size - fft_size) // self.config.hop_size)
        spectra: list[np.ndarray] = []
        window = np.hanning(fft_size).astype(np.float64)
        for start in range(0, mono.size - fft_size + 1, self.config.hop_size):
            frame = mono[start:start + fft_size] * window
            spectra.append(np.abs(np.fft.rfft(frame)) ** 2)

        if spectra:
            profile = np.median(np.vstack(spectra), axis=0)
        else:
            profile = np.abs(np.fft.rfft(mono[-fft_size:] * window)) ** 2
        self._noise_profile = np.maximum(profile, np.finfo(np.float64).tiny)
        self._channels = None
        self._previous_mask = None

    def _high_pass(self, audio: np.ndarray, channel: int) -> np.ndarray:
        high_pass = self._high_pass_filters.get(channel)
        if high_pass is None:
            high_pass = highpass(
                self.config.sample_rate,
                self.config.high_pass_hz,
                self.config.high_pass_order,
            )
            self._high_pass_filters[channel] = high_pass
        return high_pass.process(audio)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process one mono/stereo block causally and maintain filter state."""
        start_ns = time.monotonic_ns()
        samples = validate_audio(audio)
        if self.config.bypass:
            return samples.astype(np.float32)

        channel_count = 1 if samples.ndim == 1 else samples.shape[1]
        if channel_count != self._channels:
            self._channels = channel_count
            fft_bins = self.config.fft_size // 2 + 1
            self._previous_mask = np.ones(
                (channel_count, fft_bins),
                dtype=np.float64,
            )
            self._previous_output = np.zeros(
                (channel_count, self.config.hop_size),
                dtype=np.float32,
            )
            self._previous_input = np.zeros(
                (channel_count, self.config.fft_size),
                dtype=np.float32,
            )

        output = np.empty(samples.shape, dtype=np.float32)
        if samples.ndim == 1:
            output[:] = self._process_channel(samples.astype(np.float32), 0)
        else:
            for channel in range(channel_count):
                output[:, channel] = self._process_channel(
                    samples[:, channel].astype(np.float32),
                    channel,
                )

        output *= np.float32(self.config.output_gain)
        self._last_latency_ms = (
            time.monotonic_ns() - start_ns
        ) / 1_000_000.0
        return output

    def _process_channel(
        self,
        input_samples: np.ndarray,
        channel: int,
    ) -> np.ndarray:
        filtered = self._high_pass(input_samples, channel)

        fft_size = self.config.fft_size
        if filtered.size >= fft_size:
            padded = filtered[-fft_size:].astype(np.float64)
        else:
            padded = np.zeros(fft_size, dtype=np.float64)
            padded[-filtered.size:] = filtered

        window = np.hanning(fft_size).astype(np.float64)
        frame = padded * window
        spectrum = np.fft.rfft(frame)
        magnitude_squared = np.abs(spectrum) ** 2

        if self._noise_profile is not None:
            profile = self._noise_profile
            if profile.size != magnitude_squared.size:
                profile = np.interp(
                    np.arange(magnitude_squared.size),
                    np.linspace(0, magnitude_squared.size - 1, profile.size),
                    profile,
                )

            strength = self.config.suppression_strength
            residual_ratio = np.maximum(
                1.0 - strength * profile / np.maximum(magnitude_squared, np.finfo(np.float64).tiny),
                self.config.spectral_floor,
            )
            posterior_snr = np.maximum(
                magnitude_squared / np.maximum(profile, np.finfo(np.float64).tiny) - 1.0,
                0.0,
            )
            wiener_mask = posterior_snr / (posterior_snr + 1.0)
            wiener_mask = self.config.spectral_floor + (
                1.0 - self.config.spectral_floor
            ) * wiener_mask
            requested_mask = (
                residual_ratio * strength
                + wiener_mask * (1.0 - strength / 2.0)
            )
            requested_mask = np.clip(
                requested_mask,
                self.config.spectral_floor,
                1.0,
            )
        else:
            requested_mask = np.ones(magnitude_squared.size, dtype=np.float64)

        previous = self._previous_mask[channel]
        if previous.size != requested_mask.size:
            previous = np.ones(requested_mask.size, dtype=np.float64)
        smoothed_mask = (
            self.config.temporal_smoothing * previous
            + (1.0 - self.config.temporal_smoothing) * requested_mask
        )
        self._previous_mask[channel] = smoothed_mask

        filtered_spectrum = spectrum * smoothed_mask
        # A causal causal frame uses only its own and previous samples.
        synthesized = np.fft.irfft(filtered_spectrum, n=fft_size)
        synthesized *= window

        output = synthesized[-input_samples.size:] if input_samples.size < fft_size else synthesized[:input_samples.size]

        if output.size != input_samples.size:
            if output.size > input_samples.size:
                output = output[:input_samples.size]
            else:
                output = np.pad(output, (0, input_samples.size - output.size), mode="edge")

        return output.astype(np.float32)


def estimate_input_output_snr(
    clean: np.ndarray,
    noisy: np.ndarray,
    processed: np.ndarray,
) -> tuple[float, float]:
    """Estimate input and output SNR using known clean and additive noise."""
    clean = np.asarray(clean).reshape(-1)
    noisy = np.asarray(noisy).reshape(-1)
    processed = np.asarray(processed).reshape(-1)
    count = min(clean.size, noisy.size, processed.size)
    clean = clean[:count]
    noisy = noisy[:count]
    processed = processed[:count]
    noise = noisy - clean
    residual_noise = processed - clean
    input_snr = estimate_snr_db(clean, noise)
    output_snr = estimate_snr_db(clean, residual_noise)
    return input_snr, output_snr


def generate_demo_signals(
    sample_rate: int = 48_000,
    duration_seconds: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate clean speech-like, noise-only and noisy demonstration signals."""
    frame_count = int(sample_rate * duration_seconds)
    time_axis = np.arange(frame_count, dtype=np.float64) / sample_rate
    envelope = 0.35 + 0.65 * np.maximum(
        0.0,
        np.sin(2.0 * np.pi * 3.0 * time_axis),
    )
    clean = envelope * (
        0.35 * np.sin(2.0 * np.pi * 180.0 * time_axis)
        + 0.20 * np.sin(2.0 * np.pi * 320.0 * time_axis)
        + 0.12 * np.sin(2.0 * np.pi * 640.0 * time_axis)
    )
    rng = np.random.default_rng(2026)
    noise = (
        0.015 * rng.normal(size=frame_count)
        + 0.08 * np.sin(2.0 * np.pi * 100.0 * time_axis)
    )
    noisy = clean + noise
    return (
        time_axis,
        clean.astype(np.float32),
        noisy.astype(np.float32),
    )


def run_suppression_plot() -> None:
    """Plot original noisy and processed speech-like signals."""
    import matplotlib.pyplot as plt

    time_axis, clean, noisy = generate_demo_signals()
    suppressor = MicrophoneNoiseSuppressor(
        NoiseSuppressionConfig(
            sample_rate=48000,
            fft_size=512,
            hop_size=128,
            suppression_strength=1.0,
        )
    )
    suppressor.calibrate_noise_profile(
        (0.015 * np.random.default_rng(2026).normal(size=noisy.size)
         + 0.08 * np.sin(2.0 * np.pi * 100.0 * time_axis)).astype(np.float32)
    )

    processed_blocks = [
        suppressor.process(noisy[start:start + 512])
        for start in range(0, len(noisy), 512)
    ]
    processed = np.concatenate(processed_blocks)[:len(noisy)]

    input_snr, output_snr = estimate_input_output_snr(
        clean,
        noisy,
        processed,
    )
    distortion = measure_speech_distortion_db(clean, processed)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time_axis, noisy, lw=0.7)
    axes[0].set_title("Original noisy microphone signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.2)
    axes[1].plot(time_axis, processed, lw=0.7)
    axes[1].set_title(
        f"Processed microphone signal | input SNR {input_snr:.1f} dB "
        f"| output SNR {output_snr:.1f} dB "
        f"| distortion {distortion:.1f} dB"
    )
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot software microphone noise suppression demonstration"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Suppression strength from 0.0 to 2.0",
    )
    parser.add_argument("--bypass", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    if not 0.0 <= args.strength <= 2.0:
        parser.error("strength must be between 0.0 and 2.0")
    if args.bypass:
        LOGGER.info("Bypass mode requested")
    run_suppression_plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
