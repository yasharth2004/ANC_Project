"""Causal streaming digital filters for HybridANC.

These filters use second-order sections (SOS) and preserve state between calls.
They are suitable for real-time mono/stereo blocks.  Offline ``filtfilt``/
zero-phase filtering is intentionally not used because it is non-causal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import argparse
import logging
from typing import Iterable

import numpy as np
from scipy import signal


LOGGER = logging.getLogger("hybridanc.digital_filters")


class FilterType(str, Enum):
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    NOTCH = "notch"


@dataclass(frozen=True)
class StreamingFilterConfig:
    sample_rate: float
    filter_type: FilterType
    low_cutoff_hz: float | None = None
    high_cutoff_hz: float | None = None
    cutoff_hz: float | None = None
    order: int = 4
    quality_factor: float = 30.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a finite positive value")
        if isinstance(self.filter_type, str):
            object.__setattr__(
                self,
                "filter_type",
                FilterType(self.filter_type),
            )
        if not isinstance(self.order, int) or isinstance(self.order, bool):
            raise TypeError("order must be an integer")
        if not 1 <= self.order <= 20:
            raise ValueError("order must be between 1 and 20")
        if not np.isfinite(self.quality_factor) or self.quality_factor <= 0:
            raise ValueError("quality_factor must be finite and positive")


class StreamingDigitalFilter:
    """Causal SOS filter that retains state for consecutive audio blocks."""

    def __init__(self, config: StreamingFilterConfig) -> None:
        self.config = config
        self.sos = design_filter(config)
        self._channels: int | None = None
        self._state: np.ndarray | None = None
        self.reset()

    @property
    def channels(self) -> int | None:
        return self._channels

    def reset(self) -> None:
        """Reset filter memory.  A pending block may produce a short transient."""
        self._channels = None
        self._state = np.zeros(
            (self.sos.shape[0], 2, 1),
            dtype=np.float64,
        )

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Filter mono ``(N,)`` or stereo ``(N, C)`` audio without phase reversal."""
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

        is_mono = samples.ndim == 1
        channel_count = 1 if is_mono else samples.shape[1]
        if self._channels != channel_count:
            self._state = np.repeat(
                np.zeros(
                    (self.sos.shape[0], 2, 1),
                    dtype=np.float64,
                ),
                channel_count,
                axis=2,
            ).astype(np.float64)
            self._channels = channel_count

        dtype = np.result_type(samples.dtype, np.float32)
        output = np.empty(
            samples.shape,
            dtype=dtype,
        )

        if is_mono:
            filtered, state = signal.sosfilt(
                self.sos,
                samples,
                zi=self._state[:, :, 0],
            )
            output[:] = filtered
            self._state[:, :, 0] = state
        else:
            for channel in range(channel_count):
                filtered, state = signal.sosfilt(
                    self.sos,
                    samples[:, channel],
                    zi=self._state[:, :, channel],
                )
                output[:, channel] = filtered
                self._state[:, :, channel] = state

        if not np.all(np.isfinite(output)):
            self.reset()
            raise RuntimeError("filter state produced non-finite output")
        return output


def _validate_cutoff(
    sample_rate: float,
    cutoff_hz: float,
    name: str = "cutoff",
) -> None:
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError(f"{name} must be a finite positive frequency")
    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"{name} must be below Nyquist frequency ({nyquist:.3f} Hz)"
        )


def _validate_stability(sos: np.ndarray) -> None:
    if sos.shape[1] != 6 or not np.all(np.isfinite(sos)):
        raise ValueError("filter coefficients are invalid")
    poles = np.concatenate(
        [
            np.roots([section[3], section[4], section[5]])
            for section in sos
        ]
    )
    if np.any(np.abs(poles) >= 1.0):
        raise ValueError("filter configuration is unstable")


def design_filter(config: StreamingFilterConfig) -> np.ndarray:
    """Design a stable SOS representation for a streaming filter."""
    sample_rate = config.sample_rate
    filter_type = config.filter_type
    nyquist = sample_rate / 2.0

    if filter_type in (FilterType.LOWPASS, FilterType.HIGHPASS):
        if config.cutoff_hz is None:
            raise ValueError("cutoff_hz is required for low-pass and high-pass filters")
        _validate_cutoff(sample_rate, config.cutoff_hz)
        sos = signal.butter(
            config.order,
            config.cutoff_hz,
            btype=(
                "lowpass"
                if filter_type == FilterType.LOWPASS
                else "highpass"
            ),
            fs=sample_rate,
            output="sos",
        )
    elif filter_type == FilterType.BANDPASS:
        if config.low_cutoff_hz is None or config.high_cutoff_hz is None:
            raise ValueError(
                "low_cutoff_hz and high_cutoff_hz are required for band-pass"
            )
        _validate_cutoff(sample_rate, config.low_cutoff_hz, "low_cutoff_hz")
        _validate_cutoff(sample_rate, config.high_cutoff_hz, "high_cutoff_hz")
        if config.low_cutoff_hz >= config.high_cutoff_hz:
            raise ValueError("low_cutoff_hz must be below high_cutoff_hz")
        sos = signal.butter(
            config.order,
            [config.low_cutoff_hz, config.high_cutoff_hz],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )
    elif filter_type == FilterType.NOTCH:
        if config.cutoff_hz is None:
            raise ValueError("cutoff_hz is required for notch filtering")
        _validate_cutoff(sample_rate, config.cutoff_hz)
        coefficients = signal.iirnotch(
            config.cutoff_hz,
            config.quality_factor,
            fs=sample_rate,
        )
        sos = signal.tf2sos(*coefficients)
    else:
        raise ValueError(f"unsupported filter type: {filter_type}")

    _validate_stability(sos)
    return sos


def lowpass(
    sample_rate: float,
    cutoff_hz: float,
    order: int = 4,
) -> StreamingDigitalFilter:
    return StreamingDigitalFilter(
        StreamingFilterConfig(
            sample_rate=sample_rate,
            filter_type=FilterType.LOWPASS,
            cutoff_hz=cutoff_hz,
            order=order,
        )
    )


def highpass(
    sample_rate: float,
    cutoff_hz: float,
    order: int = 4,
) -> StreamingDigitalFilter:
    return StreamingDigitalFilter(
        StreamingFilterConfig(
            sample_rate=sample_rate,
            filter_type=FilterType.HIGHPASS,
            cutoff_hz=cutoff_hz,
            order=order,
        )
    )


def bandpass(
    sample_rate: float,
    low_cutoff_hz: float,
    high_cutoff_hz: float,
    order: int = 4,
) -> StreamingDigitalFilter:
    return StreamingDigitalFilter(
        StreamingFilterConfig(
            sample_rate=sample_rate,
            filter_type=FilterType.BANDPASS,
            low_cutoff_hz=low_cutoff_hz,
            high_cutoff_hz=high_cutoff_hz,
            order=order,
        )
    )


def notch(
    sample_rate: float,
    cutoff_hz: float,
    quality_factor: float = 30.0,
) -> StreamingDigitalFilter:
    return StreamingDigitalFilter(
        StreamingFilterConfig(
            sample_rate=sample_rate,
            filter_type=FilterType.NOTCH,
            cutoff_hz=cutoff_hz,
            quality_factor=quality_factor,
            order=2,
        )
    )


def generate_filter_examples(
    sample_rate: int = 48_000,
    duration_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a composite example and a two-second time axis."""
    frame_count = int(sample_rate * duration_seconds)
    time_axis = np.arange(frame_count, dtype=np.float64) / sample_rate
    signal_50 = 0.15 * np.sin(2 * np.pi * 50 * time_axis)
    signal_100 = 0.20 * np.sin(2 * np.pi * 100 * time_axis)
    signal_1000 = 0.40 * np.sin(2 * np.pi * 1000 * time_axis)
    rng = np.random.default_rng(2026)
    high_noise = 0.08 * rng.normal(size=frame_count)
    composite = signal_50 + signal_100 + signal_1000 + high_noise
    return time_axis, composite


def run_filter_plot(
    filter_type: FilterType,
    sample_rate: int = 48_000,
    order: int = 4,
) -> None:
    """Plot original and causal filtered example signals."""
    import matplotlib.pyplot as plt

    time_axis, original = generate_filter_examples(sample_rate)
    if filter_type == FilterType.LOWPASS:
        filter_object = lowpass(sample_rate, 500.0, order)
    elif filter_type == FilterType.HIGHPASS:
        filter_object = highpass(sample_rate, 500.0, order)
    elif filter_type == FilterType.BANDPASS:
        filter_object = bandpass(sample_rate, 800.0, 1200.0, order)
    else:
        filter_object = notch(sample_rate, 50.0, quality_factor=30.0)

    filtered = filter_object.process(original.astype(np.float32))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time_axis, original, lw=0.7)
    axes[0].set_title("Original composite signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.2)
    axes[1].plot(time_axis, filtered, lw=0.7)
    axes[1].set_title(f"Causally filtered signal: {filter_type.value}")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot a generated signal with a causal HybridANC filter"
    )
    parser.add_argument(
        "--type",
        choices=[item.value for item in FilterType],
        default=FilterType.LOWPASS.value,
    )
    parser.add_argument("--sample-rate", type=int, choices=[44100, 48000], default=48000)
    parser.add_argument("--order", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    run_filter_plot(
        FilterType(args.type),
        sample_rate=args.sample_rate,
        order=args.order,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
