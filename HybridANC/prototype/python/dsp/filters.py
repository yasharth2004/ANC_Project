"""Digital Butterworth filtering with SciPy SOS representation."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .validation import as_float_audio


def butter_lowpass_sos(
    sample_rate: float,
    cutoff_hz: float,
    *,
    order: int = 4,
) -> np.ndarray:
    """Design a normalized Butterworth low-pass filter."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    nyquist = sample_rate / 2.0
    if not 0.0 < cutoff_hz < nyquist:
        raise ValueError("cutoff_hz must be below the Nyquist frequency")
    if order <= 0:
        raise ValueError("order must be positive")
    return signal.butter(
        order,
        cutoff_hz,
        btype="lowpass",
        fs=sample_rate,
        output="sos",
    )


def apply_sos_filter(sos: np.ndarray, audio: object) -> np.ndarray:
    """Apply a second-order-section filter independently to every channel."""
    samples = as_float_audio(audio)
    sos_array = np.asarray(sos)
    if sos_array.ndim != 2 or sos_array.shape[1] != 6:
        raise ValueError("sos must be a two-dimensional (N, 6) array")
    if not np.all(np.isfinite(sos_array)):
        raise ValueError("sos coefficients must be finite")
    return signal.sosfilt(sos_array, samples, axis=0)
