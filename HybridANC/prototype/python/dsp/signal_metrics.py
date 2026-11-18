"""Time-domain RMS, peak and zero-crossing measurements."""

from __future__ import annotations

import numpy as np

from .validation import as_float_audio


def rms(signal: object) -> np.ndarray | float:
    """Calculate RMS amplitude per channel.

    RMS = sqrt(mean(x[n]^2)); silence is 0 and unit sine amplitude has 1/sqrt(2).
    """
    audio = as_float_audio(signal)
    if audio.ndim == 1:
        return float(np.sqrt(np.mean(np.square(audio))))
    return np.sqrt(np.mean(np.square(audio), axis=0))


def rms_dbfs(signal: object, reference: float = 1.0) -> np.ndarray | float:
    """Return RMS level relative to ``reference`` as dBFS."""
    if reference <= 0.0:
        raise ValueError("reference must be positive")
    level = rms(signal)
    if np.isscalar(level):
        return float(20.0 * np.log10(max(float(level), np.finfo(np.float64).tiny) / reference))
    level_array = np.asarray(level)
    safe = np.maximum(level_array, np.finfo(np.float64).tiny)
    return 20.0 * np.log10(safe / reference)


def peak_amplitude(signal: object) -> np.ndarray | float:
    """Calculate maximum absolute sample amplitude per channel."""
    audio = as_float_audio(signal)
    if audio.ndim == 1:
        return float(np.max(np.abs(audio)))
    return np.max(np.abs(audio), axis=0)


def zero_crossing_rate(signal: object) -> np.ndarray | float:
    """Estimate sign changes divided by the number of adjacent sample pairs."""
    audio = as_float_audio(signal)
    if audio.ndim == 1:
        if audio.size < 2:
            return 0.0
        return float(np.mean(np.signbit(audio[1:]) != np.signbit(audio[:-1])))

    if audio.shape[0] < 2:
        return np.zeros(audio.shape[1], dtype=np.float64)
    crossings = np.signbit(audio[1:, :]) != np.signbit(audio[:-1, :])
    return np.mean(crossings, axis=0)
