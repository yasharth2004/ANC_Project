"""Finite moving-average filtering."""

from __future__ import annotations

import numpy as np

from .validation import as_float_audio


def moving_average(signal: object, window_size: int) -> np.ndarray:
    """Calculate a causal trailing moving average along the time axis."""
    audio = as_float_audio(signal)
    if not isinstance(window_size, int) or isinstance(window_size, bool):
        raise TypeError("window_size must be an integer")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size == 1:
        return np.array(audio, copy=True)

    padding = ((window_size - 1, 0), (0, 0)) if audio.ndim == 2 else (window_size - 1, 0)
    padded = np.pad(audio, padding, mode="constant")
    cumulative = np.cumsum(padded, dtype=np.float64, axis=0)
    prefix_shape = (1, audio.shape[1]) if audio.ndim == 2 else (1,)
    cumulative = np.vstack((np.zeros(prefix_shape, dtype=np.float64), cumulative)) if audio.ndim == 2 else np.concatenate(
        (np.zeros(1, dtype=np.float64), cumulative)
    )
    summed = cumulative[window_size:] - cumulative[:-window_size]
    counts = np.minimum(
        np.arange(1, audio.shape[0] + 1),
        window_size,
    )
    if audio.ndim == 1:
        return summed / counts
    return summed / counts[:, np.newaxis]
