"""Common FFT window functions."""

from __future__ import annotations

import numpy as np
from scipy import signal


def _window(size: int, kind: str) -> np.ndarray:
    if not isinstance(size, int) or isinstance(size, bool):
        raise TypeError("size must be an integer")
    if size <= 0:
        raise ValueError("size must be positive")
    if size == 1:
        return np.ones(1, dtype=np.float64)
    window = getattr(signal.windows, kind)(size, sym=False).astype(np.float64)
    return np.maximum(window, 0.0)


def hann_window(size: int) -> np.ndarray:
    """Periodic Hann window; suitable for overlapping FFT frames."""
    return _window(size, "hann")


def hamming_window(size: int) -> np.ndarray:
    """Periodic Hamming window."""
    return _window(size, "hamming")


def blackman_window(size: int) -> np.ndarray:
    """Periodic Blackman window."""
    return _window(size, "blackman")
