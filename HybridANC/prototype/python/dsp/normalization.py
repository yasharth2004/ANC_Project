"""Amplitude normalization for floating-point mono and stereo audio."""

from __future__ import annotations

import numpy as np

from .validation import as_float_audio


def normalize(
    signal: object,
    target_peak: float = 1.0,
    *,
    per_channel: bool = True,
) -> np.ndarray:
    """Scale samples so each selected channel reaches ``target_peak``."""
    audio = as_float_audio(signal)
    if not 0.0 < target_peak <= 1.0:
        raise ValueError("target_peak must be greater than 0 and at most 1")

    result = np.array(audio, dtype=np.float64, copy=True)
    if result.ndim == 1:
        peak = float(np.max(np.abs(result)))
        if peak > 0.0:
            result *= target_peak / peak
        return result

    if per_channel:
        peaks = np.max(np.abs(result), axis=0)
        for channel, peak in enumerate(peaks):
            if peak > 0.0:
                result[:, channel] *= target_peak / peak
    else:
        peak = float(np.max(np.abs(result)))
        if peak > 0.0:
            result *= target_peak / peak
    return result
