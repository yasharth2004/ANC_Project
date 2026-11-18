"""Shared validation helpers for prototype DSP functions."""

from __future__ import annotations

import numpy as np


class DSPInputError(ValueError):
    """Raised for empty, non-floating, dimensional or non-finite input."""


def as_float_audio(signal: object, *, name: str = "signal") -> np.ndarray:
    """Convert supported mono/stereo signals to one- or two-dimensional float64."""
    audio = np.asarray(signal, dtype=np.float64)
    if audio.size == 0:
        raise DSPInputError(f"{name} must not be empty")
    if audio.ndim not in (1, 2):
        raise DSPInputError(f"{name} must be mono (N,) or stereo (N, C)")
    if audio.ndim == 2 and audio.shape[1] not in (1, 2):
        raise DSPInputError("stereo audio may contain only one or two channels")
    if not np.all(np.isfinite(audio)):
        raise DSPInputError(f"{name} must contain only finite values")
    return audio


def channel_count(signal: np.ndarray) -> int:
    return 1 if signal.ndim == 1 else signal.shape[1]
