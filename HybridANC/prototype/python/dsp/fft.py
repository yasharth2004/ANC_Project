"""Real-valued FFT helpers."""

from __future__ import annotations

import numpy as np

from .validation import as_float_audio


def rfft(
    signal: object,
    nfft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided FFT bins and complex spectrum for mono/stereo audio."""
    audio = as_float_audio(signal)
    if nfft is not None:
        if not isinstance(nfft, int) or isinstance(nfft, bool):
            raise TypeError("nfft must be an integer")
        if nfft <= 0:
            raise ValueError("nfft must be positive")
    spectrum = np.fft.rfft(audio, n=nfft, axis=0)
    return spectrum, np.arange(spectrum.shape[0], dtype=np.float64)


def rfft_magnitude(
    signal: object,
    nfft: int | None = None,
) -> np.ndarray:
    """Return absolute one-sided FFT magnitude."""
    spectrum, _ = rfft(signal, nfft)
    return np.abs(spectrum)
