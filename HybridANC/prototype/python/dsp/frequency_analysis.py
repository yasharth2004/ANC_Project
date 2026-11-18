"""Frequency-domain peak and dominant-frequency analysis."""

from __future__ import annotations

import numpy as np

from .validation import as_float_audio


def frequency_axis(sample_rate: float, fft_size: int) -> np.ndarray:
    """Return frequencies in Hz for an FFT of ``fft_size`` real samples."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not isinstance(fft_size, int) or fft_size <= 0:
        raise ValueError("fft_size must be positive")
    return np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)


def frequency_peaks(
    signal: object,
    sample_rate: float,
    *,
    nfft: int | None = None,
    minimum_bin: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return magnitude spectrum and frequency axis for mono/stereo audio."""
    audio = as_float_audio(signal)
    fft_size = len(audio) if nfft is None else nfft
    if fft_size <= 0:
        raise ValueError("nfft must be positive")
    frequencies = frequency_axis(sample_rate, fft_size)
    spectrum = np.abs(np.fft.rfft(audio, n=nfft, axis=0))
    if minimum_bin > 0:
        spectrum = spectrum.copy()
        spectrum[:minimum_bin] = 0.0
    return frequencies, spectrum


def dominant_frequencies(
    signal: object,
    sample_rate: float,
    *,
    nfft: int | None = None,
) -> np.ndarray | float:
    """Return peak spectral frequency for mono or every stereo channel."""
    frequencies, spectrum = frequency_peaks(
        signal,
        sample_rate,
        nfft=nfft,
    )
    if spectrum.ndim == 1:
        return float(frequencies[int(np.argmax(spectrum))])
    peaks = np.argmax(spectrum, axis=0)
    return frequencies[peaks]
