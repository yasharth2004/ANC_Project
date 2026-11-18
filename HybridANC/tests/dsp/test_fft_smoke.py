"""Basic DSP environment smoke test.

This test verifies the NumPy FFT toolchain only. It does not implement an
audio-processing algorithm or a production DSP path.
"""

import numpy as np


def test_fft_preserves_sine_frequency_peak() -> None:
    sample_rate = 48_000
    frequency = 1_000
    frame = np.arange(1_024)
    signal = np.sin(2 * np.pi * frequency * frame / sample_rate)

    magnitude = np.abs(np.fft.rfft(signal))
    peak_bin = int(np.argmax(magnitude))
    measured_frequency = peak_bin * sample_rate / len(frame)

    assert abs(measured_frequency - frequency) < sample_rate / len(frame)
