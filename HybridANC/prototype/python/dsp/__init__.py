"""Foundational NumPy/SciPy digital signal processing helpers."""

from .signal_metrics import peak_amplitude, rms, rms_dbfs, zero_crossing_rate
from .normalization import normalize
from .moving_average import moving_average
from .filters import apply_sos_filter, butter_lowpass_sos
from .windows import hann_window, hamming_window, blackman_window
from .fft import rfft, rfft_magnitude
from .frequency_analysis import frequency_peaks, dominant_frequencies

__all__ = [
    "normalize",
    "rms",
    "rms_dbfs",
    "peak_amplitude",
    "zero_crossing_rate",
    "moving_average",
    "apply_sos_filter",
    "butter_lowpass_sos",
    "hann_window",
    "hamming_window",
    "blackman_window",
    "rfft",
    "rfft_magnitude",
    "frequency_peaks",
    "dominant_frequencies",
]
