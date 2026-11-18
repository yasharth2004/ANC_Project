"""Unit tests for foundational HybridANC DSP operations."""

import numpy as np
import pytest

from prototype.python.dsp.normalization import normalize
from prototype.python.dsp.signal_metrics import (
    peak_amplitude,
    rms,
    rms_dbfs,
    zero_crossing_rate,
)
from prototype.python.dsp.moving_average import moving_average
from prototype.python.dsp.filters import apply_sos_filter, butter_lowpass_sos
from prototype.python.dsp.windows import (
    hann_window,
    hamming_window,
    blackman_window,
)
from prototype.python.dsp.fft import rfft_magnitude
from prototype.python.dsp.frequency_analysis import (
    dominant_frequencies,
    frequency_axis,
)
from prototype.python.dsp.validation import DSPInputError
from prototype.python.dsp.demo import generate_sine


@pytest.mark.parametrize("bad", [[], np.array([]), np.zeros((2, 2, 2))])
def test_empty_invalid_shape_is_rejected(bad):
    with pytest.raises(DSPInputError):
        rms(bad)


def test_nonfinite_signal_is_rejected():
    with pytest.raises(DSPInputError):
        normalize([1.0, np.nan])


def test_normalize_mono_and_stereo():
    mono = normalize([0.1, -0.5, 0.25])
    stereo = normalize(np.array([[0.2, 0.4], [0.1, -0.8]]))

    assert np.max(np.abs(mono)) == pytest.approx(1.0)
    assert np.max(np.abs(stereo), axis=0) == pytest.approx([1.0, 1.0])


def test_rms_peak_and_dbfs():
    sine = generate_sine(1000.0, 48000.0, 0.1)

    assert rms(sine) == pytest.approx(1.0 / np.sqrt(8.0), rel=1e-3)
    assert peak_amplitude(sine) == pytest.approx(0.5)
    assert rms_dbfs(sine) < 0.0


def test_rms_peak_and_zcr_support_stereo():
    stereo = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]])

    assert rms(stereo).shape == (2,)
    assert peak_amplitude(stereo).shape == (2,)
    assert zero_crossing_rate(stereo).shape == (2,)
    assert np.all(zero_crossing_rate(stereo) == 1.0)


def test_moving_average_mono_and_stereo():
    mono = moving_average([1.0, 2.0, 3.0], 3)
    stereo = moving_average([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], 2)

    assert mono.shape == (3,)
    assert mono == pytest.approx([1.0, 1.5, 2.0])
    assert stereo.shape == (3, 2)


def test_butterworth_filter_and_response():
    sample_rate = 48000
    sos = butter_lowpass_sos(sample_rate, 1000)
    sine = generate_sine(100.0, sample_rate, 0.1)

    filtered = apply_sos_filter(sos, sine)
    assert filtered.shape == sine.shape
    assert np.max(np.abs(filtered)) <= 1.0


def test_window_functions_are_normalized_shapes():
    for window in (hann_window, hamming_window, blackman_window):
        values = window(64)
        assert values.shape == (64,)
        assert np.all(values >= 0.0)
        assert np.max(values) > 0.0


def test_fft_and_frequency_analysis_find_sine():
    sample_rate = 48000
    sine = generate_sine(1000.0, sample_rate, 0.1)
    frequencies = frequency_axis(sample_rate, len(sine))

    assert frequencies[-1] > 20_000
    assert rfft_magnitude(sine).shape == (2401,)
    assert dominant_frequencies(sine, sample_rate) == pytest.approx(
        1000.0,
        abs=1.0,
    )


def test_invalid_filter_cutoff_is_rejected():
    with pytest.raises(ValueError):
        butter_lowpass_sos(48000, 24000)


def test_generated_sine_obeys_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        generate_sine(24000.0, 48000.0, 0.1)
