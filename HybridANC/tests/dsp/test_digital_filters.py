"""Comprehensive tests for causal streaming digital filters."""

import numpy as np
import pytest
from scipy import signal

from prototype.python.digital_filters import (
    FilterType,
    StreamingFilterConfig,
    StreamingDigitalFilter,
    bandpass,
    highpass,
    lowpass,
    notch,
)


def sine(freq_hz, sample_rate=48000, seconds=1.0):
    time = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
    return np.sin(2 * np.pi * freq_hz * time)


def steady_rms(audio, fraction=0.75):
    start = int(len(audio) * (1.0 - fraction))
    return float(np.sqrt(np.mean(np.square(audio[start:]))))


def test_lowpass_removes_high_frequency_tone():
    sample_rate = 48000
    original = sine(100, sample_rate) + sine(1000, sample_rate)
    filtered = lowpass(sample_rate, 300, order=4).process(original.astype(np.float32))

    assert steady_rms(filtered) < steady_rms(original) * 0.75


def test_highpass_removes_low_frequency_tone():
    sample_rate = 48000
    original = sine(100, sample_rate) + sine(1000, sample_rate)
    filtered = highpass(sample_rate, 300, order=4).process(original.astype(np.float32))

    assert steady_rms(filtered) > steady_rms(original) * 0.25
    assert np.max(np.abs(filtered - sine(1000, sample_rate).astype(np.float32))) < 1.0


def test_bandpass_preserves_1khz_tone():
    sample_rate = 48000
    original = sine(100, sample_rate) + sine(1000, sample_rate) + sine(5000, sample_rate)
    filtered = bandpass(sample_rate, 800, 1200, order=4).process(
        original.astype(np.float32)
    )

    assert steady_rms(filtered) > 0.2
    assert steady_rms(filtered) < 0.8


def test_notch_removes_50hz_interference():
    sample_rate = 48000
    original = sine(50, sample_rate) + sine(1000, sample_rate)
    filtered = notch(sample_rate, 50, quality_factor=30).process(
        original.astype(np.float32)
    )

    filtered_50 = notch(sample_rate, 50, quality_factor=30).process(
        sine(50, sample_rate).astype(np.float32)
    )
    assert steady_rms(filtered_50) < 0.10


def test_streaming_blocks_match_zero_state_full_filter():
    sample_rate = 48000
    audio = sine(1000, sample_rate)
    filter_object = lowpass(sample_rate, 500, order=4)

    chunks = [
        filter_object.process(audio[start:start + 512].astype(np.float32))
        for start in range(0, len(audio), 512)
    ]
    streamed = np.concatenate(chunks)
    filter_object.reset()
    expected = signal.sosfilt(filter_object.sos, audio)

    assert streamed.shape == audio.shape
    assert np.max(np.abs(streamed - expected)) < 0.001


def test_reset_clears_filter_state():
    audio = sine(1000, 48000)
    filter_object = lowpass(48000, 500)
    first = filter_object.process(audio[:4096].astype(np.float32))
    filter_object.reset()
    reset_output = filter_object.process(audio[:4096].astype(np.float32))

    assert np.allclose(first, reset_output)


def test_mono_and_stereo():
    mono = np.ones(4096, dtype=np.float32)
    stereo = np.column_stack((mono, mono * 0.5))
    filter_object = lowpass(48000, 1000)

    assert filter_object.process(mono).shape == mono.shape
    assert filter_object.process(stereo).shape == stereo.shape


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cutoff_hz": 24000},
        {"cutoff_hz": 0},
        {"cutoff_hz": -100},
        {"cutoff_hz": np.nan},
    ],
)
def test_cutoff_is_validated_against_nyquist(kwargs):
    with pytest.raises(ValueError):
        lowpass(48000, kwargs["cutoff_hz"])


def test_bandpass_cutoffs_must_be_ordered():
    with pytest.raises(ValueError):
        bandpass(48000, 3000, 1000)


def test_invalid_order_and_input_are_rejected():
    with pytest.raises(ValueError):
        StreamingFilterConfig(
            sample_rate=48000,
            filter_type=FilterType.LOWPASS,
            cutoff_hz=1000,
            order=21,
        )

    filter_object = lowpass(48000, 1000)
    with pytest.raises(ValueError):
        filter_object.process(np.array([], dtype=np.float32))
    with pytest.raises(ValueError):
        filter_object.process(np.full((32, 3), 0.1, dtype=np.float32))
    with pytest.raises(TypeError):
        filter_object.process(np.ones(32, dtype=np.float64).astype(np.int16))
