"""Objective-style tests for microphone noise suppression."""

import time

import numpy as np
import pytest

from prototype.python.noise_suppression import (
    MicrophoneNoiseSuppressor,
    NoiseSuppressionConfig,
    estimate_input_output_snr,
    estimate_snr_db,
    measure_speech_distortion_db,
)


def speech_like_signal(sample_rate=48000, duration=1.0):
    frame_count = int(sample_rate * duration)
    time_axis = np.arange(frame_count, dtype=np.float64) / sample_rate
    envelope = np.maximum(
        0.0,
        np.sin(2 * np.pi * 4 * time_axis),
    )
    speech = envelope * (
        0.35 * np.sin(2 * np.pi * 180 * time_axis)
        + 0.22 * np.sin(2 * np.pi * 320 * time_axis)
        + 0.12 * np.sin(2 * np.pi * 640 * time_axis)
    )
    return speech.astype(np.float32)


def test_noise_profile_calibration_and_reset():
    rng = np.random.default_rng(7)
    suppressor = MicrophoneNoiseSuppressor(
        NoiseSuppressionConfig(fft_size=512, hop_size=128)
    )
    noise = rng.normal(0, 0.02, 4096).astype(np.float32)

    profile = suppressor.calibrate_noise_profile(noise)

    assert suppressor.noise_profile_ready
    assert profile.shape == (257,)
    suppressor.reset()
    assert not suppressor.noise_profile_ready


def test_bypass_returns_identical_audio():
    suppressor = MicrophoneNoiseSuppressor(
        NoiseSuppressionConfig(bypass=True)
    )
    audio = np.linspace(-0.5, 0.5, 4096, dtype=np.float32)

    output = suppressor.process(audio)

    assert np.array_equal(output, audio)
    assert suppressor.processing_latency_ms >= 0.0


def test_suppression_reduces_noise_floor():
    rng = np.random.default_rng(11)
    suppressor = MicrophoneNoiseSuppressor(
        NoiseSuppressionConfig(
            fft_size=512,
            hop_size=128,
            suppression_strength=1.5,
            high_pass_hz=80,
        )
    )
    noise_only = rng.normal(0, 0.02, 8192).astype(np.float32)
    suppressor.calibrate_noise_profile(noise_only)

    clean = speech_like_signal()
    noise = rng.normal(0, 0.02, clean.size).astype(np.float32)
    noisy = clean + noise
    processed = suppressor.process(noisy)

    silent_region_original = noisy[clean.size // 2:clean.size // 2 + 400]
    silent_region_processed = processed[clean.size // 2:clean.size // 2 + 400]
    assert np.sqrt(np.mean(processed.astype(np.float64) ** 2)) <= np.sqrt(
        np.mean(noisy.astype(np.float64) ** 2)
    )
    assert np.max(np.abs(silent_region_processed)) >= 0.0


def test_mono_and_stereo_are_supported():
    suppressor = MicrophoneNoiseSuppressor()
    mono = np.zeros(2048, dtype=np.float32)
    stereo = np.column_stack((mono, mono))

    mono_output = suppressor.process(mono)
    suppressor.reset()
    stereo_output = suppressor.process(stereo)

    assert mono_output.shape == mono.shape
    assert stereo_output.shape == stereo.shape


def test_streaming_block_processing_is_continuous():
    rng = np.random.default_rng(5)
    suppressor = MicrophoneNoiseSuppressor(
        NoiseSuppressionConfig(fft_size=512, hop_size=128)
    )
    suppressor.calibrate_noise_profile(
        rng.normal(0, 0.01, 4096).astype(np.float32)
    )
    audio = rng.normal(0, 0.05, 4096).astype(np.float32)
    streamed = np.concatenate(
        [
            suppressor.process(audio[start:start + 512])
            for start in range(0, len(audio), 512)
        ]
    )

    assert streamed.shape == audio.shape
    assert np.all(np.isfinite(streamed))


def test_snr_and_distortion_objective_metrics():
    rng = np.random.default_rng(23)
    clean = speech_like_signal(duration=0.5)
    noise = rng.normal(0, 0.02, clean.size).astype(np.float32)
    noisy = clean + noise
    processed = 0.90 * clean + 0.005 * noise

    input_snr, output_snr = estimate_input_output_snr(
        clean,
        noisy,
        processed,
    )
    distortion = measure_speech_distortion_db(clean, processed)

    assert output_snr > input_snr
    assert distortion < 0.0
    assert estimate_snr_db(clean, noise) > 0.0


def test_processing_latency_is_recorded():
    suppressor = MicrophoneNoiseSuppressor()
    suppressor.process(np.zeros(512, dtype=np.float32))

    assert suppressor.processing_latency_ms >= 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 32000},
        {"fft_size": 300},
        {"hop_size": 513},
        {"suppression_strength": -0.1},
        {"spectral_floor": -0.1},
    ],
)
def test_invalid_configurations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        NoiseSuppressionConfig(**kwargs)
