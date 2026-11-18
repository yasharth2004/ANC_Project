"""Unit tests for the offline ANC mathematical laboratory."""

import json

import numpy as np
import pytest
import soundfile as sf

from prototype.python.anc_simulation import (
    ANCMismatchConfig,
    apply_time_delay,
    attenuation_db,
    calculate_residual,
    generate_multi_frequency_noise,
    generate_sine,
    generate_white_noise,
    load_recorded_noise,
    run_cancellation_experiment,
)


def test_100hz_sine_is_generated_at_expected_frequency():
    sine = generate_sine(100.0, 48000, 1.0)
    spectrum = np.abs(np.fft.rfft(sine))
    peak_bin = int(np.argmax(spectrum))
    peak_frequency = peak_bin * 48000 / sine.size

    assert peak_frequency == pytest.approx(100.0)


def test_exact_inversion_cancels_to_near_zero():
    noise = generate_sine(100.0, 48000, 0.5)
    result = run_cancellation_experiment(
        noise,
        48000,
        ANCMismatchConfig(),
    )

    assert result.rms_after < 1e-12
    assert result.attenuation_db < -100.0
    assert np.allclose(result.residual, 0.0)


def test_amplitude_mismatch_reduces_cancellation():
    noise = generate_sine(100.0, 48000, 0.5)
    ideal = run_cancellation_experiment(
        noise,
        48000,
        ANCMismatchConfig(),
    )
    mismatch = run_cancellation_experiment(
        noise,
        48000,
        ANCMismatchConfig(amplitude_ratio=0.5),
    )

    assert mismatch.rms_after > ideal.rms_after
    assert mismatch.attenuation_db > ideal.attenuation_db


def test_phase_mismatch_reduces_cancellation():
    noise = generate_sine(100.0, 48000, 0.5)
    ideal = run_cancellation_experiment(
        noise,
        48000,
        ANCMismatchConfig(),
    )
    mismatch = run_cancellation_experiment(
        noise,
        48000,
        ANCMismatchConfig(phase_shift_degrees=90.0),
    )

    assert mismatch.rms_after > ideal.rms_after


def test_time_delay_is_sample_based_and_bounded():
    source = np.ones(16)
    delayed = apply_time_delay(source, 2)
    assert np.array_equal(delayed[:2], np.zeros(2))
    assert np.array_equal(delayed[2:], source[:14])
    with pytest.raises(ValueError):
        apply_time_delay(source, 17)


def test_multi_frequency_and_white_noise_generation():
    multi = generate_multi_frequency_noise(
        (100.0, 300.0, 1000.0),
        (1.0, 0.5, 0.25),
        48000,
        0.5,
    )
    white = generate_white_noise(
        48000,
        0.5,
        amplitude=0.2,
        seed=7,
    )

    assert multi.shape == (24000,)
    assert white.shape == (24000,)
    assert np.std(white) > 0.0


def test_recorded_noise_loading(tmp_path):
    path = tmp_path / "noise.wav"
    audio = np.column_stack(
        (
            generate_sine(100.0, 48000, 0.1),
            generate_sine(100.0, 48000, 0.1),
        )
    )
    sf.write(path, audio, 48000)

    mono, sample_rate = load_recorded_noise(path)

    assert mono.shape == (4800,)
    assert sample_rate == 48000


def test_residual_metrics_and_invalid_delay():
    original = generate_sine(100.0, 48000, 0.1)
    residual = calculate_residual(original, -original)

    assert np.allclose(residual, 0.0)
    assert attenuation_db(original, residual) < -100.0
