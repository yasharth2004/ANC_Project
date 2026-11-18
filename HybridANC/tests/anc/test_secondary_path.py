"""Unit tests for the synthetic FIR secondary acoustic-path module."""

import numpy as np
import pytest

from prototype.python.secondary_path import (
    FIRSecondaryPath,
    FIRSecondaryPathConfig,
    add_measurement_noise,
    estimate_peak_delay,
    estimate_secondary_path,
    generate_controlled_excitation,
    generate_synthetic_secondary_path,
    normalized_mse,
)


def test_config_validation():
    with pytest.raises(ValueError):
        FIRSecondaryPathConfig(sample_rate=32000)
    with pytest.raises(ValueError):
        FIRSecondaryPathConfig(fir_length=0)
    with pytest.raises(ValueError):
        FIRSecondaryPathConfig(delay_samples=-1)
    with pytest.raises(ValueError):
        FIRSecondaryPathConfig(resonance_hz=24000)


def test_impulse_delay_is_causal_and_exact():
    impulse = np.zeros(128)
    impulse[16] = 1.0
    path = FIRSecondaryPath(impulse, 48000, source="synthetic")
    excitation = np.zeros(256)
    excitation[0] = 1.0

    output = path.convolve(excitation)

    assert path.peak_delay_samples() == 16
    assert output.shape == excitation.shape
    assert output[16] == pytest.approx(1.0)
    assert np.all(output[:16] == 0.0)


def test_synthetic_path_has_different_delays():
    short = generate_synthetic_secondary_path(
        FIRSecondaryPathConfig(
            fir_length=128,
            delay_samples=8,
        )
    )
    long = generate_synthetic_secondary_path(
        FIRSecondaryPathConfig(
            fir_length=128,
            delay_samples=32,
        )
    )

    assert short.peak_delay_samples() == 8
    assert long.peak_delay_samples() == 32
    assert not np.allclose(
        short.impulse_response,
        long.impulse_response,
    )


def test_frequency_response_contains_magnitude_and_phase():
    path = generate_synthetic_secondary_path(
        FIRSecondaryPathConfig(
            fir_length=128,
            resonance_hz=1000.0,
        )
    )

    frequencies, magnitude_db, phase = path.frequency_response(1024)

    assert frequencies.shape == magnitude_db.shape == phase.shape
    assert np.max(frequencies) <= 24000.0
    assert np.all(np.isfinite(magnitude_db))
    assert np.all(np.isfinite(phase))


def test_controlled_excitation_is_bounded_and_reproducible():
    first = generate_controlled_excitation(
        48000,
        0.5,
        seed=2026,
    )
    second = generate_controlled_excitation(
        48000,
        0.5,
        seed=2026,
    )

    assert np.array_equal(first, second)
    assert np.max(np.abs(first)) <= 0.05


def test_measurement_noise_is_seeded():
    clean = np.sin(np.arange(1000) / 100)
    noisy_first = add_measurement_noise(clean, 0.01, seed=123)
    noisy_second = add_measurement_noise(clean, 0.01, seed=123)

    assert np.array_equal(noisy_first, noisy_second)
    assert not np.array_equal(noisy_first, clean)


def test_secondary_path_estimation_recovers_exact_impulse():
    rng = np.random.default_rng(42)
    excitation = rng.normal(0.0, 0.05, 8192)
    true_impulse = np.zeros(128)
    true_impulse[16] = 1.0
    true_impulse[17] = -0.20
    true_impulse[20] = 0.12
    true_path = FIRSecondaryPath(
        true_impulse,
        48000,
        source="synthetic",
    )
    clean_measurement = true_path.convolve(excitation)

    estimated = estimate_secondary_path(
        excitation,
        clean_measurement,
        128,
    )

    mse = normalized_mse(true_impulse, estimated)
    assert mse < 1e-12
    assert estimate_peak_delay(estimated) == 16


def test_estimation_with_noise_has_bounded_error():
    rng = np.random.default_rng(44)
    excitation = rng.normal(0.0, 0.05, 8192)
    true_impulse = generate_synthetic_secondary_path(
        FIRSecondaryPathConfig(
            fir_length=128,
            delay_samples=16,
            noise_gain=0.0,
        )
    ).impulse_response
    true_path = FIRSecondaryPath(
        true_impulse,
        48000,
        source="synthetic",
    )
    clean = true_path.convolve(excitation)
    noisy = add_measurement_noise(clean, 0.001, seed=44)

    estimated = estimate_secondary_path(
        excitation,
        noisy,
        128,
    )

    assert normalized_mse(true_impulse, estimated) < 0.10
