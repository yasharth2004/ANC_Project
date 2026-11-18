"""Unit tests for offline FxLMS simulation."""

import numpy as np
import pytest

from prototype.python.fxlms import (
    FxLMSConfig,
    FxLMSFilter,
    compare_methods,
    fixed_phase_inversion,
    generate_changing_amplitude,
    generate_changing_reference,
    generate_multi_frequency,
    generate_stationary_sine,
    noise_reduction_db,
    secondary_path_with_delay,
    rms,
)


def test_config_validation():
    with pytest.raises(ValueError):
        FxLMSConfig(filter_length=0)
    with pytest.raises(ValueError):
        FxLMSConfig(learning_rate=0)
    with pytest.raises(ValueError):
        FxLMSConfig(epsilon=0)
    with pytest.raises(ValueError):
        FxLMSConfig(weight_limit=-1)


def test_stationary_sine_reduces_residual():
    reference, primary = generate_stationary_sine(
        duration_seconds=1.0
    )
    secondary = secondary_path_with_delay(
        48000,
        8,
        64,
    )
    result = FxLMSFilter(
        FxLMSConfig(
            filter_length=64,
            learning_rate=0.001,
            convergence_threshold=1e-8,
        ),
        secondary,
        secondary,
    ).process(reference, primary)

    assert np.all(np.isfinite(result.residual_error))
    assert result.residual_rms < rms(primary)
    assert result.noise_reduction_db < 0.0


def test_silent_reference_has_safe_update():
    secondary = secondary_path_with_delay(48000, 8, 32)
    filter_object = FxLMSFilter(
        FxLMSConfig(filter_length=32),
        secondary,
        secondary,
    )
    result = filter_object.process(
        np.zeros(1000),
        np.ones(1000),
    )

    assert np.all(np.isfinite(result.residual_error))
    assert np.array_equal(result.weights, np.zeros(32))


def test_secondary_path_delay_variations():
    short_reference, short_primary = generate_stationary_sine(
        duration_seconds=0.5
    )
    short = secondary_path_with_delay(48000, 8, 64)
    long = secondary_path_with_delay(48000, 32, 64)

    short_result = FxLMSFilter(
        FxLMSConfig(filter_length=64, learning_rate=0.0005),
        short,
        short,
    ).process(short_reference, short_primary)
    long_result = FxLMSFilter(
        FxLMSConfig(filter_length=64, learning_rate=0.0005),
        long,
        long,
    ).process(short_reference, short_primary)

    assert np.all(np.isfinite(short_result.residual_error))
    assert np.all(np.isfinite(long_result.residual_error))


def test_secondary_path_estimation_mismatch_is_supported():
    reference, primary = generate_stationary_sine(
        duration_seconds=0.5
    )
    true_secondary = secondary_path_with_delay(48000, 16, 64)
    estimated_secondary = secondary_path_with_delay(48000, 24, 64)

    result = FxLMSFilter(
        FxLMSConfig(filter_length=64, learning_rate=0.0005),
        estimated_secondary,
        true_secondary,
    ).process(reference, primary)

    assert np.all(np.isfinite(result.residual_error))


def test_multi_frequency_and_changing_references():
    reference_multi, primary_multi = generate_multi_frequency(
        duration_seconds=0.5
    )
    reference_changing, primary_changing = generate_changing_amplitude(
        duration_seconds=0.5
    )
    reference_new, primary_new = generate_changing_reference(
        duration_seconds=0.5
    )
    secondary = secondary_path_with_delay(48000, 16, 64)

    for reference, primary in (
        (reference_multi, primary_multi),
        (reference_changing, primary_changing),
        (reference_new, primary_new),
    ):
        result = FxLMSFilter(
            FxLMSConfig(filter_length=64, learning_rate=0.0005),
            secondary,
            secondary,
        ).process(reference, primary)
        assert np.all(np.isfinite(result.residual_error))


def test_fixed_phase_inversion_and_comparison_methods():
    reference, primary = generate_stationary_sine(
        duration_seconds=0.5
    )
    fixed = fixed_phase_inversion(reference)
    assert np.allclose(primary + fixed, np.zeros_like(primary))

    secondary = secondary_path_with_delay(48000, 16, 64)
    reports = compare_methods(
        reference,
        primary,
        secondary,
        secondary,
        FxLMSConfig(filter_length=64, learning_rate=0.0005),
    )
    assert set(reports) == {
        "no_cancellation",
        "fixed_phase_inversion",
        "lms",
        "nlms",
        "fxlms",
    }


def test_reset_and_invalid_signals():
    secondary = secondary_path_with_delay(48000, 8, 32)
    filter_object = FxLMSFilter(
        FxLMSConfig(filter_length=32),
        secondary,
        secondary,
    )
    result = filter_object.process(
        np.ones(512),
        np.ones(512),
    )
    filter_object.reset()
    reset_result = filter_object.process(
        np.ones(512),
        np.ones(512),
    )

    assert np.all(np.isfinite(result.residual_error))
    assert np.all(np.isfinite(reset_result.residual_error))
    with pytest.raises(ValueError):
        filter_object.process(
            np.array([]),
            np.array([]),
        )
    with pytest.raises(ValueError):
        filter_object.process(
            np.ones(10),
            np.ones(11),
        )
