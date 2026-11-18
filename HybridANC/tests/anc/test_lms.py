"""Unit tests for the basic LMS FIR adaptive filter."""

import numpy as np
import pytest

from prototype.python.lms import (
    LMSConfig,
    LMSFilter,
    fixed_fir_filter,
    generate_changing_amplitude,
    generate_constant_sine,
    generate_multi_frequency,
    generate_white_noise_reference,
    rms_error,
)


def test_lms_config_validation():
    with pytest.raises(ValueError):
        LMSConfig(filter_length=0)
    with pytest.raises(ValueError):
        LMSConfig(learning_rate=0)
    with pytest.raises(ValueError):
        LMSConfig(weight_limit=-1)


def test_known_identity_path():
    reference = np.ones(1000, dtype=np.float64)
    desired = np.ones(1000, dtype=np.float64)
    adaptive = LMSFilter(
        LMSConfig(filter_length=1, learning_rate=0.1)
    ).process(reference, desired)

    assert adaptive.weights[0] > 0.0
    assert rms_error(adaptive.error[-200:]) < 0.001


def test_one_tap_sine_converges():
    reference, desired = generate_constant_sine(duration_seconds=1.0)
    result = LMSFilter(
        LMSConfig(filter_length=32, learning_rate=0.02)
    ).process(reference, desired)

    assert rms_error(result.error[-2000:]) < rms_error(desired)
    assert result.coefficient_history.shape[1] == 32


def test_coefficient_history_is_recorded():
    reference = np.ones(100, dtype=np.float64)
    desired = np.ones(100, dtype=np.float64)
    filter_object = LMSFilter(
        LMSConfig(
            filter_length=8,
            learning_rate=0.01,
            coefficient_history_limit=20,
        )
    )
    result = filter_object.process(reference, desired)

    assert result.coefficient_history.shape == (20, 8)


def test_changing_amplitude_is_supported():
    reference, desired = generate_changing_amplitude(
        duration_seconds=0.5
    )
    result = LMSFilter(
        LMSConfig(filter_length=32, learning_rate=0.005)
    ).process(reference, desired)

    assert np.all(np.isfinite(result.error))
    assert rms_error(result.error) < 1.0


def test_multi_frequency_and_white_reference_are_supported():
    reference_multi, desired_multi = generate_multi_frequency(
        duration_seconds=0.5
    )
    reference_white, desired_white = generate_white_noise_reference(
        duration_seconds=0.5
    )
    multi_result = LMSFilter(
        LMSConfig(filter_length=32, learning_rate=0.001)
    ).process(reference_multi, desired_multi)
    white_result = LMSFilter(
        LMSConfig(filter_length=32, learning_rate=0.001)
    ).process(reference_white, desired_white)

    assert np.all(np.isfinite(multi_result.error))
    assert np.all(np.isfinite(white_result.error))


def test_fixed_filter_benchmark():
    reference, desired = generate_constant_sine(duration_seconds=0.5)
    fixed_error, weights = fixed_fir_filter(reference, desired, 32)

    assert weights.shape == (32,)
    assert np.all(np.isfinite(fixed_error))


def test_divergence_and_invalid_inputs_are_blocked():
    filter_object = LMSFilter(
        LMSConfig(
            filter_length=8,
            learning_rate=100.0,
            weight_limit=1.0,
        )
    )
    result = filter_object.process(
        np.ones(100, dtype=np.float64),
        np.ones(100, dtype=np.float64),
    )

    assert np.max(np.abs(result.weights)) <= 1.0
    with pytest.raises(ValueError):
        LMSFilter().process(
            np.array([]),
            np.array([]),
        )
    with pytest.raises(ValueError):
        LMSFilter().process(
            np.ones(10),
            np.ones(11),
        )
