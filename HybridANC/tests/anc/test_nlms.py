"""Unit tests for normalized LMS filtering and LMS/NLMS comparison."""

import numpy as np
import pytest

from prototype.python.nlms import (
    NLMSConfig,
    NLMSFilter,
    compare_lms_and_nlms,
    generate_changing_amplitude_reference,
    generate_constant_reference,
    rms_error,
)


def test_nlms_config_validation():
    with pytest.raises(ValueError):
        NLMSConfig(filter_length=0)
    with pytest.raises(ValueError):
        NLMSConfig(step_size=0)
    with pytest.raises(ValueError):
        NLMSConfig(step_size=1.1)
    with pytest.raises(ValueError):
        NLMSConfig(epsilon=0)


def test_silent_reference_does_not_update_weights():
    filter_object = NLMSFilter(
        NLMSConfig(filter_length=8, epsilon=1e-6)
    )
    reference = np.zeros(1000, dtype=np.float64)
    desired = np.linspace(0.0, 0.1, 1000)

    result = filter_object.process(reference, desired)

    assert np.array_equal(result.weights, np.zeros(8))
    assert np.all(np.isfinite(result.error))


def test_low_energy_reference_remains_safe():
    filter_object = NLMSFilter(
        NLMSConfig(filter_length=8, epsilon=1e-3)
    )
    reference = np.full(1000, 1e-8, dtype=np.float64)
    desired = np.ones(1000, dtype=np.float64)

    result = filter_object.process(reference, desired)

    assert np.all(np.isfinite(result.output))
    assert np.all(np.isfinite(result.error))
    assert np.max(np.abs(result.weights)) < 1.0


def test_streaming_blocks_preserve_state():
    reference, desired = generate_constant_reference(duration_seconds=0.5)
    filter_object = NLMSFilter(
        NLMSConfig(filter_length=32, step_size=0.5)
    )
    outputs = []
    errors = []

    for start in range(0, len(reference), 512):
        block_output, block_error = filter_object.process_block(
            reference[start:start + 512],
            desired[start:start + 512],
        )
        outputs.append(block_output)
        errors.append(block_error)

    assert len(np.concatenate(outputs)) == len(reference)
    assert len(np.concatenate(errors)) == len(reference)


def test_reset_restores_initial_state():
    reference, desired = generate_constant_reference(duration_seconds=0.2)
    filter_object = NLMSFilter(
        NLMSConfig(filter_length=32)
    )
    first = filter_object.process(reference, desired)
    filter_object.reset()
    second = filter_object.process(reference, desired)

    assert np.allclose(first.output, second.output)
    assert np.allclose(first.error, second.error)


def test_coefficient_history_and_residual_rms_history():
    reference, desired = generate_constant_reference(duration_seconds=0.2)
    filter_object = NLMSFilter(
        NLMSConfig(filter_length=16, coefficient_history_limit=100)
    )
    result = filter_object.process(reference, desired)

    assert result.coefficient_history.shape[1] == 16
    assert result.residual_rms_history.shape == (len(reference),)
    assert np.all(np.isfinite(result.residual_rms_history))


def test_identical_signals_compare_lms_and_nlms():
    reference, desired = generate_changing_amplitude_reference(
        duration_seconds=1.0
    )
    comparison = compare_lms_and_nlms(
        reference,
        desired,
        filter_length=32,
        lms_learning_rate=0.001,
        nlms_step_size=0.5,
    )

    assert comparison["lms"].error.shape == reference.shape
    assert comparison["nlms"].error.shape == reference.shape
    assert comparison["lms"].mean_error_power >= 0.0
    assert comparison["nlms"].mean_error_power >= 0.0


def test_changing_amplitude_improves_nlms_stability():
    reference, desired = generate_changing_amplitude_reference(
        duration_seconds=1.0
    )
    comparison = compare_lms_and_nlms(
        reference,
        desired,
        filter_length=32,
        lms_learning_rate=0.01,
        nlms_step_size=0.5,
    )

    assert np.all(np.isfinite(comparison["lms"].error))
    assert np.all(np.isfinite(comparison["nlms"].error))
    assert not comparison["nlms"].divergence_detected


@pytest.mark.parametrize("step_size", [0.1, 0.5, 1.0])
def test_learning_step_selection_is_supported(step_size):
    reference, desired = generate_constant_reference(duration_seconds=0.2)
    result = NLMSFilter(
        NLMSConfig(filter_length=16, step_size=step_size)
    ).process(reference, desired)

    assert rms_error(result.error) >= 0.0
    assert np.all(np.isfinite(result.weights))
