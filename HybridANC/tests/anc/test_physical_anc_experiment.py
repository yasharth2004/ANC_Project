"""Offline tests for the controlled physical-ANC experiment workflow."""

import numpy as np
import pytest
import soundfile as sf

from prototype.python.physical_anc_experiment import (
    PhysicalExperimentConfig,
    calculate_residual_metrics,
    estimate_impulse_and_delay,
    load_calibration,
    load_residual_metrics,
    validate_mono_float,
    write_template,
)


def test_physical_experiment_config():
    config = PhysicalExperimentConfig(
        configuration="A_existing_hardware",
    )
    assert config.configuration == "A_existing_hardware"
    with pytest.raises(ValueError):
        PhysicalExperimentConfig(configuration="invalid")
    with pytest.raises(ValueError):
        PhysicalExperimentConfig(
            configuration="B_future_measurement",
            sample_rate=32000,
        )


def test_residual_metrics_are_computed_not_fabricated():
    before = np.ones(1000, dtype=np.float64)
    after = np.full(1000, 0.5, dtype=np.float64)
    metrics = calculate_residual_metrics(before, after, 48000)

    assert metrics.rms_before == pytest.approx(1.0)
    assert metrics.rms_after == pytest.approx(0.5)
    assert metrics.reduction_db == pytest.approx(-6.020599913)


def test_template_generation(tmp_path):
    path = write_template(tmp_path / "report.md")
    text = path.read_text(encoding="utf-8")

    assert path.is_file()
    assert "Configuration A" in text
    assert "Configuration B" in text
    assert "no measured hardware result" in text.lower()
    assert "physical output" in text


def test_wav_loading_is_mono_and_valid(tmp_path):
    path = tmp_path / "recording.wav"
    stereo = np.column_stack((np.ones(1000), np.ones(1000)))
    sf.write(path, stereo, 48000)

    mono, sample_rate = validate_mono_float(path, "test")

    assert mono.shape == (1000,)
    assert sample_rate == 48000


def test_offline_impulse_estimation_recovers_delay():
    rng = np.random.default_rng(123)
    excitation = rng.normal(0.0, 0.05, 8192)
    impulse_true = np.zeros(256)
    impulse_true[32] = 1.0
    impulse_true[33] = -0.20

    measurement = np.zeros(8192)
    for tap, coefficient in enumerate(impulse_true):
        measurement[tap:tap + excitation.size] += (
            coefficient * excitation[:measurement.size - tap]
        )
    measurement = measurement[:8192]

    impulse_estimated, delay, confidence, normalized_mse = (
        estimate_impulse_and_delay(
            excitation,
            measurement,
            48000,
            fir_length=256,
            delay_search_limit=128,
        )
    )

    assert delay == 32
    assert confidence > 0.0
    assert normalized_mse < 1e-12
    assert impulse_estimated.shape == (256,)


def test_calibration_loading_with_synthetic_wav(tmp_path):
    rng = np.random.default_rng(7)
    excitation = rng.normal(0.0, 0.05, 4096)
    impulse = np.zeros(128)
    impulse[16] = 1.0
    measurement = np.zeros(4096)
    for tap, coefficient in enumerate(impulse):
        if tap < measurement.size:
            measurement[tap:tap + excitation.size] += (
                coefficient * excitation[:measurement.size - tap]
            )
    excitation_path = tmp_path / "excitation.wav"
    measurement_path = tmp_path / "measurement.wav"
    sf.write(excitation_path, excitation, 48000)
    sf.write(measurement_path, measurement, 48000)

    result = load_calibration(
        excitation_path,
        measurement_path,
        PhysicalExperimentConfig(
            configuration="B_future_measurement",
            fir_length=128,
            delay_search_limit=64,
        ),
    )

    assert result["estimated_delay_samples"] == 16
    assert result["physical_output_enabled"] is False
    assert result["hardware_cancellation_claim"] is False
    assert result["normalized_mse"] < 1e-12


def test_residual_loading_rejects_mismatched_rates(tmp_path):
    before_path = tmp_path / "before.wav"
    after_path = tmp_path / "after.wav"
    sf.write(before_path, np.ones(1000), 48000)
    sf.write(after_path, np.ones(1000), 44100)

    with pytest.raises(ValueError):
        load_residual_metrics(
            before_path,
            after_path,
            PhysicalExperimentConfig(
                configuration="B_future_measurement",
                sample_rate=48000,
            ),
        )
