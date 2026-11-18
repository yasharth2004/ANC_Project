"""Tests for acoustic-performance evaluation from before/after recordings."""

import csv
import json

import numpy as np
import pytest
import soundfile as sf

from prototype.python.anc_evaluation import (
    EvaluationConfig,
    band_attenuation,
    calculate_noise_reduction_db,
    calculate_rms,
    calculate_convergence_time,
    evaluate,
    export_csv,
    export_json,
    load_evaluation_wavs,
    stability_metrics,
)


def generate_scenario(
    scenario: str,
    sample_rate: int = 48000,
    duration_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = int(sample_rate * duration_seconds)
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    if scenario == "fan":
        noise = (
            0.12 * np.sin(2 * np.pi * 60 * time)
            + 0.05 * np.sin(2 * np.pi * 120 * time)
        )
    elif scenario == "tone":
        noise = 0.25 * np.sin(2 * np.pi * 1000 * time)
    elif scenario == "multi":
        noise = (
            0.20 * np.sin(2 * np.pi * 100 * time)
            + 0.15 * np.sin(2 * np.pi * 350 * time)
            + 0.10 * np.sin(2 * np.pi * 1000 * time)
        )
    elif scenario == "changing":
        noise = (
            np.linspace(0.1, 0.5, frame_count)
            * np.sin(2 * np.pi * 300 * time)
        )
    elif scenario == "background":
        rng = np.random.default_rng(2026)
        noise = 0.05 * rng.normal(size=frame_count)
    else:
        raise ValueError("unsupported test scenario")
    after = 0.5 * noise
    return noise, after


@pytest.mark.parametrize(
    "scenario",
    ["fan", "tone", "multi", "changing", "background"],
)
def test_test_scenarios_produce_before_after_measurements(scenario):
    before, after = generate_scenario(scenario)
    summary, bands, stability, spectrum = evaluate(
        before,
        after,
        EvaluationConfig(
            fft_size=4096,
            before_trial_count=2,
            after_trial_count=2,
        ),
        trial_name=scenario,
    )

    assert summary.rms_before > 0
    assert summary.rms_after > 0
    assert summary.noise_reduction_db < 0
    assert len(bands) > 0
    assert stability.stability_status in (
        "stable",
        "unstable_or_variable",
    )
    assert spectrum[0].shape == spectrum[1].shape
    assert summary.repeatability_status == "repeat_trials_available"


def test_rms_and_noise_reduction_formula():
    before = np.ones(1000)
    after = np.full(1000, 0.5)

    rms_before = calculate_rms(before)
    rms_after = calculate_rms(after)
    reduction = calculate_noise_reduction_db(rms_before, rms_after)

    assert rms_before == pytest.approx(1.0)
    assert rms_after == pytest.approx(0.5)
    assert reduction == pytest.approx(-6.020599913)


def test_spectral_band_attenuation():
    before, after = generate_scenario("multi")
    bands = band_attenuation(
        before,
        after,
        48000,
        4096,
    )

    assert len(bands) > 0
    assert all(np.isfinite(band.attenuation_db) for band in bands)


def test_convergence_time_calculation():
    residual = np.concatenate(
        (
            np.ones(1000),
            np.full(1000, 0.05),
        )
    )
    convergence = calculate_convergence_time(
        residual,
        block_size=128,
        threshold_fraction=0.10,
    )

    assert convergence is not None
    assert convergence >= 1000


def test_stability_monitoring():
    before = np.sin(np.linspace(0, 100, 4096))
    after = 0.5 * np.sin(np.linspace(0, 100, 4096))
    stability = stability_metrics(before, after, block_size=512)

    assert stability.stability_status in (
        "stable",
        "unstable_or_variable",
    )
    assert np.isfinite(stability.relative_rms_drift)


def test_before_after_length_alignment():
    before = np.ones(100)
    after = np.ones(50)

    summary, _, _, _ = evaluate(
        before,
        after,
        EvaluationConfig(),
    )

    assert summary.frame_count == 50


def test_uncalibrated_status_and_calibration_flag():
    before, after = generate_scenario("tone")
    uncalibrated, _, _, _ = evaluate(before, after, EvaluationConfig())
    calibrated, _, _, _ = evaluate(
        before,
        after,
        EvaluationConfig(microphone_calibration_factor_db=-25.0),
    )

    assert uncalibrated.absolute_spl_status == "uncalibrated_electrical_rms"
    assert calibrated.absolute_spl_status == "calibrated_spl_requires_confirmed_reference"


def test_wav_loading_and_exports(tmp_path):
    before, after = generate_scenario("tone")
    before_path = tmp_path / "before.wav"
    after_path = tmp_path / "after.wav"
    sf.write(before_path, before, 48000)
    sf.write(after_path, after, 48000)
    config = EvaluationConfig(
        before_trial_count=3,
        after_trial_count=3,
    )

    loaded_before, loaded_after, sample_rate = load_evaluation_wavs(
        before_path,
        after_path,
        config,
    )
    summary, bands, stability, _ = evaluate(
        loaded_before,
        loaded_after,
        config,
    )
    csv_path = tmp_path / "evaluation.csv"
    json_path = tmp_path / "evaluation.json"
    export_csv(csv_path, summary, bands, stability)
    export_json(json_path, summary, bands, stability)

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert sample_rate == 48000
    assert len(rows) > 10
    assert "summary" in payload
    assert "frequency_bands" in payload
    assert "stability" in payload


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        EvaluationConfig(sample_rate=32000)
    with pytest.raises(ValueError):
        EvaluationConfig(fft_size=64)
    with pytest.raises(ValueError):
        calculate_noise_reduction_db(-1.0, 0.5)
    with pytest.raises(ValueError):
        calculate_rms(np.array([], dtype=np.float64))
