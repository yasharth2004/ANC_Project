"""Tests for the experimental offline ANC comparison dashboard."""

import csv
import json

import numpy as np
import pytest

from prototype.python.anc_comparison_dashboard import (
    ALGORITHMS,
    DashboardConfig,
    cumulative_residual_rms,
    export_csv,
    export_json,
    generate_noise,
    generate_secondary_path,
    run_comparison,
)


def test_dashboard_config_validation():
    with pytest.raises(ValueError):
        DashboardConfig(noise_frequency_hz=-1)
    with pytest.raises(ValueError):
        DashboardConfig(noise_amplitude=-0.1)
    with pytest.raises(ValueError):
        DashboardConfig(noise_type="invalid")
    with pytest.raises(ValueError):
        DashboardConfig(filter_length=0)
    with pytest.raises(ValueError):
        DashboardConfig(secondary_delay_samples=-1)


def test_noise_generators_use_shared_configuration():
    sine = DashboardConfig(
        noise_type="sine",
        noise_frequency_hz=100,
        noise_amplitude=0.5,
        duration_seconds=0.2,
    )
    sine_first = generate_noise(sine)
    sine_second = generate_noise(sine)

    assert np.array_equal(sine_first, sine_second)
    assert sine_first.shape == (9600,)
    assert np.max(np.abs(sine_first)) == pytest.approx(0.5)


def test_white_noise_is_reproducible():
    config = DashboardConfig(
        noise_type="white",
        noise_amplitude=0.25,
        duration_seconds=0.2,
        seed=2026,
    )
    first = generate_noise(config)
    second = generate_noise(config)

    assert np.array_equal(first, second)
    assert np.std(first) > 0.0


def test_all_algorithms_run_against_identical_original():
    config = DashboardConfig(
        noise_type="sine",
        noise_frequency_hz=100,
        duration_seconds=0.2,
        filter_length=32,
        learning_rate=0.001,
        secondary_delay_samples=8,
    )
    original, secondary, results = run_comparison(config)

    assert set(result.name for result in results.values()) == {
        "Fixed phase inversion",
        "LMS",
        "NLMS",
        "FxLMS",
    }
    assert all(result.residual.shape == original.shape for result in results.values())
    assert all(result.residual_rms >= 0.0 for result in results.values())
    assert all(np.isfinite(result.noise_reduction_db) for result in results.values())
    assert secondary.shape == (32,)


def test_secondary_path_models_are_distinct():
    base = DashboardConfig(
        filter_length=64,
        secondary_delay_samples=16,
        secondary_path_model="unit_delay",
    )
    unit = generate_secondary_path(base)
    lead = generate_secondary_path(
        DashboardConfig(
            filter_length=64,
            secondary_delay_samples=16,
            secondary_path_model="delayed_lead",
        )
    )
    resonant = generate_secondary_path(
        DashboardConfig(
            filter_length=64,
            secondary_delay_samples=16,
            secondary_path_model="delayed_resonant",
        )
    )

    assert not np.allclose(unit, lead)
    assert not np.allclose(unit, resonant)


def test_cumulative_residual_rms_is_computed():
    residual = np.ones(100)
    values = cumulative_residual_rms(residual)

    assert values.shape == residual.shape
    assert np.allclose(values, 1.0)


def test_csv_and_json_exports(tmp_path):
    config = DashboardConfig(
        noise_type="sine",
        duration_seconds=0.05,
        filter_length=16,
        secondary_delay_samples=4,
    )
    original, _, results = run_comparison(config)
    csv_path = tmp_path / "comparison.csv"
    json_path = tmp_path / "comparison.json"

    export_csv(csv_path, config, original, results)
    export_json(json_path, config, results)

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert len(rows) > 100
    assert "summaries" in payload
    assert len(payload["summaries"]) == 4
