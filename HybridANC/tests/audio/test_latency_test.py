"""Unit tests for latency report construction and loopback correlation."""

import numpy as np
import pytest

from prototype.python.latency_test import (
    LatencyConfig,
    build_latency_report,
    estimate_loopback_delay,
)


def test_latency_config_rejects_unlisted_format():
    with pytest.raises(ValueError, match="44100 or 48000"):
        LatencyConfig(sample_rate=32000)
    with pytest.raises(ValueError, match="64, 128, 256 or 512"):
        LatencyConfig(block_size=1024)


def test_report_contains_observed_statistics():
    report = build_latency_report(
        sample_rate=48000,
        block_size=512,
        routing_latency=[10.0, 12.0, 14.0],
        processing_times=[0.2, 0.4],
        callback_intervals=[10.6, 10.7],
        cpu_load_percent=12.5,
        input_overflow_count=2,
        output_underrun_count=3,
    )

    assert report.average_latency_ms == 12.0
    assert report.minimum_latency_ms == 10.0
    assert report.maximum_latency_ms == 14.0
    assert report.dropout_count == 5
    assert report.latency_sample_count == 3
    assert report.processing_sample_count == 2
    assert report.input_buffer_ms == pytest.approx(512 * 1000 / 48000)
    assert report.acoustic_round_trip_ms is None
    assert report.loopback_status == "unavailable"


def test_loopback_correlation_finds_known_delay():
    sample_rate = 48000
    reference = np.zeros(2048, dtype=np.float64)
    reference[256:1280] = np.sin(
        2 * np.pi * 1000 * np.arange(1024) / sample_rate
    )
    recording = np.zeros(4096, dtype=np.float64)
    delay_samples = 100
    recording[100:100 + 2048] = reference

    delay_ms, confidence = estimate_loopback_delay(
        reference,
        recording,
        sample_rate,
        minimum_confidence=0.35,
    )

    assert delay_ms is not None
    assert delay_ms == pytest.approx(delay_samples * 1000 / sample_rate)
    assert confidence > 0.35


def test_loopback_correlation_rejects_uncorrelated_signal():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=2048)
    recording = rng.normal(size=4096)

    delay_ms, confidence = estimate_loopback_delay(
        reference,
        recording,
        48000,
    )

    assert delay_ms is None
