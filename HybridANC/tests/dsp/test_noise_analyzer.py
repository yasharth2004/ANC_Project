"""Unit tests for environmental-noise analysis."""

from types import SimpleNamespace

import numpy as np
import pytest

from prototype.python.noise_analyzer import (
    NoiseAnalyzer,
    NoiseAnalyzerConfig,
    create_sine_demo,
)


class MockCapture:
    def __init__(self):
        self.is_running = False
        self.overflow_count = 0

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False

    def read(self, timeout=None):
        raise TimeoutError

    def raise_if_failed(self):
        pass


def test_config_rejects_invalid_fft_or_refresh():
    with pytest.raises(ValueError, match="fft_size"):
        NoiseAnalyzerConfig(fft_size=300)
    with pytest.raises(ValueError, match="refresh_interval_ms"):
        NoiseAnalyzerConfig(refresh_interval_ms=10)


def test_sine_analysis_reports_peak_rms_and_dominant_frequency():
    analyzer = NoiseAnalyzer(
        NoiseAnalyzerConfig(
            sample_rate=48000,
            fft_size=2048,
            block_size=512,
        ),
        capture=MockCapture(),
    )
    sine = create_sine_demo(48000, 1.0)

    snapshot = analyzer.analyze_window(sine)

    assert snapshot.peak == pytest.approx(0.25)
    assert snapshot.rms == pytest.approx(0.25 / np.sqrt(2), rel=0.05)
    assert snapshot.dominant_frequency_hz == pytest.approx(1000.0, abs=25.0)
    assert snapshot.spectrum_db.shape == (1025,)
    assert snapshot.spectrogram_db.shape == (1025, 120)
    assert "Mids" in snapshot.band_levels_db


def test_empty_analysis_is_rejected():
    analyzer = NoiseAnalyzer(
        NoiseAnalyzerConfig(),
        capture=MockCapture(),
    )
    with pytest.raises(ValueError, match="must not be empty"):
        analyzer.analyze_window(np.array([], dtype=np.float32))


def test_snapshot_is_a_copy():
    analyzer = NoiseAnalyzer(
        NoiseAnalyzerConfig(),
        capture=MockCapture(),
    )
    snapshot = analyzer.snapshot()
    snapshot.waveform[:] = 1.0

    fresh = analyzer.snapshot()
    assert not np.all(fresh.waveform == 1.0)
