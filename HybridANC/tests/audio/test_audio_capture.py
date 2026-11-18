"""Unit tests for the callback-based audio capture module."""

from types import SimpleNamespace

import numpy as np
import pytest

from prototype.python.audio_capture import (
    AudioCapture,
    AudioCaptureConfig,
    DeviceDisconnectedError,
)


class MockInputStream:
    def __init__(self, callback, started=False):
        self.callback = callback
        self.started = started
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0

    def start(self):
        self.start_calls += 1
        self.started = True

    def stop(self):
        self.stop_calls += 1
        self.started = False

    def close(self):
        self.close_calls += 1


class MockAudioBackend:
    def __init__(self):
        self.stream = None

    def check_input_settings(self, **kwargs):
        if kwargs["samplerate"] not in (44100, 48000):
            raise RuntimeError("unsupported sample rate")

    def InputStream(self, callback, **kwargs):
        self.stream = MockInputStream(callback)
        return self.stream


def test_config_rejects_stereo_and_unsupported_rate():
    with pytest.raises(ValueError, match="44100 or 48000"):
        AudioCaptureConfig(sample_rate=32000)
    with pytest.raises(ValueError, match="mono"):
        AudioCaptureConfig(channels=2)


def test_start_stop_and_read_callback_block():
    backend = MockAudioBackend()
    capture = AudioCapture(
        AudioCaptureConfig(sample_rate=48000, block_size=8),
        backend=backend,
    )

    capture.start()
    assert capture.is_running
    assert backend.stream.start_calls == 1

    indata = np.ones((8, 1), dtype=np.float32) * 0.25
    backend.stream.callback(indata, 8, None, None)
    block = capture.read(timeout=0.1)

    assert block.frames == 8
    assert block.sequence == 0
    assert block.sample_rate == 48000
    assert np.all(block.samples == 0.25)

    block.release()
    capture.stop()
    assert not capture.is_running
    assert backend.stream.stop_calls == 1
    assert backend.stream.close_calls == 1


def test_input_disconnection_status_is_reported():
    backend = MockAudioBackend()
    capture = AudioCapture(
        AudioCaptureConfig(block_size=8),
        backend=backend,
    )
    capture.start()

    backend.stream.callback(
        np.zeros((8, 1), dtype=np.float32),
        8,
        None,
        "Device disconnected",
    )

    with pytest.raises(DeviceDisconnectedError):
        capture.raise_if_failed()
    capture.stop()


def test_callback_overflow_is_counted():
    backend = MockAudioBackend()
    capture = AudioCapture(
        AudioCaptureConfig(block_size=8, queue_depth=1, buffer_count=1),
        backend=backend,
    )
    capture.start()
    indata = np.zeros((8, 1), dtype=np.float32)

    backend.stream.callback(indata, 8, None, "input overflow")

    assert capture.overflow_count == 1
    capture.stop()


def test_callback_interval_history_is_bounded():
    backend = MockAudioBackend()
    capture = AudioCapture(
        AudioCaptureConfig(block_size=8),
        backend=backend,
    )
    capture.start()
    indata = np.zeros((8, 1), dtype=np.float32)

    for _ in range(300):
        backend.stream.callback(indata, 8, None, None)
        block = capture.read(timeout=0.1)
        block.release()

    intervals = capture.callback_interval_samples
    assert intervals.shape == (256,)
    capture.stop()
