"""Unit tests for the real-time playback module."""

from types import SimpleNamespace

import numpy as np
import pytest

from prototype.python.audio_playback import (
    AudioPlayback,
    AudioPlaybackConfig,
    OutputDeviceDisconnectedError,
)


class MockOutputStream:
    def __init__(self, callback):
        self.callback = callback
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1

    def close(self):
        self.close_calls += 1


class MockPlaybackBackend:
    def __init__(self):
        self.stream = None

    def check_output_settings(self, **kwargs):
        if kwargs["samplerate"] not in (44100, 48000):
            raise RuntimeError("unsupported sample rate")

    def OutputStream(self, callback, **kwargs):
        self.stream = MockOutputStream(callback)
        return self.stream


def test_config_rejects_invalid_output_format():
    with pytest.raises(ValueError, match="44100 or 48000"):
        AudioPlaybackConfig(sample_rate=32000)
    with pytest.raises(ValueError, match="1 or 2"):
        AudioPlaybackConfig(channels=4)
    with pytest.raises(ValueError, match="gain"):
        AudioPlaybackConfig(gain=1.5)


def test_start_write_mono_and_stop():
    backend = MockPlaybackBackend()
    playback = AudioPlayback(
        AudioPlaybackConfig(
            sample_rate=48000,
            channels=1,
            block_size=8,
            gain=0.25,
        ),
        backend=backend,
    )

    playback.start()
    playback.clear()
    playback.write(np.ones(8, dtype=np.float32), timeout=0.1)
    outdata = np.zeros((8, 1), dtype=np.float32)
    backend.stream.callback(outdata, 8, None, None)

    assert np.all(outdata == 0.25)
    playback.stop()
    assert backend.stream.start_calls == 1
    assert backend.stream.stop_calls == 1
    assert backend.stream.close_calls == 1


def test_limiter_never_exceeds_peak():
    playback = AudioPlayback(
        AudioPlaybackConfig(
            channels=2,
            block_size=8,
            gain=0.25,
            limiter_peak=0.5,
        ),
        backend=MockPlaybackBackend(),
    )
    playback.write(np.ones((8, 2), dtype=np.float32) * 2.0)
    outdata = np.zeros((8, 2), dtype=np.float32)

    playback._audio_callback(outdata, 8, None, None)

    assert np.max(outdata) <= 0.5
    assert np.min(outdata) >= -0.5


def test_mono_block_is_duplicated_to_stereo():
    playback = AudioPlayback(
        AudioPlaybackConfig(
            channels=2,
            block_size=8,
        ),
        backend=MockPlaybackBackend(),
    )

    playback.write(np.ones(8, dtype=np.float32))
    block = playback._input_queue.get_nowait()

    assert block.shape == (8, 2)
    assert np.all(block == 1.0)


def test_underrun_generates_silence():
    playback = AudioPlayback(
        AudioPlaybackConfig(channels=2, block_size=8),
        backend=MockPlaybackBackend(),
    )
    playback.start()
    playback.clear()
    outdata = np.ones((8, 2), dtype=np.float32)

    playback._audio_callback(outdata, 8, None, None)

    assert np.all(outdata == 0.0)
    assert playback.underrun_count == 1
    playback.stop()


def test_disconnection_status_is_reported():
    playback = AudioPlayback(
        AudioPlaybackConfig(channels=1, block_size=8),
        backend=MockPlaybackBackend(),
    )
    playback.start()
    playback._audio_callback(
        np.zeros((8, 1), dtype=np.float32),
        8,
        None,
        "Output device disconnected",
    )

    with pytest.raises(OutputDeviceDisconnectedError):
        playback.raise_if_failed()
    playback.stop()


def test_callback_interval_history_is_bounded():
    playback = AudioPlayback(
        AudioPlaybackConfig(channels=1, block_size=8),
        backend=MockPlaybackBackend(),
    )
    playback.start()
    playback.clear()
    output = np.zeros((8, 1), dtype=np.float32)

    for _ in range(300):
        playback._audio_callback(output, 8, None, None)

    intervals = playback.callback_interval_samples
    assert intervals.shape == (256,)
    playback.stop()
