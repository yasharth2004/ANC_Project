"""Unit tests for capture-to-playback passthrough coordination."""

import numpy as np
import pytest

from prototype.python.audio_passthrough import (
    AudioPassthrough,
    AudioPassthroughConfig,
)


class MockBlock:
    def __init__(self):
        self.samples = np.full(8, 0.25, dtype=np.float32)

    def release(self):
        pass


class MockCapture:
    def __init__(self):
        self.is_running = False
        self.overflow_count = 0

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False

    def read(self, timeout=None):
        if not self.is_running:
            raise TimeoutError
        return MockBlock()

    def raise_if_failed(self):
        pass


class MockPlayback:
    def __init__(self):
        self.is_running = False
        self.underrun_count = 0
        self.gain = 0.05
        self.written = []

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False

    def set_gain(self, gain):
        self.gain = gain

    def write(self, samples, timeout=None):
        self.written.append(samples.copy())

    def write_silence(self):
        pass

    def clear(self):
        self.written.clear()

    def raise_if_failed(self):
        pass


def test_rejects_uncontrolled_gain():
    with pytest.raises(ValueError, match="0.0 and 0.25"):
        AudioPassthroughConfig(gain=0.5)


def test_start_status_and_emergency_stop():
    capture = MockCapture()
    playback = MockPlayback()
    passthrough = AudioPassthrough(
        AudioPassthroughConfig(
            input_device=2,
            output_device=1,
            block_size=8,
            gain=0.05,
        ),
        capture=capture,
        playback=playback,
    )

    passthrough.start()
    status = passthrough.status()
    assert status.running
    assert status.input_device == 2
    assert status.output_device == 1

    passthrough.emergency_stop()
    assert not passthrough.is_running
    assert passthrough.gain == 0.0


def test_switch_devices_stops_and_restarts():
    capture = MockCapture()
    playback = MockPlayback()
    passthrough = AudioPassthrough(
        AudioPassthroughConfig(),
        capture=capture,
        playback=playback,
    )
    passthrough.start()

    passthrough.switch_devices(2, 1)

    assert passthrough.status().input_device == 2
    assert passthrough.status().output_device == 1
    assert passthrough.status().running
    passthrough.stop()
