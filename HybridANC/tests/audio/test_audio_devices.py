"""Unit tests for audio-device discovery using mocked PortAudio responses."""

from types import SimpleNamespace

import pytest

from prototype.python.audio_devices import (
    discover_audio_devices,
    get_default_input,
    get_default_output,
    list_input_devices,
    list_output_devices,
    select_device,
    validate_device_settings,
)


class MockAudioBackend:
    """Minimal sounddevice-compatible backend used by unit tests."""

    PortAudioError = RuntimeError
    default = SimpleNamespace(device=(0, 1))

    def query_hostapis(self):
        return [{"name": "Mock Host"}, {"name": "Second Host"}]

    def query_devices(self, device=None):
        devices = [
            {
                "name": "Built-in Microphone",
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "Wired Headphones",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            },
            {
                "name": "Disconnected USB Microphone",
                "hostapi": 1,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
        ]
        if device is None:
            return devices
        if device == 2:
            raise RuntimeError("device disconnected")
        return devices[device]

    def check_input_settings(self, device, channels, samplerate):
        if channels > 1 or samplerate != 48000:
            raise RuntimeError("unsupported input format")

    def check_output_settings(self, device, channels, samplerate):
        if channels > 2 or samplerate != 44100:
            raise RuntimeError("unsupported output format")


def test_discovery_separates_inputs_outputs_and_defaults():
    catalog = discover_audio_devices(MockAudioBackend())

    assert len(catalog.devices) == 3
    assert [device.device_id for device in list_input_devices(catalog)] == [0]
    assert [device.device_id for device in list_output_devices(catalog)] == [1]
    assert get_default_input(catalog).name == "Built-in Microphone"
    assert get_default_output(catalog).name == "Wired Headphones"


def test_disconnected_device_is_unavailable():
    catalog = discover_audio_devices(MockAudioBackend())
    disconnected = catalog.get(2)

    assert disconnected.usable is False
    assert disconnected.input_available is False
    assert disconnected.output_available is False


def test_selection_by_id_rejects_wrong_direction():
    catalog = discover_audio_devices(MockAudioBackend())

    assert select_device(0, "input", catalog=catalog).name == "Built-in Microphone"
    with pytest.raises(ValueError, match="no output channels"):
        select_device(0, "output", catalog=catalog)


def test_format_validation_uses_portaudio():
    catalog = discover_audio_devices(MockAudioBackend())
    backend = MockAudioBackend()

    assert validate_device_settings(
        0,
        direction="input",
        sample_rate=48000,
        channels=1,
        catalog=catalog,
        backend=backend,
    )
    assert not validate_device_settings(
        0,
        direction="input",
        sample_rate=44100,
        channels=1,
        catalog=catalog,
        backend=backend,
    )


def test_sounddevice_input_output_pair_shape():
    class InputOutputPair:
        input = 0
        output = 1

    class PairBackend(MockAudioBackend):
        default = SimpleNamespace(device=InputOutputPair())

    catalog = discover_audio_devices(PairBackend())

    assert catalog.default_input_id == 0
    assert catalog.default_output_id == 1
