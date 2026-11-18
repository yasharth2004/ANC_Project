"""Audio-device discovery for the HybridANC Python prototype.

This module has no DSP or audio-processing behavior. It only discovers,
describes, and validates PortAudio/SoundDevice devices.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from types import ModuleType
from typing import Any, Iterable

import sounddevice as sd


LOGGER = logging.getLogger("hybridanc.audio_devices")


@dataclass(frozen=True)
class AudioDeviceInfo:
    """Stable platform-independent description of an audio device."""

    device_id: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    input_available: bool
    output_available: bool
    usable: bool = True


@dataclass(frozen=True)
class DeviceCatalog:
    """Complete discovered device catalog."""

    devices: tuple[AudioDeviceInfo, ...]
    default_input_id: int | None
    default_output_id: int | None

    def input_devices(self) -> tuple[AudioDeviceInfo, ...]:
        return tuple(
            device for device in self.devices if device.input_available
        )

    def output_devices(self) -> tuple[AudioDeviceInfo, ...]:
        return tuple(
            device for device in self.devices if device.output_available
        )

    def get(self, device_id: int) -> AudioDeviceInfo:
        for device in self.devices:
            if device.device_id == device_id:
                return device
        raise KeyError(f"No discovered audio device with ID {device_id}")


def _integer(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback


def _number(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return fallback


def _portaudio_error(backend: ModuleType) -> tuple[type[Exception], ...]:
    error_type = getattr(backend, "PortAudioError", Exception)
    if isinstance(error_type, type):
        return (error_type,)
    return (Exception,)


def _host_api_names(backend: ModuleType) -> dict[int, str]:
    try:
        host_apis = backend.query_hostapis()
    except Exception as exc:
        LOGGER.warning("Could not query PortAudio host APIs: %s", exc)
        return {}

    names: dict[int, str] = {}
    if isinstance(host_apis, dict):
        host_apis = [host_apis]
    for index, host_api in enumerate(host_apis):
        if isinstance(host_api, dict):
            names[index] = str(host_api.get("name", f"Host API {index}"))
    return names


def _default_device_ids(backend: ModuleType) -> tuple[int | None, int | None]:
    try:
        default_devices = backend.default.device
        if isinstance(default_devices, (list, tuple)):
            input_id = _integer(default_devices[0], -1)
            output_id = _integer(default_devices[1], -1)
        else:
            input_id = _integer(default_devices, -1)
            output_id = None
    except Exception as exc:
        LOGGER.warning("Could not determine default audio devices: %s", exc)
        input_id = None
        output_id = None

    return (
        input_id if input_id is not None and input_id >= 0 else None,
        output_id if output_id is not None and output_id >= 0 else None,
    )


def discover_audio_devices(backend: ModuleType = sd) -> DeviceCatalog:
    """Discover all visible devices and identify default input/output devices."""
    host_apis = _host_api_names(backend)
    default_input_id, default_output_id = _default_device_ids(backend)
    discovered: list[AudioDeviceInfo] = []

    try:
        raw_devices = backend.query_devices()
    except Exception as exc:
        LOGGER.error("Could not enumerate PortAudio devices: %s", exc)
        return DeviceCatalog((), default_input_id, default_output_id)

    if isinstance(raw_devices, dict):
        raw_devices = [raw_devices]

    for device_id, raw_device in enumerate(raw_devices):
        if not isinstance(raw_device, dict):
            LOGGER.warning("Ignoring invalid device entry at ID %d", device_id)
            continue

        name = str(raw_device.get("name", f"Unnamed device {device_id}"))
        host_api_id = _integer(raw_device.get("hostapi"), -1)
        host_api = host_apis.get(host_api_id, "Unknown")

        max_input_channels = _integer(raw_device.get("max_input_channels"))
        max_output_channels = _integer(raw_device.get("max_output_channels"))
        default_sample_rate = _number(raw_device.get("default_samplerate"))

        usable = True
        try:
            backend.query_devices(device=device_id)
        except Exception as exc:
            usable = False
            LOGGER.warning("Device %d (%s) is unavailable: %s", device_id, name, exc)

        discovered.append(
            AudioDeviceInfo(
                device_id=device_id,
                name=name,
                host_api=host_api,
                max_input_channels=max_input_channels,
                max_output_channels=max_output_channels,
                default_sample_rate=default_sample_rate,
                input_available=usable and max_input_channels > 0,
                output_available=usable and max_output_channels > 0,
                usable=usable,
            )
        )

    return DeviceCatalog(
        devices=tuple(discovered),
        default_input_id=default_input_id,
        default_output_id=default_output_id,
    )


def list_input_devices(
    catalog: DeviceCatalog | None = None,
    backend: ModuleType = sd,
) -> tuple[AudioDeviceInfo, ...]:
    """Return devices that expose at least one input channel."""
    resolved = catalog if catalog is not None else discover_audio_devices(backend)
    return resolved.input_devices()


def list_output_devices(
    catalog: DeviceCatalog | None = None,
    backend: ModuleType = sd,
) -> tuple[AudioDeviceInfo, ...]:
    """Return devices that expose at least one output channel."""
    resolved = catalog if catalog is not None else discover_audio_devices(backend)
    return resolved.output_devices()


def get_default_input(catalog: DeviceCatalog) -> AudioDeviceInfo | None:
    if catalog.default_input_id is None:
        return None
    try:
        return catalog.get(catalog.default_input_id)
    except KeyError:
        return None


def get_default_output(catalog: DeviceCatalog) -> AudioDeviceInfo | None:
    if catalog.default_output_id is None:
        return None
    try:
        return catalog.get(catalog.default_output_id)
    except KeyError:
        return None


def select_device(
    device_id: int,
    direction: str,
    catalog: DeviceCatalog | None = None,
    backend: ModuleType = sd,
) -> AudioDeviceInfo:
    """Select an available input or output device by numeric ID."""
    if direction not in {"input", "output"}:
        raise ValueError("direction must be 'input' or 'output'")

    resolved = catalog if catalog is not None else discover_audio_devices(backend)
    try:
        device = resolved.get(device_id)
    except KeyError as exc:
        raise LookupError(str(exc)) from exc

    if not device.usable:
        raise RuntimeError(f"Audio device {device_id} ({device.name}) is unavailable")
    if direction == "input" and not device.input_available:
        raise ValueError(f"Audio device {device_id} ({device.name}) has no input channels")
    if direction == "output" and not device.output_available:
        raise ValueError(f"Audio device {device_id} ({device.name}) has no output channels")
    return device


def validate_device_settings(
    device_id: int,
    *,
    direction: str,
    sample_rate: float,
    channels: int,
    catalog: DeviceCatalog | None = None,
    backend: ModuleType = sd,
) -> bool:
    """Check whether PortAudio accepts the requested device format."""
    if direction not in {"input", "output"}:
        raise ValueError("direction must be 'input' or 'output'")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")

    select_device(device_id, direction, catalog=catalog, backend=backend)

    try:
        if direction == "input":
            backend.check_input_settings(
                device=device_id,
                channels=channels,
                samplerate=sample_rate,
            )
        else:
            backend.check_output_settings(
                device=device_id,
                channels=channels,
                samplerate=sample_rate,
            )
    except Exception as exc:
        LOGGER.warning(
            "Device %d does not support %d-channel %g Hz %s settings: %s",
            device_id,
            channels,
            sample_rate,
            direction,
            exc,
        )
        return False
    return True


def format_device_table(catalog: DeviceCatalog) -> str:
    """Format the catalog as a readable monospace table."""
    headers = (
        "ID",
        "NAME",
        "HOST API",
        "IN",
        "OUT",
        "SAMPLE RATE",
        "DEFAULT",
    )
    rows: list[tuple[str, ...]] = []

    for device in catalog.devices:
        defaults: list[str] = []
        if device.device_id == catalog.default_input_id:
            defaults.append("IN")
        if device.device_id == catalog.default_output_id:
            defaults.append("OUT")

        status = "!"
        if not device.usable:
            status = "UNAVAILABLE"
        elif device.input_available or device.output_available:
            status = "-"

        rows.append(
            (
                str(device.device_id),
                device.name,
                device.host_api,
                str(device.max_input_channels),
                str(device.max_output_channels),
                f"{device.default_sample_rate:.1f}",
                "/".join(defaults) if defaults else "-",
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(columns: Iterable[str]) -> str:
        values = list(columns)
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        )

    lines = [render(headers), render("-" * width for width in widths)]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def main() -> int:
    """Command-line utility that prints the detected audio devices."""
    import argparse

    parser = argparse.ArgumentParser(
        description="List audio devices available to HybridANC"
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    catalog = discover_audio_devices()
    if not catalog.devices:
        print("No PortAudio devices were detected.")
        return 0

    print(format_device_table(catalog))

    default_input = get_default_input(catalog)
    default_output = get_default_output(catalog)
    print(
        "\nDefault input:",
        f"{default_input.device_id} - {default_input.name}"
        if default_input
        else "None",
    )
    print(
        "Default output:",
        f"{default_output.device_id} - {default_output.name}"
        if default_output
        else "None",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
