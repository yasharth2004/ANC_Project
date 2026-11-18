"""Initial HybridANC Python audio prototype.

This executable performs environment checks and optional short microphone
capture measurements. It intentionally does not implement ANC or generate
anti-noise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

import numpy as np
import scipy
import sounddevice as sd
import soundfile as sf

try:
    from .config import DeviceConfig, OperatingMode, PrototypeConfig
    from .audio_devices import format_device_table, discover_audio_devices
except ImportError:
    # Permit direct execution with "python prototype/python/main.py".
    from config import DeviceConfig, OperatingMode, PrototypeConfig
    from audio_devices import format_device_table, discover_audio_devices


LOGGER = logging.getLogger("hybridanc")


def configure_logging(level: str) -> None:
    """Configure console logging for the prototype."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"invalid logging level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def print_environment() -> None:
    """Print version and host audio information."""
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"SoundDevice: {sd.__version__}")
    print(f"SoundFile: {sf.__version__}")
    try:
        print(f"Host APIs: {sd.query_hostapis()}")
    except sd.PortAudioError as exc:
        print(f"Host APIs unavailable: {exc}")
    print("\nAudio devices:")
    try:
        print(sd.query_devices())
    except sd.PortAudioError as exc:
        print(f"Audio devices unavailable: {exc}")


def capture_rms(config: PrototypeConfig, duration_seconds: float) -> float:
    """Capture a short mono block and return RMS level without playback."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    frames = int(config.dsp.sample_rate * duration_seconds)
    LOGGER.info(
        "Capturing %.2f seconds at %d Hz",
        duration_seconds,
        config.dsp.sample_rate,
    )

    try:
        samples = sd.rec(
            frames,
            samplerate=config.dsp.sample_rate,
            channels=config.dsp.channels,
            dtype="float32",
            device=config.device.input_device,
            blocking=True,
        )
    except sd.PortAudioError as exc:
        LOGGER.error("Microphone capture failed: %s", exc)
        raise

    mono = np.mean(samples, axis=1) if samples.ndim > 1 else samples
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64), dtype=np.float64)))
    LOGGER.info("Capture RMS: %.6f", rms)
    return rms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HybridANC initial Python prototype"
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in OperatingMode],
        default=OperatingMode.CHECK_ONLY.value,
        help="Prototype operation mode",
    )
    parser.add_argument(
        "--input-device",
        default=None,
        help="SoundDevice input device index or name",
    )
    parser.add_argument(
        "--output-device",
        default=None,
        help="Reserved for future playback setup; index or name",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Capture duration in seconds",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print detected input and output devices",
    )
    return parser


def parse_device(value: str | None) -> int | str | None:
    """Resolve numeric device strings as device indices."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    config = PrototypeConfig(
        mode=OperatingMode(args.mode),
        device=DeviceConfig(
            input_device=parse_device(args.input_device),
            output_device=parse_device(args.output_device),
        ),
    )

    LOGGER.info("Starting HybridANC prototype")
    LOGGER.debug("Configuration: %s", config.to_dict())
    print_environment()

    if args.list_devices:
        catalog = discover_audio_devices()
        print("\nDetected devices:")
        print(format_device_table(catalog))

    if config.mode == OperatingMode.MICROPHONE_CAPTURE:
        capture_rms(config, args.duration)

    LOGGER.info("No ANC or anti-noise output was activated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
