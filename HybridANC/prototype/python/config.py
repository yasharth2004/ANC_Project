"""Typed runtime configuration for the HybridANC Python prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OperatingMode(str, Enum):
    """Experimental prototype operating modes."""

    CHECK_ONLY = "check-only"
    MICROPHONE_CAPTURE = "microphone-capture"


@dataclass(frozen=True)
class DeviceConfig:
    """Device selection for audio capture and playback."""

    input_device: int | str | None = None
    output_device: int | str | None = None


@dataclass(frozen=True)
class DSPConfig:
    """DSP format shared with future C++ implementations."""

    sample_rate: int = 48_000
    block_size: int = 512
    channels: int = 1
    fft_size: int = 512

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if self.fft_size < 32 or self.fft_size & (self.fft_size - 1):
            raise ValueError("fft_size must be a power of two of at least 32")


@dataclass(frozen=True)
class SafetyConfig:
    """Safety defaults for microphone and future anti-noise experiments."""

    microphone_peak_limit_dbfs: float = -1.0
    anti_noise_enabled: bool = False
    calibration_required: bool = True
    output_ramp_ms: int = 100

    def __post_init__(self) -> None:
        if self.output_ramp_ms < 0:
            raise ValueError("output_ramp_ms cannot be negative")
        if self.microphone_peak_limit_dbfs > 0:
            raise ValueError("microphone_peak_limit_dbfs must not be above 0 dBFS")


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


@dataclass(frozen=True)
class PrototypeConfig:
    """Top-level configuration for the Python prototype."""

    mode: OperatingMode = OperatingMode.CHECK_ONLY
    device: DeviceConfig = field(default_factory=DeviceConfig)
    dsp: DSPConfig = field(default_factory=DSPConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, serializable representation."""
        return {
            "mode": self.mode.value,
            "device": {
                "input_device": self.device.input_device,
                "output_device": self.device.output_device,
            },
            "dsp": {
                "sample_rate": self.dsp.sample_rate,
                "block_size": self.dsp.block_size,
                "channels": self.dsp.channels,
                "fft_size": self.dsp.fft_size,
            },
            "safety": {
                "microphone_peak_limit_dbfs": self.safety.microphone_peak_limit_dbfs,
                "anti_noise_enabled": self.safety.anti_noise_enabled,
                "calibration_required": self.safety.calibration_required,
                "output_ramp_ms": self.safety.output_ramp_ms,
            },
            "logging": {
                "level": self.logging.level,
            },
        }
