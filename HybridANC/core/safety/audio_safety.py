"""Comprehensive platform-independent audio safety primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import threading
import time

import numpy as np


class SafetyState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    ACTIVE = "active"
    MUTED = "muted"
    EMERGENCY_STOPPED = "emergency_stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class AudioSafetyConfig:
    maximum_peak: float = 0.8912509381337456
    soft_clip_peak: float = 0.8912509381337456
    maximum_gain: float = 0.25
    startup_gain: float = 0.05
    ramp_duration_seconds: float = 0.25
    maximum_rms: float = 0.70
    divergence_weight_limit: float = 10_000.0
    divergence_error_limit: float = 10_000.0
    dropout_timeout_seconds: float = 0.50
    maximum_weight_count: int = 4096

    def __post_init__(self) -> None:
        if not np.isfinite(self.maximum_peak) or not 0.0 < self.maximum_peak <= 1.0:
            raise ValueError("maximum_peak must be greater than zero and at most 1.0")
        if not np.isfinite(self.soft_clip_peak) or not 0.0 < self.soft_clip_peak <= 1.0:
            raise ValueError("soft_clip_peak must be greater than zero and at most 1.0")
        if not np.isfinite(self.maximum_gain) or not 0.0 < self.maximum_gain <= 1.0:
            raise ValueError("maximum_gain must be greater than zero and at most 1.0")
        if not np.isfinite(self.startup_gain) or not 0.0 <= self.startup_gain <= self.maximum_gain:
            raise ValueError("startup_gain must be within output gain limits")
        if not np.isfinite(self.ramp_duration_seconds) or self.ramp_duration_seconds < 0:
            raise ValueError("ramp_duration_seconds must be finite and non-negative")
        if not np.isfinite(self.maximum_rms) or not 0.0 < self.maximum_rms <= 1.0:
            raise ValueError("maximum_rms must be greater than zero and at most 1.0")
        if not np.isfinite(self.divergence_weight_limit) or self.divergence_weight_limit <= 0:
            raise ValueError("divergence_weight_limit must be positive")
        if not np.isfinite(self.divergence_error_limit) or self.divergence_error_limit <= 0:
            raise ValueError("divergence_error_limit must be positive")
        if not np.isfinite(self.dropout_timeout_seconds) or self.dropout_timeout_seconds <= 0:
            raise ValueError("dropout_timeout_seconds must be positive")
        if not isinstance(self.maximum_weight_count, int) or self.maximum_weight_count <= 0:
            raise ValueError("maximum_weight_count must be positive")


class GainRamp:
    """Thread-safe linear gain ramp."""

    def __init__(
        self,
        start_gain: float = 0.0,
        maximum_gain: float = 0.25,
        ramp_duration_seconds: float = 0.25,
        *,
        clock=time.monotonic,
    ) -> None:
        if not np.isfinite(start_gain) or not 0.0 <= start_gain <= maximum_gain:
            raise ValueError("start_gain is outside the configured range")
        if not np.isfinite(maximum_gain) or not 0.0 < maximum_gain <= 1.0:
            raise ValueError("maximum_gain must be greater than zero and at most 1.0")
        if not np.isfinite(ramp_duration_seconds) or ramp_duration_seconds < 0:
            raise ValueError("ramp_duration_seconds must be non-negative")
        self._maximum_gain = float(maximum_gain)
        self._duration = float(ramp_duration_seconds)
        self._clock = clock
        self._current = float(start_gain)
        self._target = float(start_gain)
        self._started_at = clock()

    @property
    def current_gain(self) -> float:
        if self._target == self._current:
            elapsed_fraction = 1.0
        elif self._duration == 0:
            elapsed_fraction = 1.0
        else:
            elapsed_fraction = min(
                1.0,
                (self._clock() - self._started_at) / self._duration,
            )
        if elapsed_fraction >= 1.0:
            self._current = self._target
            return self._current
        start = getattr(self, "_ramp_start", self._current)
        gain = start + (self._target - start) * elapsed_fraction
        self._current = float(gain)
        return self._current

    def set_target(self, target_gain: float) -> None:
        if not np.isfinite(target_gain) or not 0.0 <= target_gain <= self._maximum_gain:
            raise ValueError("target_gain is outside the configured range")
        self._ramp_start = self.current_gain
        self._target = float(target_gain)
        self._started_at = self._clock()

    def mute(self) -> float:
        self.set_target(0.0)
        self._ramp_start = 0.0
        self._current = 0.0
        self._target = 0.0
        return 0.0


class PhysicalANCInterlock:
    """Requires explicit activation plus a validated calibrated state."""

    def __init__(self) -> None:
        self._enabled = False
        self._calibrated = False
        self._calibration_id: str | None = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def calibrated(self) -> bool:
        with self._lock:
            return self._calibrated

    @property
    def calibration_id(self) -> str | None:
        with self._lock:
            return self._calibration_id

    def set_calibration(
        self,
        valid: bool,
        calibration_id: str | None = None,
    ) -> None:
        if valid and (
            calibration_id is None
            or not isinstance(calibration_id, str)
            or not calibration_id.strip()
        ):
            raise ValueError(
                "valid calibration requires a non-empty calibration_id"
            )
        with self._lock:
            self._calibrated = bool(valid)
            self._calibration_id = (
                calibration_id if valid else None
            )
            if not valid:
                self._enabled = False

    def enable(
        self,
        operator_confirmed: bool,
        calibration_id: str | None = None,
    ) -> bool:
        with self._lock:
            if not operator_confirmed:
                self._enabled = False
                return False
            if not self._calibrated:
                self._enabled = False
                return False
            if calibration_id is not None and calibration_id != self._calibration_id:
                self._enabled = False
                return False
            self._enabled = True
            return True

    def disable(self) -> None:
        with self._lock:
            self._enabled = False


class AdaptiveFilterSafety:
    """Detects invalid, divergent or oversized adaptive-filter state."""

    def __init__(self, config: AudioSafetyConfig) -> None:
        self.config = config
        self._divergence_detected = False

    @property
    def divergence_detected(self) -> bool:
        return self._divergence_detected

    def reset(self) -> None:
        self._divergence_detected = False

    def validate(
        self,
        weights: np.ndarray,
        error: object,
        output: object,
    ) -> bool:
        weights_array = np.asarray(weights)
        if weights_array.ndim != 1:
            self._divergence_detected = True
            return False
        if weights_array.size > self.config.maximum_weight_count:
            self._divergence_detected = True
            return False

        values = (weights_array, np.asarray(error), np.asarray(output))
        for value in values:
            if not np.all(np.isfinite(value)):
                self._divergence_detected = True
                return False

        if np.max(np.abs(weights_array)) > self.config.divergence_weight_limit:
            self._divergence_detected = True
            return False
        if float(np.max(np.abs(np.asarray(error)))) > self.config.divergence_error_limit:
            self._divergence_detected = True
            return False
        if float(np.max(np.abs(np.asarray(output)))) > self.config.maximum_peak * 2.0:
            self._divergence_detected = True
            return False
        return True


class DeviceSafety:
    """Detect invalid device identifiers and callback dropouts."""

    def __init__(self, config: AudioSafetyConfig) -> None:
        self.config = config
        self._last_callback_monotonic: float | None = None
        self._invalid_device = False
        self._dropout_detected = False

    @property
    def invalid_device(self) -> bool:
        return self._invalid_device

    @property
    def dropout_detected(self) -> bool:
        return self._dropout_detected

    def validate_device(
        self,
        device: int | str | None,
        available_device_ids: set[int] | None = None,
    ) -> bool:
        if device is None:
            self._invalid_device = False
            return True
        if isinstance(device, bool):
            self._invalid_device = True
            return False
        if isinstance(device, int):
            if available_device_ids is not None and device not in available_device_ids:
                self._invalid_device = True
                return False
            self._invalid_device = False
            return True
        if isinstance(device, str):
            valid = bool(device.strip())
            self._invalid_device = not valid
            return valid
        self._invalid_device = True
        return False

    def callback_seen(self) -> bool:
        now = time.monotonic()
        previous = self._last_callback_monotonic
        self._last_callback_monotonic = now
        if previous is None:
            self._dropout_detected = False
            return True
        elapsed = now - previous
        if elapsed > self.config.dropout_timeout_seconds:
            self._dropout_detected = True
            return False
        self._dropout_detected = False
        return True

    def reset(self) -> None:
        self._last_callback_monotonic = None
        self._invalid_device = False
        self._dropout_detected = False


class AudioSafetyController:
    """Central fail-safe controller for startup, gain, mute and shutdown."""

    def __init__(self, config: AudioSafetyConfig | None = None) -> None:
        self.config = config or AudioSafetyConfig()
        self.state = SafetyState.IDLE
        self._muted = True
        self._emergency_stop = False
        self._lock = threading.Lock()
        self.gain_ramp = GainRamp(
            start_gain=0.0,
            maximum_gain=self.config.maximum_gain,
            ramp_duration_seconds=self.config.ramp_duration_seconds,
        )
        self.physical_anc = PhysicalANCInterlock()
        self.adaptive_filter = AdaptiveFilterSafety(self.config)
        self.device = DeviceSafety(self.config)

    @property
    def muted(self) -> bool:
        with self._lock:
            return self._muted

    @property
    def emergency_stopped(self) -> bool:
        with self._lock:
            return self._emergency_stop

    def soft_clip(self, audio: np.ndarray) -> np.ndarray:
        values = np.asarray(audio)
        if values.size == 0:
            return np.asarray(values, copy=True)
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError("audio must contain floating-point samples")
        if not np.all(np.isfinite(values)):
            self.emergency_mute("non-finite audio samples")
            return np.zeros_like(values)

        peak = float(np.max(np.abs(values)))
        rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
        if peak > self.config.maximum_peak or rms > self.config.maximum_rms:
            result = np.tanh(values / self.config.soft_clip_peak)
            result *= self.config.soft_clip_peak
        else:
            result = np.asarray(values, copy=True)
        result = np.clip(
            result,
            -self.config.maximum_peak,
            self.config.maximum_peak,
        )
        return result

    def start_safe(self) -> bool:
        with self._lock:
            if self._emergency_stop:
                return False
            self.state = SafetyState.STARTING
            self._muted = True
        self.gain_ramp.mute()
        self.gain_ramp.set_target(self.config.startup_gain)
        with self._lock:
            self.state = SafetyState.ACTIVE
            self._muted = False
        return True

    def set_gain(self, gain: float) -> bool:
        if not np.isfinite(gain) or gain < 0 or gain > self.config.maximum_gain:
            self.emergency_mute("gain outside safe limits")
            return False
        with self._lock:
            if self._emergency_stop:
                return False
        self.gain_ramp.set_target(gain)
        return True

    def emergency_mute(self, reason: str = "emergency stop") -> None:
        del reason
        with self._lock:
            self._emergency_stop = True
            self._muted = True
            self.state = SafetyState.EMERGENCY_STOPPED
        self.gain_ramp.mute()
        self.physical_anc.disable()

    def reset_emergency_stop(self) -> bool:
        with self._lock:
            self._emergency_stop = False
            self._muted = True
            self.state = SafetyState.IDLE
        self.adaptive_filter.reset()
        self.device.reset()
        self.gain_ramp.mute()
        return True

    def safe_shutdown(self) -> None:
        self.gain_ramp.mute()
        self.physical_anc.disable()
        with self._lock:
            self._muted = True
            self.state = SafetyState.IDLE

    def validate_algorithm(
        self,
        weights: np.ndarray,
        error: object,
        output: object,
    ) -> bool:
        valid = self.adaptive_filter.validate(weights, error, output)
        if not valid:
            self.emergency_mute("adaptive filter divergence")
        return valid
