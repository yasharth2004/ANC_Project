"""Comprehensive safety tests for HybridANC audio output."""

import numpy as np
import pytest

from core.safety.audio_safety import (
    AdaptiveFilterSafety,
    AudioSafetyConfig,
    AudioSafetyController,
    DeviceSafety,
    GainRamp,
    PhysicalANCInterlock,
    SafetyState,
)


def test_maximum_amplitude_and_soft_clipping():
    safety = AudioSafetyController(
        AudioSafetyConfig(maximum_peak=0.5)
    )
    result = safety.soft_clip(
        np.array([2.0, -3.0, 0.25], dtype=np.float64)
    )

    assert np.max(np.abs(result)) <= 0.5
    assert np.max(np.abs(result)) > 0.0
    assert safety.emergency_stopped is False


def test_nonfinite_audio_immediately_emergency_mutes():
    safety = AudioSafetyController()
    result = safety.soft_clip(
        np.array([0.1, np.nan, np.inf])
    )

    assert not np.any(result)
    assert safety.emergency_stopped
    assert safety.muted
    assert safety.state == SafetyState.EMERGENCY_STOPPED


def test_emergency_mute_and_reset():
    safety = AudioSafetyController()
    assert safety.start_safe()
    safety.emergency_mute("test")

    assert safety.emergency_stopped
    assert safety.muted
    assert safety.physical_anc.enabled is False
    assert safety.gain_ramp.current_gain == 0.0
    assert safety.reset_emergency_stop()
    assert not safety.emergency_stopped


def test_physical_anc_disabled_by_default():
    interlock = PhysicalANCInterlock()
    assert interlock.enabled is False
    assert interlock.enable(True) is False


def test_physical_anc_requires_calibration_and_operator_confirmation():
    interlock = PhysicalANCInterlock()
    interlock.set_calibration(False)
    assert interlock.enable(True) is False

    interlock.set_calibration(True, "calibration-001")
    assert interlock.enable(False) is False
    assert interlock.enable(False, "calibration-001") is False
    assert interlock.enable(True, "wrong-id") is False
    assert interlock.enable(True, "calibration-001") is True

    interlock.set_calibration(False)
    assert interlock.enabled is False


def test_invalid_calibration_id_is_rejected():
    interlock = PhysicalANCInterlock()
    with pytest.raises(ValueError):
        interlock.set_calibration(True, "")


def test_gain_ramp_stays_within_limit():
    now = [0.0]
    clock = lambda: now[0]
    ramp = GainRamp(
        start_gain=0.05,
        maximum_gain=0.25,
        ramp_duration_seconds=0.25,
        clock=clock,
    )
    ramp.set_target(0.20)

    assert 0.05 <= ramp.current_gain <= 0.20
    now[0] = 0.125
    assert 0.05 <= ramp.current_gain <= 0.20
    now[0] = 0.25
    assert ramp.current_gain <= 0.25
    assert ramp.mute() == 0.0


def test_gain_limits_and_ramp_safety():
    safety = AudioSafetyController(
        AudioSafetyConfig(maximum_gain=0.25)
    )
    assert safety.start_safe()
    assert safety.set_gain(0.20)
    assert safety.set_gain(-0.1) is False
    assert safety.emergency_stopped


def test_adaptive_filter_divergence_detection():
    safety = AdaptiveFilterSafety(
        AudioSafetyConfig(
            divergence_weight_limit=10.0,
            divergence_error_limit=2.0,
        )
    )

    assert safety.validate(
        np.ones(8),
        np.array([0.1]),
        np.array([0.2]),
    )
    assert safety.validate(
        np.array([11.0]),
        np.array([0.1]),
        np.array([0.2]),
    ) is False
    assert safety.divergence_detected


def test_adaptive_filter_nan_infinity_and_shape():
    safety = AdaptiveFilterSafety(AudioSafetyConfig())

    assert safety.validate(
        np.array([np.nan]),
        np.array([0.1]),
        np.array([0.2]),
    ) is False
    assert safety.validate(
        np.array([[1.0, 2.0]]),
        np.array([0.1]),
        np.array([0.2]),
    ) is False
    assert safety.validate(
        np.ones(4097),
        np.array([0.1]),
        np.array([0.2]),
    ) is False


def test_audio_safety_controller_mutes_on_divergence():
    safety = AudioSafetyController()
    valid = safety.validate_algorithm(
        np.array([np.inf]),
        np.array([0.1]),
        np.array([0.2]),
    )

    assert valid is False
    assert safety.emergency_stopped
    assert safety.muted
    assert safety.physical_anc.enabled is False


def test_device_validation_handles_missing_disconnected_and_named_devices():
    device_safety = DeviceSafety(AudioSafetyConfig())

    assert device_safety.validate_device(None) is True
    assert device_safety.validate_device(2, {0, 2}) is True
    assert device_safety.validate_device(3, {0, 2}) is False
    assert device_safety.invalid_device is True
    assert device_safety.validate_device("  ") is False
    assert device_safety.validate_device("External Microphone") is True
    assert device_safety.validate_device(True) is False


def test_dropout_detection(monkeypatch):
    config = AudioSafetyConfig(dropout_timeout_seconds=0.01)
    device_safety = DeviceSafety(config)
    current = [100.0]
    monkeypatch.setattr(
        "core.safety.audio_safety.time.monotonic",
        lambda: current[0],
    )

    assert device_safety.callback_seen()
    current[0] += 0.020
    assert device_safety.callback_seen() is False
    assert device_safety.dropout_detected


def test_safe_startup_and_shutdown():
    safety = AudioSafetyController()
    assert safety.state == SafetyState.IDLE
    assert safety.start_safe()
    assert safety.state == SafetyState.ACTIVE
    assert safety.muted is False

    safety.safe_shutdown()
    assert safety.state == SafetyState.IDLE
    assert safety.muted
    assert safety.physical_anc.enabled is False
