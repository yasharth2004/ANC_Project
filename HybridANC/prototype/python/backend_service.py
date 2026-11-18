"""Localhost-only HTTP/SSE integration service for the HybridANC desktop.

The service owns backend lifecycle state and delegates capture, playback,
suppression and simulation to existing module APIs. SSE sends metrics only;
raw audio is never serialized to JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import logging
import threading
import time
from urllib.parse import urlparse

import numpy as np

try:
    from .audio_capture import AudioCapture, AudioCaptureConfig
    from .audio_devices import discover_audio_devices
    from .audio_passthrough import AudioPassthrough, AudioPassthroughConfig
    from .fxlms import FxLMSConfig, FxLMSFilter
    from .noise_suppression import (
        MicrophoneNoiseSuppressor,
        NoiseSuppressionConfig,
    )
except ImportError:
    from audio_capture import AudioCapture, AudioCaptureConfig
    from audio_devices import discover_audio_devices
    from audio_passthrough import AudioPassthrough, AudioPassthroughConfig
    from fxlms import FxLMSConfig, FxLMSFilter
    from noise_suppression import (
        MicrophoneNoiseSuppressor,
        NoiseSuppressionConfig,
    )


LOGGER = logging.getLogger("hybridanc.backend")
HOST = "127.0.0.1"
PORT = 8765


@dataclass(frozen=True)
class EngineConfiguration:
    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: int = 48000
    buffer_size: int = 512
    gain: float = 0.05
    suppression_enabled: bool = False
    simulation_enabled: bool = False
    physical_anc_enabled: bool = False

    def validate(self) -> None:
        if self.sample_rate not in (44100, 48000):
            raise ValueError("sample_rate must be 44100 or 48000")
        if self.buffer_size not in (128, 256, 512, 1024):
            raise ValueError("buffer_size must be 128, 256, 512 or 1024")
        if not np.isfinite(self.gain) or not 0.0 <= self.gain <= 0.25:
            raise ValueError("gain must be between 0.0 and 0.25")
        if self.physical_anc_enabled:
            raise ValueError(
                "physical ANC output is disabled; no uncalibrated playback is allowed"
            )


class EngineState:
    """Communication-independent audio lifecycle and metrics state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.config = EngineConfiguration()
        self.capture: AudioCapture | None = None
        self.passthrough: AudioPassthrough | None = None
        self.capture_running = False
        self.passthrough_running = False
        self.emergency_muted = False
        self.status = "idle"
        self.last_error: str | None = None
        self.capture_overflows = 0
        self.playback_underruns = 0
        self.input_rms = 0.0
        self.input_peak = 0.0
        self.input_dbfs = -float("inf")
        self.callback_interval_ms: float | None = None
        self.anc_result: dict[str, object] | None = None
        self.suppressor = MicrophoneNoiseSuppressor(
            NoiseSuppressionConfig(
                sample_rate=48000,
                fft_size=512,
                hop_size=128,
            )
        )

    def status_payload(self) -> dict[str, object]:
        with self._lock:
            return {
                "status": self.status,
                "captureRunning": self.capture_running,
                "passthroughRunning": self.passthrough_running,
                "emergencyMuted": self.emergency_muted,
                "lastError": self.last_error,
                "configuration": {
                    "inputDeviceId": (
                        str(self.config.input_device)
                        if self.config.input_device is not None
                        else ""
                    ),
                    "outputDeviceId": (
                        str(self.config.output_device)
                        if self.config.output_device is not None
                        else ""
                    ),
                    "sampleRate": self.config.sample_rate,
                    "bufferSize": self.config.buffer_size,
                    "gain": self.config.gain,
                    "suppressionEnabled": self.config.suppression_enabled,
                    "simulationEnabled": self.config.simulation_enabled,
                    "physicalANCEnabled": False,
                    "physicalCalibrationConfirmed": False,
                    "physicalCalibrationId": None,
                },
                "noiseMetrics": {
                    "rms": self.input_rms,
                    "peak": self.input_peak,
                    "dbfs": (
                        self.input_dbfs
                        if np.isfinite(self.input_dbfs)
                        else None
                    ),
                },
                "latencyMetrics": {
                    "callbackIntervalMs": self.callback_interval_ms,
                    "captureOverflows": self.capture_overflows,
                    "playbackUnderruns": self.playback_underruns,
                },
                "ancResult": self.anc_result,
            }

    def update_configuration(self, payload: dict[str, object]) -> dict[str, object]:
        aliases = {
            "inputDeviceId": "input_device",
            "outputDeviceId": "output_device",
            "bufferSize": "buffer_size",
            "suppressionEnabled": "suppression_enabled",
            "simulationEnabled": "simulation_enabled",
            "physicalANCEnabled": "physical_anc_enabled",
        }
        mapped: dict[str, object] = {}
        for key, value in payload.items():
            mapped_key = aliases.get(key, key)
            if mapped_key in (
                "input_device",
                "output_device",
                "sample_rate",
                "buffer_size",
                "gain",
                "suppression_enabled",
                "simulation_enabled",
                "physical_anc_enabled",
            ):
                mapped[mapped_key] = value

        for key in ("input_device", "output_device"):
            if key in mapped:
                value = mapped[key]
                if isinstance(value, str) and value.isdigit():
                    mapped[key] = int(value)
                elif value == "":
                    mapped[key] = None

        with self._lock:
            values = asdict(self.config)
            values.update(mapped)
            next_config = EngineConfiguration(**values)
            next_config.validate()
            self.config = next_config
            return self.status_payload()

    def devices(self) -> dict[str, object]:
        catalog = discover_audio_devices()
        inputs = []
        outputs = []
        for device in catalog.devices:
            if device.input_available:
                inputs.append({
                    "id": str(device.device_id),
                    "name": device.name,
                    "kind": "input",
                    "hostApi": device.host_api,
                    "channels": device.max_input_channels,
                    "defaultSampleRate": int(round(device.default_sample_rate)),
                    "available": device.usable,
                    "disconnected": not device.usable,
                })
            if device.output_available:
                outputs.append({
                    "id": str(device.device_id),
                    "name": device.name,
                    "kind": "output",
                    "hostApi": device.host_api,
                    "channels": device.max_output_channels,
                    "defaultSampleRate": int(round(device.default_sample_rate)),
                    "available": device.usable,
                    "disconnected": not device.usable,
                })
        return {
            "inputs": inputs,
            "outputs": outputs,
            "defaults": {
                "input": (
                    str(catalog.default_input_id)
                    if catalog.default_input_id is not None else None
                ),
                "output": (
                    str(catalog.default_output_id)
                    if catalog.default_output_id is not None else None
                ),
            },
        }

    def start_capture(self) -> dict[str, object]:
        with self._lock:
            if self.emergency_muted:
                raise RuntimeError("audio is emergency muted")
            if self.capture_running:
                return self.status_payload()
            capture = AudioCapture(
                AudioCaptureConfig(
                    device=self.config.input_device,
                    sample_rate=self.config.sample_rate,
                    block_size=self.config.buffer_size,
                    channels=1,
                    queue_depth=32,
                )
            )
            try:
                capture.start()
            except Exception as exc:
                self.last_error = str(exc)
                self.status = "error"
                raise
            self.capture = capture
            self.capture_running = True
            self.status = "running"
            threading.Thread(
                target=self._capture_metrics_loop,
                name="HybridANC-BackendMetrics",
                daemon=True,
            ).start()
            return self.status_payload()

    def _capture_metrics_loop(self) -> None:
        while True:
            with self._lock:
                capture = self.capture
                running = self.capture_running
            if not running or capture is None:
                return
            try:
                block = capture.read(timeout=0.10)
            except TimeoutError:
                continue
            try:
                samples = np.asarray(block.samples, dtype=np.float64)
                if self.config.suppression_enabled:
                    processed = self.suppressor.process(
                        samples.astype(np.float32)
                    )
                    samples = np.asarray(processed, dtype=np.float64)
                rms = float(
                    np.sqrt(np.mean(np.square(samples, dtype=np.float64)))
                )
                peak = float(np.max(np.abs(samples)))
                dbfs = (
                    -float("inf")
                    if rms <= np.finfo(np.float64).eps
                    else float(20.0 * np.log10(rms))
                )
                with self._lock:
                    self.input_rms = rms
                    self.input_peak = peak
                    self.input_dbfs = dbfs
                    self.capture_overflows = capture.overflow_count
                    self.callback_interval_ms = capture.callback_interval_ms
            finally:
                block.release()

    def stop_capture(self) -> dict[str, object]:
        with self._lock:
            capture = self.capture
            self.capture = None
            self.capture_running = False
        if capture is not None:
            capture.stop()
        self.status = "idle"
        return self.status_payload()

    def start_passthrough(self) -> dict[str, object]:
        with self._lock:
            if self.emergency_muted:
                raise RuntimeError("audio is emergency muted")
            if self.passthrough_running:
                return self.status_payload()
            passthrough = AudioPassthrough(
                AudioPassthroughConfig(
                    input_device=self.config.input_device,
                    output_device=self.config.output_device,
                    sample_rate=self.config.sample_rate,
                    block_size=self.config.buffer_size,
                    output_channels=2,
                    gain=self.config.gain,
                )
            )
            try:
                passthrough.start()
            except Exception as exc:
                self.last_error = str(exc)
                self.status = "error"
                raise
            self.passthrough = passthrough
            self.passthrough_running = True
            self.status = "passthrough"
            return self.status_payload()

    def stop_passthrough(self) -> dict[str, object]:
        with self._lock:
            passthrough = self.passthrough
            self.passthrough = None
            self.passthrough_running = False
        if passthrough is not None:
            passthrough.emergency_stop()
        self.status = "idle"
        return self.status_payload()

    def set_suppression(self, enabled: bool) -> dict[str, object]:
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        with self._lock:
            values = asdict(self.config)
            values["suppression_enabled"] = enabled
            self.config = EngineConfiguration(**values)
            if not enabled:
                self.suppressor.reset()
        return self.status_payload()

    def start_anc_simulation(self) -> dict[str, object]:
        with self._lock:
            if self.config.physical_anc_enabled:
                raise RuntimeError("physical ANC output is never enabled by simulation")
            values = asdict(self.config)
            values["simulation_enabled"] = True
            self.config = EngineConfiguration(**values)
            self.status = "simulation"

        frame_count = int(self.config.sample_rate * 0.25)
        time_axis = np.arange(frame_count, dtype=np.float64) / self.config.sample_rate
        reference = 0.25 * np.sin(2.0 * np.pi * 100.0 * time_axis)
        secondary = np.zeros(64, dtype=np.float64)
        secondary[16] = 1.0
        controller = FxLMSFilter(
            FxLMSConfig(filter_length=64, learning_rate=0.001),
            secondary,
            secondary,
        )
        result = controller.process(reference, reference)
        rms_before = float(np.sqrt(np.mean(reference * reference)))
        rms_after = float(np.sqrt(np.mean(result.residual_error * result.residual_error)))
        reduction_db = float(
            20.0 * np.log10(
                max(rms_after, np.finfo(np.float64).tiny)
                / max(rms_before, np.finfo(np.float64).tiny)
            )
        )
        with self._lock:
            self.anc_result = {
                "mode": "simulation",
                "physical_output": False,
                "algorithm": "FxLMS",
                "reference_hz": 100.0,
                "rms_before": rms_before,
                "rms_after": rms_after,
                "noise_reduction_db": reduction_db,
                "convergence_time_samples": result.convergence_time_samples,
                "divergence_detected": result.divergence_detected,
            }
        return self.status_payload()

    def stop_anc_simulation(self) -> dict[str, object]:
        with self._lock:
            values = asdict(self.config)
            values["simulation_enabled"] = False
            values["physical_anc_enabled"] = False
            self.config = EngineConfiguration(**values)
            self.status = "idle"
        return self.status_payload()

    def emergency_mute(self) -> dict[str, object]:
        with self._lock:
            passthrough = self.passthrough
            capture = self.capture
            self.passthrough = None
            self.capture = None
            self.passthrough_running = False
            self.capture_running = False
            self.emergency_muted = True
            self.status = "muted"
            values = asdict(self.config)
            values.update({
                "gain": 0.0,
                "simulation_enabled": False,
                "physical_anc_enabled": False,
            })
            self.config = EngineConfiguration(**values)
        if passthrough is not None:
            passthrough.emergency_stop()
        if capture is not None:
            capture.stop()
        return self.status_payload()

    def reset_emergency_mute(self) -> dict[str, object]:
        with self._lock:
            self.emergency_muted = False
            self.status = "idle"
        return self.status_payload()


def read_json(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise ValueError("invalid Content-Length") from exc
    if length <= 0 or length > 100_000:
        raise ValueError("request body must contain JSON and be at most 100 KB")
    try:
        payload = json.loads(handler.rfile.read(length).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


class HybridANCHandler(BaseHTTPRequestHandler):
    server_version = "HybridANC/0.1"
    protocol_version = "HTTP/1.1"

    def origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        return origin.startswith("http://127.0.0.1:") or origin.startswith(
            "http://localhost:"
        )

    def add_security_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin and self.origin_allowed():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")

    def send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.add_security_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("Frontend disconnected while receiving JSON")

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json(status, {"ok": False, "error": message})

    def do_OPTIONS(self) -> None:
        if not self.origin_allowed():
            self.send_error_json(HTTPStatus.FORBIDDEN, "origin not allowed")
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.add_security_headers()
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, PUT, OPTIONS",
        )
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if not self.origin_allowed():
            self.send_error_json(HTTPStatus.FORBIDDEN, "origin not allowed")
            return
        path = urlparse(self.path).path
        engine = self.server.engine  # type: ignore[attr-defined]

        if path == "/api/v1/health":
            self.send_json(
                HTTPStatus.OK,
                {"ok": True, "service": "HybridANC engine", "host": HOST},
            )
            return
        if path in (
            "/api/v1/devices",
            "/api/v1/devices/inputs",
            "/api/v1/devices/outputs",
        ):
            try:
                payload = engine.devices()
                if path == "/api/v1/devices/inputs":
                    payload = {
                        "inputs": payload["inputs"],
                        "outputs": [],
                        "defaults": payload["defaults"],
                    }
                elif path == "/api/v1/devices/outputs":
                    payload = {
                        "inputs": [],
                        "outputs": payload["outputs"],
                        "defaults": payload["defaults"],
                    }
                self.send_json(HTTPStatus.OK, payload)
            except Exception as exc:
                engine.last_error = str(exc)
                self.send_error_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    str(exc),
                )
            return
        if path == "/api/v1/status":
            self.send_json(
                HTTPStatus.OK,
                {"ok": True, **engine.status_payload()},
            )
            return
        if path in (
            "/api/v1/metrics/noise",
            "/api/v1/metrics/latency",
        ):
            payload = engine.status_payload()
            key = "noiseMetrics" if path.endswith("noise") else "latencyMetrics"
            self.send_json(
                HTTPStatus.OK,
                {"ok": True, key: payload[key]},
            )
            return
        if path == "/api/v1/anc/results":
            self.send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "mode": "simulation",
                    "result": engine.anc_result,
                },
            )
            return
        if path == "/api/v1/metrics/stream":
            self.serve_sse(engine)
            return
        self.send_error_json(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def do_PUT(self) -> None:
        if not self.origin_allowed():
            self.send_error_json(HTTPStatus.FORBIDDEN, "origin not allowed")
            return
        if urlparse(self.path).path != "/api/v1/configuration":
            self.send_error_json(HTTPStatus.NOT_FOUND, "unknown endpoint")
            return
        engine = self.server.engine  # type: ignore[attr-defined]
        try:
            payload = read_json(self)
            result = engine.update_configuration(payload)
            self.send_json(HTTPStatus.OK, {"ok": True, **result})
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            engine.last_error = str(exc)
            engine.status = "error"
            LOGGER.exception("Backend crash while updating configuration")
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "backend state unavailable",
            )

    def do_POST(self) -> None:
        if not self.origin_allowed():
            self.send_error_json(HTTPStatus.FORBIDDEN, "origin not allowed")
            return
        path = urlparse(self.path).path
        engine = self.server.engine  # type: ignore[attr-defined]
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = read_json(self) if length > 0 else {}
            if path == "/api/v1/capture/start":
                result = engine.start_capture()
            elif path == "/api/v1/capture/stop":
                result = engine.stop_capture()
            elif path == "/api/v1/passthrough/start":
                result = engine.start_passthrough()
            elif path == "/api/v1/passthrough/stop":
                result = engine.stop_passthrough()
            elif path == "/api/v1/suppression":
                if "enabled" not in payload:
                    raise ValueError("enabled is required")
                enabled = payload["enabled"]
                if not isinstance(enabled, bool):
                    raise ValueError("enabled must be a boolean")
                result = engine.set_suppression(enabled)
            elif path == "/api/v1/anc/simulation/start":
                result = engine.start_anc_simulation()
            elif path == "/api/v1/anc/simulation/stop":
                result = engine.stop_anc_simulation()
            elif path == "/api/v1/emergency-mute":
                result = engine.emergency_mute()
            elif path == "/api/v1/emergency-mute/reset":
                result = engine.reset_emergency_mute()
            else:
                self.send_error_json(HTTPStatus.NOT_FOUND, "unknown endpoint")
                return
            self.send_json(HTTPStatus.OK, {"ok": True, **result})
        except ValueError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except RuntimeError as exc:
            engine.last_error = str(exc)
            engine.status = "error"
            self.send_error_json(HTTPStatus.CONFLICT, str(exc))
        except Exception as exc:
            engine.last_error = str(exc)
            engine.status = "error"
            LOGGER.exception("Backend crash while handling operation")
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "backend operation failed",
            )

    def serve_sse(self, engine: EngineState) -> None:
        self.send_response(HTTPStatus.OK)
        self.add_security_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                payload = engine.status_payload()
                body = json.dumps(payload).encode("utf-8")
                self.wfile.write(
                    b"event: status\ndata: "
                    + body
                    + b"\n\n"
                )
                self.wfile.flush()
                time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("Frontend disconnected from metrics stream")

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)


class HybridANCServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        engine: EngineState,
    ) -> None:
        super().__init__(server_address, HybridANCHandler)
        self.engine = engine


def create_server(
    host: str = HOST,
    port: int = PORT,
    engine: EngineState | None = None,
) -> HybridANCServer:
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("HybridANC local service may bind only to localhost")
    if not 1024 <= port <= 65535:
        raise ValueError("port must be between 1024 and 65535")
    return HybridANCServer(
        (host, port),
        engine or EngineState(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the localhost HybridANC integration service"
    )
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    try:
        server = create_server(args.host, args.port)
    except ValueError as exc:
        LOGGER.error("Refusing unsafe server configuration: %s", exc)
        return 1

    LOGGER.info(
        "HybridANC local service listening at http://%s:%d",
        args.host,
        args.port,
    )
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        LOGGER.info("Stopping HybridANC local service")
    finally:
        server.engine.emergency_mute()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
