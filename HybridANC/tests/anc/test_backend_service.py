"""Integration tests for the localhost HybridANC HTTP/SSE backend."""

import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from prototype.python.backend_service import (
    EngineState,
    create_server,
)


@pytest.fixture
def engine_server():
    server = create_server("127.0.0.1", 18765, EngineState())
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    thread.start()
    yield server, "http://127.0.0.1:18765"
    server.engine.emergency_mute()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def request_json(base_url, method, path, payload=None):
    body = None
    headers = {"Origin": "http://127.0.0.1:5173"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(
        base_url + path,
        data=body,
        headers=headers,
        method=method,
    )
    with urlopen(request, timeout=3.0) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_health_status_and_devices(engine_server):
    _, base_url = engine_server

    _, health = request_json(base_url, "GET", "/api/v1/health")
    _, status = request_json(base_url, "GET", "/api/v1/status")
    _, devices = request_json(base_url, "GET", "/api/v1/devices")

    assert health["ok"]
    assert status["status"] == "idle"
    assert "inputs" in devices
    assert "outputs" in devices


def test_configuration_validation(engine_server):
    _, base_url = engine_server

    _, result = request_json(
        base_url,
        "PUT",
        "/api/v1/configuration",
        {
            "sampleRate": 48000,
            "bufferSize": 512,
            "gain": 0.05,
        },
    )
    assert result["ok"]
    assert result["configuration"]["sampleRate"] == 48000

    request = Request(
        base_url + "/api/v1/configuration",
        data=json.dumps({"gain": 1.0}).encode("utf-8"),
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Content-Type": "application/json",
        },
        method="PUT",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=3.0)
    assert error.value.code == 400


def test_physical_anc_never_starts(engine_server):
    _, base_url = engine_server

    request = Request(
        base_url + "/api/v1/configuration",
        data=json.dumps({"physicalANCEnabled": True}).encode("utf-8"),
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Content-Type": "application/json",
        },
        method="PUT",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=3.0)
    assert error.value.code == 400


def test_suppression_and_simulation_controls(engine_server):
    _, base_url = engine_server

    _, suppression = request_json(
        base_url,
        "POST",
        "/api/v1/suppression",
        {"enabled": True},
    )
    assert suppression["configuration"]["suppressionEnabled"]

    _, simulation = request_json(
        base_url,
        "POST",
        "/api/v1/anc/simulation/start",
        {},
    )
    assert simulation["status"] == "simulation"
    assert simulation["configuration"]["physicalANCEnabled"] is False

    _, stopped = request_json(
        base_url,
        "POST",
        "/api/v1/anc/simulation/stop",
        {},
    )
    assert stopped["status"] == "idle"


def test_emergency_mute_is_latched(engine_server):
    _, base_url = engine_server

    _, muted = request_json(
        base_url,
        "POST",
        "/api/v1/emergency-mute",
        {},
    )
    assert muted["emergencyMuted"]
    assert muted["configuration"]["gain"] == 0.0

    _, reset = request_json(
        base_url,
        "POST",
        "/api/v1/emergency-mute/reset",
        {},
    )
    assert reset["emergencyMuted"] is False


def test_noise_and_latency_metrics_shapes(engine_server):
    _, base_url = engine_server

    _, noise = request_json(
        base_url,
        "GET",
        "/api/v1/metrics/noise",
    )
    _, latency = request_json(
        base_url,
        "GET",
        "/api/v1/metrics/latency",
    )

    assert "rms" in noise["noiseMetrics"]
    assert "callbackIntervalMs" in latency["latencyMetrics"]


def test_localhost_origin_is_rejected(engine_server):
    _, base_url = engine_server
    request = Request(
        base_url + "/api/v1/status",
        headers={"Origin": "https://example.com"},
        method="GET",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=3.0)
    assert error.value.code == 403
