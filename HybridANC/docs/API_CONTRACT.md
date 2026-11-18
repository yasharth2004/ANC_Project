# HybridANC Local Desktop API

This API is a localhost integration contract between the React frontend and
the Python audio engine. It is not a public web service.

## Transport

- Bind: `127.0.0.1` only.
- Default port: `8765`.
- Control/status: HTTP JSON.
- Live metrics: Server-Sent Events (SSE).
- Raw audio: never transported as JSON.

The backend must remain independent of DSP implementations. DSP, capture,
playback and simulation classes are called by lifecycle state, not by HTTP.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Service availability |
| GET | `/api/v1/devices` | Inputs and outputs |
| GET | `/api/v1/status` | Backend and configuration status |
| PUT | `/api/v1/configuration` | Validate and update configuration |
| POST | `/api/v1/capture/start` | Start microphone capture |
| POST | `/api/v1/capture/stop` | Stop microphone capture |
| POST | `/api/v1/passthrough/start` | Start low-gain passthrough |
| POST | `/api/v1/passthrough/stop` | Stop passthrough |
| POST | `/api/v1/suppression` | Enable/disable microphone suppression |
| GET | `/api/v1/metrics/noise` | Latest aggregate RMS/peak metrics |
| GET | `/api/v1/metrics/latency` | Callback/overflow metrics |
| GET | `/api/v1/metrics/stream` | SSE status and metric stream |
| POST | `/api/v1/anc/simulation/start` | Start synthetic ANC simulation |
| POST | `/api/v1/anc/simulation/stop` | Stop synthetic ANC simulation |
| GET | `/api/v1/anc/results` | Latest simulation result |
| POST | `/api/v1/emergency-mute` | Immediate stop/mute |
| POST | `/api/v1/emergency-mute/reset` | Leave emergency-mute state |

## Configuration

`PUT /api/v1/configuration`

```json
{
  "inputDeviceId": "2",
  "outputDeviceId": "1",
  "sampleRate": 48000,
  "bufferSize": 512,
  "gain": 0.05,
  "suppressionEnabled": false,
  "simulationEnabled": false,
  "physicalANCEnabled": false
}
```

Validation:

- Sample rate: `44100` or `48000`.
- Buffer size: `128`, `256`, `512`, or `1024`.
- Gain: inclusive `0.0` to `0.25`.
- Device IDs: numeric IDs or service-defined string IDs.
- `physicalANCEnabled`: always rejected unless a future calibrated native
  authorization path is implemented.

## Status

`GET /api/v1/status`

```json
{
  "ok": true,
  "status": "running",
  "captureRunning": false,
  "passthroughRunning": false,
  "emergencyMuted": false,
  "lastError": null,
  "configuration": {},
  "noiseMetrics": {
    "rms": 0.0123,
    "peak": 0.041,
    "dbfs": -38.2
  },
  "latencyMetrics": {
    "callbackIntervalMs": 10.67,
    "captureOverflows": 0,
    "playbackUnderruns": 0
  },
  "ancResult": null
}
```

SSE sends the same status payload at roughly 250 ms:

```text
event: status
data: {"status":"running", "...": "..."}

```

## Device Contract

`GET /api/v1/devices`

```json
{
  "inputs": [],
  "outputs": [],
  "defaults": {
    "input": "2",
    "output": "1"
  }
}
```

The presence of a wired inline microphone does not imply a residual
error-microphone role.

## Error Responses

```json
{
  "ok": false,
  "error": "descriptive safe error"
}
```

- `400`: invalid JSON or configuration.
- `404`: unknown endpoint.
- `409`: operation conflicts with state or safety.
- `500`: backend operation failed.
- `503`: device/service unavailable.

## Safety

- Bind only to loopback.
- Reject non-loopback host bindings.
- Reject cross-origin browser origins except localhost.
- Reject payloads above 100 KB.
- Never send raw audio frames through JSON or SSE.
- An emergency mute stops playback/capture and disables simulation/physical flags.
- Physical ANC remains disabled.
