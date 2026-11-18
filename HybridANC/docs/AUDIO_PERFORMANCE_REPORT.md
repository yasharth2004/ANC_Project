# HybridANC Real-Time Audio Performance Report

**Status:** Short baseline measured on the local macOS environment. Long-duration,
Duplex underrun and acoustic-loopback claims remain to be measured.

## Measured Environment

- Machine: macOS local environment
- Capture device: MacBook built-in microphone, device ID 2
- C++ benchmark compiler: AppleClang 21 Release build
- Capture matrix date: 2026-11-19
- Matrix duration: 2 seconds per configuration

## Callback Lock/History Optimization

Before optimization, capture and playback callbacks used Python locks and
growing callback-history lists. The implementation now uses:

- Preallocated NumPy capture/playback block buffers.
- Fixed 256-entry callback interval rings.
- No callback-side `threading.Lock`.
- No callback-side history list append/delete.
- Queue-based block handoff outside callback state.
- UI and backend DSP separated by existing threads/service state.

## Capture Baseline

| Sample rate | Block size | Blocks | Overflows | Jitter ms | CPU % | Status |
|---:|---:|---:|---:|---:|---:|---|
| 44100 | 64 | 1185 | 0 | 2.93 | Not recorded in matrix | completed |
| 44100 | 128 | 634 | 0 | 3.70 | Not recorded in matrix | completed |
| 44100 | 256 | 322 | 0 | 5.17 | Not recorded in matrix | completed |
| 44100 | 512 | 160 | 0 | 3.03 | Not recorded in matrix | completed |
| 48000 | 64 | 1393 | 0 | 3.33 | Not recorded in matrix | completed |
| 48000 | 128 | 701 | 0 | 4.42 | Not recorded in matrix | completed |
| 48000 | 256 | 341 | 0 | 5.15 | Not recorded in matrix | completed |
| 48000 | 512 | 175 | 0 | 0.20 | Not recorded in matrix | completed |

CPU values for these rows were not captured because the matrix's JSON report
did not include a stable process-CPU field in the first run.

## LMS Language Benchmark

Identical 64-tap LMS workload:

- 48,000 samples.
- 100 Hz sine reference and 0.8-scaled desired.
- 200 iterations.

| Implementation | Measured result |
|---|---|
| Python | 142.697678 s; 1.402 simulated-seconds/second |
| C++ AppleClang Release | 1.11427 s; 179.49 simulated-seconds/second |

The C++ executable is materially faster in this single measured workload.
This is not a claim for every algorithm, block size or future platform.

## Remaining Measurements

The following have not been validated by this report:

- 10-minute continuous capture/playback.
- 30-minute continuous capture/playback.
- Playback underrun matrix.
- Input/output clock drift.
- Device hot-disconnect/reconnect behavior.
- High-CPU scheduling stress.
- Acoustic round-trip latency.

Run the stability harness for the missing long durations:

```bash
hybridanc-stability --input-device 2 --duration 600
hybridanc-stability --input-device 2 --duration 1800
```

Run duplex latency tooling separately before any acoustic claim.

## Recommended Next Optimization

Measured evidence is sufficient to retain preallocated queues and remove
callback locks. Do not introduce C++ audio callbacks or sample-rate conversion
until a measured port target proves it necessary. The Python callback timing
must remain stable at selected block sizes before production DSP migration.
