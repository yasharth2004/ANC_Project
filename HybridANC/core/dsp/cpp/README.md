# HybridANC C++ DSP Core

The C++ core is callback-safe DSP infrastructure. It has no React, Python,
audio-device, file-I/O or JUCE dependency.

## Modules

- FIR processing for mono and stereo.
- Preallocated FFT magnitude calculation.
- RMS and peak calculation.
- LMS, NLMS and FxLMS.
- Secondary FIR filtering.
- Hard/soft-style bounded output limiter.
- Caller-owned scratch buffers.

## Real-Time Rules

- Constructors and setters allocate/preallocate outside callbacks.
- Callback methods do not allocate on the C++ heap.
- Callback methods do not lock, log, perform file I/O or call platform APIs.
- Input pointers and block lengths are validated.
- Non-finite samples are rejected or muted by the limiter.
- Adaptive weights and outputs need safety-layer limits around callback use.

The public types currently use `std::vector` for state. Allocation is allowed
during construction/reset setup, not in callback processing.

## macOS Build

```bash
cd core/dsp/cpp
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DHYBRIDANC_BUILD_TESTS=ON \
  -DHYBRIDANC_BUILD_BENCHMARK=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure
./build/hybridanc_dsp_benchmark
```

With Xcode tools:

```bash
xcode-select --install
```

JUCE and Eigen are intentionally not dependencies of the initial core. Add
them only after a measured use case requires them.

## Python Benchmark

From the repository root:

```bash
python scripts/benchmark_lms.py
```

The C++ benchmark executable runs the same 64-tap, 48 kHz-equivalent 48,000-
sample sine experiment for 200 iterations. Processing should be compared under
the same release build, optimization flags and machine load.

No performance claim is made until both executable timings are measured.
