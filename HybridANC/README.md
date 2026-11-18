# HybridANC

HybridANC is a modular audio research project for:

- software microphone noise suppression
- offline and simulated active noise cancellation
- future physical ANC calibration and validation

The initial implementation is a Python prototype. It does not generate
anti-noise or claim physical cancellation. ANC algorithms and output safety
controls will be implemented only after the audio and measurement architecture
is established.

## Repository Layout

```text
HybridANC/
  apps/
    desktop/frontend/       React/TypeScript desktop interface
    desktop/native-bridge/  Native UI/audio-engine bridge
    android/                Future Kotlin/Oboe/AAudio adapter
    ios/                    Future Swift/AVAudioEngine adapter
  core/
    audio/                  Platform-independent audio contracts
    dsp/                    Production DSP implementation
    anc/                    LMS, NLMS, FxLMS and path models
    safety/                 Output limits and safety interlocks
  prototype/
    python/                 Initial NumPy/SciPy/SoundDevice prototype
    experiments/            Offline notebooks, recordings and plots
  tests/
    audio/                  Device and stream tests
    dsp/                    DSP unit and integration tests
    anc/                    Adaptive-filter and path-model tests
    integration/            Cross-module tests
  docs/                     Engineering documentation and reports
  scripts/                Reproducible setup and validation scripts
```

## Important Files

- `requirements.txt`: minimal Python prototype dependencies.
- `pyproject.toml`: package metadata, script entry point and test configuration.
- `prototype/python/config.py`: validated typed configuration for devices,
  DSP format, logging and safety limits.
- `prototype/python/main.py`: environment discovery and microphone capture
  entry point. It never generates playback or anti-noise.
- `scripts/validate_dependencies.py`: imports and reports every required
  runtime and test dependency.
- `prototype/python/audio_devices.py`: mocked-testable PortAudio device
  discovery, selection and format validation.
- `prototype/python/audio_capture.py`: callback-based mono real-time capture
  with bounded preallocated buffers and no callback-side I/O.
- `prototype/python/audio_playback.py`: independent mono/stereo playback,
  gain control and conservative peak limiting.
- `prototype/python/audio_passthrough.py`: low-gain real-time microphone-to
  earphone passthrough coordinator. It has no ANC algorithms.
- `prototype/python/dsp/`: normalized foundational NumPy/SciPy DSP helpers.
- `prototype/python/noise_analyzer.py`: real-time Matplotlib environmental
  microphone analyzer. Relative dBFS-like levels are not calibrated SPL.
- `tests/dsp/test_fft_smoke.py`: basic NumPy FFT smoke test with no production
  audio-processing implementation.
- `.gitignore`: excludes virtual environments, generated audio, plots and
  Python caches.
- `prototype/experiments/`: experimental code and assets, kept separate from
  future production implementation.

## Python Setup

### macOS/Linux

```bash
cd HybridANC
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/validate_dependencies.py
python -m pytest
python prototype/python/main.py --mode check-only
```

### Windows PowerShell

```powershell
cd HybridANC
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/validate_dependencies.py
python -m pytest
python prototype/python/main.py --mode check-only
```

### Install as an Editable Package

```bash
python -m pip install -e ".[dev]"
hybridanc --mode check-only
```

On Windows:

```powershell
python -m pip install -e ".[dev]"
hybridanc --mode check-only
```

## Prototype Commands

List audio devices and verify imports:

```bash
python prototype/python/main.py --mode check-only
```

Perform a short microphone capture and print the RMS level:

```bash
python prototype/python/main.py --mode microphone-capture --duration 1.0
```

Select a named or indexed input device:

```bash
python prototype/python/main.py --mode microphone-capture --input-device 0 --duration 2.0
```

Enable debug logging:

```bash
python prototype/python/main.py --log-level DEBUG --mode check-only
```

Print a readable audio-device table:

```bash
python prototype/python/audio_devices.py
hybridanc-devices
python prototype/python/main.py --mode check-only --list-devices
```

Capture real-time mono microphone blocks:

```bash
python prototype/python/audio_capture.py --sample-rate 48000 --block-size 512 --duration 3
hybridanc-capture --sample-rate 48000 --block-size 512 --duration 3
```

Block consumers should release each block after use so its backing buffer can
be reused:

```python
from prototype.python.audio_capture import AudioCapture, AudioCaptureConfig

capture = AudioCapture(
    AudioCaptureConfig(sample_rate=48000, block_size=512)
)

with capture:
    block = capture.read(timeout=1.0)
    try:
        process_block_without_copying(block.samples)
    finally:
        block.release()
```

    Run a conservative playback test tone:

    ```bash
    python prototype/python/audio_playback.py --device 1 --channels 2 --sample-rate 48000 --block-size 512 --duration 1
    hybridanc-playback --device 1 --channels 2 --sample-rate 48000 --block-size 512 --duration 1
    ```

    Playback is compatible with mono or stereo output, and with 44100 Hz or
    48000 Hz sample rates. Use the device module to confirm that the selected
    device advertises the required channel count and supported sample rate.
    Capture and playback are not automatically synchronized: their sample rates
    and block sizes must be matched explicitly.

## Foundational DSP

Run the generated-sine demonstration:

```bash
python prototype/python/dsp/demo.py
python -m prototype.python.dsp.demo
```

The package in `prototype/python/dsp/` provides:

- Peak and RMS amplitude measurements.
- Zero-crossing rate.
- Causal moving average.
- Butterworth digital filtering.
- Hann, Hamming and Blackman windows.
- Real FFT and frequency-axis analysis.
- Per-channel normalization.

### DSP Concepts

- **Digital samples:** individual amplitude values sampled at discrete time indices.
- **Sampling frequency (`fs`):** samples per second. A 48 kHz signal contains
  48,000 samples per second.
- **Nyquist frequency:** `fs / 2`, the highest frequency represented without
  aliasing under ideal band-limited sampling.
- **Amplitude:** signal magnitude at a digital sample. RMS and peak amplitude
  summarize amplitude across a frame.
- **Frequency:** cycles per second, measured in Hz.
- **Phase:** the position of a sinusoid within its cycle, measured in radians or
  degrees.
- **Time-domain representation:** amplitude-versus-sample or amplitude-versus-time.
- **Frequency-domain representation:** magnitude and phase versus frequency,
  obtained through an FFT.

## Audio Passthrough

The passthrough coordinator combines capture and playback with low-gain,
bounded bridging. It does not implement ANC, phase inversion or noise
cancellation.

Run with the built-in/laptop microphone:

```bash
python prototype/python/audio_passthrough.py \
  --input-device 2 \
  --output-device 1 \
  --gain 0.05 \
  --duration 10
```

Run with a wired-headset inline microphone:

```bash
python prototype/python/audio_passthrough.py \
  --input-device 0 \
  --output-device 1 \
  --gain 0.05 \
  --duration 10
```

Use the installed entry point:

```bash
hybridanc-passthrough --input-device 2 --output-device 1 --gain 0.05
```

The default gain is 5 percent. It is deliberately capped at 25 percent.
Playback still applies a conservative `-1 dBFS` peak limiter.

Run the dependency validator separately:

```bash
python scripts/validate_dependencies.py
```

Run tests with coverage:

```bash
python -m pytest --cov=prototype/python --cov-report=term-missing
```

## Environmental Noise Analyzer

Run the live Matplotlib analyzer:

```bash
hybridanc-noise --input-device 2 --sample-rate 48000 --fft-size 2048 --refresh-ms 100
```

Direct execution also works:

```bash
python prototype/python/noise_analyzer.py --input-device 2 --duration 30
```

The display includes:

- Time-domain waveform.
- One-sided FFT magnitude spectrum.
- Rolling spectrogram.
- Relative RMS level.
- Peak amplitude.
- Dominant frequency.
- Frequency-band relative levels.
- Capture overflow count.

RMS and dBFS-like values describe the microphone signal in digital units.
They are not calibrated environmental SPL values and must not be reported as
dB SPL without a microphone sensitivity calibration and reference measurement.

## Latency Measurement

Measure the digital capture, processing and playback queue at all required
block sizes and sample rates:

```bash
hybridanc-latency \
  --input-device 2 \
  --output-device 1 \
  --duration 5 \
  --json-report docs/latency-report.json \
  --csv-report docs/latency-report.csv
```

The matrix tests:

- 44.1 kHz and 48 kHz.
- 64, 128, 256 and 512-frame blocks.
- Capture/processing/playback callbacks and bridge processing.
- CPU process-load estimate.
- Input overflow and output underrun counts.

Routing latency is not acoustic latency. It measures capture timestamp to
bridge/playback scheduling under the current software and device topology.
It cannot establish sound-pressure delay at a listener's ear.

Attempt controlled hardware chirp correlation with:

```bash
hybridanc-latency --input-device 2 --output-device 1 --loopback --duration 5
```

A real acoustic loopback requires a known playback-to-microphone path and
normally a wired or instrumented setup. Ordinary laptop speakers and the built-in
microphone may add uncontrolled room delay and reflections. `N/A` values mean no
reliable samples were observed; `null` acoustic round-trip values mean no
validated loopback correlation was available.

## Digital Filtering

Plot original and causally filtered generated signals:

```bash
hybridanc-filters --type lowpass --sample-rate 48000 --order 4
hybridanc-filters --type highpass --sample-rate 48000 --order 4
hybridanc-filters --type bandpass --sample-rate 48000 --order 4
hybridanc-filters --type notch --sample-rate 48000 --order 2
```

The generated demonstration contains:

- 50 Hz interference.
- 100 Hz low-frequency tone.
- 1 kHz tone.
- Broadband high-frequency noise.

Streaming usage:

```python
from prototype.python.digital_filters import lowpass

filter_object = lowpass(48000, 5000, order=4)
filtered_block = filter_object.process(audio_block)
filter_object.reset()
```

Digital filters delay and reshape a signal:

- Amplitude: frequencies inside/at the passband are retained or attenuated
  according to the filter response.
- Phase: causal IIR filters introduce frequency-dependent phase delay.
- Order: higher order gives a steeper transition but greater delay and ringing
  potential.
- Streaming state: each block continues from the previous block without
  zero-phase `filtfilt` reversal.

## Microphone Noise Suppression

Run the original-versus-processed microphone demonstration:

```bash
hybridanc-suppress --strength 1.0
```

Direct execution:

```bash
python prototype/python/noise_suppression.py --strength 1.0
```

Streaming Python usage:

```python
from prototype.python.noise_suppression import (
    MicrophoneNoiseSuppressor,
    NoiseSuppressionConfig,
)

suppressor = MicrophoneNoiseSuppressor(
    NoiseSuppressionConfig(
        sample_rate=48000,
        fft_size=512,
        hop_size=128,
        suppression_strength=1.0,
    )
)

suppressor.calibrate_noise_profile(noise_only_audio)
processed_block = suppressor.process(microphone_block)

suppressor.set_bypass(True)
original_block = suppressor.process(microphone_block)
```

The calibrator should receive noise-only microphone audio. Suppression strength
is adjustable from `0.0` to `2.0`; higher values remove more noise but may
distort speech. The spectral floor and temporal-mask smoothing control residual
noise and abrupt changes.

Measured indicators include:

- Input and output broadband SNR.
- Processing latency in milliseconds.
- Difference-based speech distortion indicators.
- Peak and RMS amplitudes.

### Limitations

- This software changes microphone samples. It does not reduce acoustic sound
  pressure at the ear and is not physical headphone ANC.
- The prototype is causal, but FFT hop output is a simplified first-stage
  overlap implementation rather than an overlap-add production algorithm.
- Spectral subtraction and Wiener estimates can introduce artifacts, especially
  on low-SNR speech or rapidly changing noise.
- Noise calibration may be incomplete for non-stationary interference.
- Results are not universal microphone SPL or intelligibility measurements.
- A real subjective speech test and objective speech-quality model are required
  before production claims.

## ANC Simulation Laboratory

Run offline ideal and mismatched cancellation experiments:

```bash
hybridanc-anc-lab \
  --sample-rate 48000 \
  --duration 0.5 \
  --experiment B_theoretical_inverted \
  --json-report docs/anc-experiment.json
```

Direct execution:

```bash
python prototype/python/anc_simulation.py --experiment D_phase_mismatch
```

The laboratory provides:

- 100 Hz sine generation.
- 100 Hz inverted anti-noise.
- Amplitude, phase and sample-delay mismatch.
- Multi-frequency and seeded white-noise generation.
- WAV recording loading.
- Time-domain residual plots.
- Before/after frequency spectra.
- RMS-before, RMS-after and attenuation metrics.

Attenuation below zero dB means the residual RMS is lower than the original.
The laboratory never opens playback or sends anti-noise to headphones.

### Mathematical Cancellation vs. Physical ANC

Ideal mathematical cancellation adds equal and opposite samples in software.
It is not evidence of reduced sound pressure near a human ear.

Physical ANC additionally requires:

- A measured playback-to-ear/acoustic secondary path.
- A calibrated reference and error-sensor topology.
- Output, playback and capture latency measurements.
- Controlled acoustic experiments and hearing-safety limits.

## Secondary-Path Modeling

Run the offline synthetic secondary-path laboratory:

```bash
hybridanc-secondary-path \
  --experiment A_simulated_secondary_path \
  --sample-rate 48000 \
  --json-report docs/secondary-path.json
```

The module models:

```text
y[n] = sum(k=0..M-1) h[k] * x[n-k]
```

- `x[n]`: synthetic controlled excitation.
- `h[k]`: impulse-response taps.
- `y[n]`: modeled measurement location.
- `M`: FIR length.

It supports delay, amplitude scale, resonant frequency response and additive
measurement noise. It estimates an impulse response from least squares:

```text
h_hat = argmin_h ||Xh-y||^2
```

Synthetic secondary paths are not measurements of a real earphone. Future
hardware calibration must use controlled low-level excitation, a specified
measurement microphone, and a validated measurement location.

## Filtered-X LMS

Run the offline FxLMS laboratory:

```bash
hybridanc-fxlms \
  --experiment stationary_100hz \
  --filter-length 64 \
  --learning-rate 0.001 \
  --secondary-delay 16 \
  --json-report docs/fxlms-experiment.json
```

The FxLMS algorithm uses:

```text
x_f[n] = S_hat(z) * x[n]
y[n] = w[n]^T x[n]
e[n] = d[n] - S_true(z) * y[n]
w[n+1] = w[n] + mu * e[n] * x_f[n]
```

Variables:

- `x[n]`: reference signal from a reference microphone role.
- `w[n]`: adaptive FIR controller coefficients.
- `y[n]`: calculated anti-noise controller output.
- `S_true(z)`: simulated physical secondary path.
- `S_hat(z)`: estimated secondary path.
- `x_f[n]`: secondary-path-filtered reference vector.
- `d[n]`: primary residual arriving at the simulated error location.
- `e[n]`: simulated residual error.
- `mu`: learning-rate step size.

The implementation records residual RMS, coefficient history, convergence time,
divergence state and processing time. It compares no cancellation, fixed phase
inversion, LMS, NLMS and FxLMS using identical offline arrays.

Reference and error signals are distinct variables. An ordinary wired
earphone inline microphone must not be assumed to be an error microphone:
it may measure communications voice or a cable-earpiece location rather than
residual sound pressure.

FxLMS remains an offline mathematical simulation. It does not open a playback
device or enable anti-noise output without measured hardware calibration.

## LMS Adaptive Filtering

Run an offline LMS experiment:

```bash
hybridanc-lms \
  --experiment constant_frequency \
  --filter-length 64 \
  --learning-rate 0.01 \
  --json-report docs/lms-experiment.json
```

Available experiments:

- `constant_frequency`
- `changing_amplitude`
- `multi_frequency`
- `white_noise_reference`

Streaming-style direct usage:

```python
from prototype.python.lms import LMSConfig, LMSFilter

adaptive = LMSFilter(
    LMSConfig(
        filter_length=64,
        learning_rate=0.01,
        weight_limit=100.0,
        output_limit=100.0,
    )
)

for reference_sample, desired_sample in zip(reference, desired):
    output, error = adaptive.update(reference_sample, desired_sample)
```

### LMS Mathematics

The FIR vector is ordered from newest to oldest sample:

```text
x[n] = [x[n], x[n-1], ..., x[n-N+1]]^T
```

- `w[n]`: current coefficient vector.
- `d[n]`: known desired/reference target.
- `y[n] = w[n]^T x[n]`: filter output.
- `e[n] = d[n] - y[n]`: residual error.
- `mu`: learning-rate step size.
- `w[n+1] = w[n] + mu * e[n] * x[n]`: coefficient update.

The learning rate trades convergence speed against stability. A large `mu`
can overshoot and diverge, especially with high-energy or correlated input.
The implementation bounds output and coefficient magnitudes and marks
divergence if those limits are reached.

Coefficient history and error history are retained for convergence analysis.
The fixed-filter benchmark uses least-squares FIR coefficients on the complete
offline recording.

### LMS Limitations

The basic LMS implementation has no measured acoustic secondary path and does
not implement FxLMS, phase compensation or physical safety control. It must
not be described as a complete physical headphone ANC solution.

## NLMS Adaptive Filtering

Run an offline NLMS/LMS comparison:

```bash
hybridanc-nlms \
  --experiment changing_amplitude \
  --filter-length 64 \
  --step-size 0.5 \
  --epsilon 1e-6 \
  --json-report docs/nlms-experiment.json
```

Streaming block usage:

```python
from prototype.python.nlms import NLMSConfig, NLMSFilter

adaptive = NLMSFilter(
    NLMSConfig(
        filter_length=64,
        step_size=0.5,
        epsilon=1e-6,
    )
)

output, error = adaptive.process_block(
    reference_block,
    desired_block,
)
```

### NLMS Normalization

For FIR vector `x[n]` and coefficient vector `w[n]`:

```text
P[n] = x[n]^T x[n]
y[n] = w[n]^T x[n]
e[n] = d[n] - y[n]
w[n+1] = w[n] + mu * e[n] * x[n] / (epsilon + P[n])
```

- `epsilon` prevents division by zero for silent or very low-energy input.
- `mu` is a normalized step size, typically in `(0, 1]`.
- Normalization makes coefficient updates less sensitive to reference amplitude.
- It can improve convergence under changing input amplitude, but it cannot
  compensate for phase delay, acoustic path mismatch or unknown hardware latency.

The comparison and plots use identical reference and desired arrays for LMS and
NLMS. These are offline mathematical experiments, not complete physical ANC.

## Experimental ANC Comparison Dashboard

Run the separate Matplotlib comparison dashboard:

```bash
hybridanc-anc-dashboard \
  --algorithm all \
  --noise-type sine \
  --noise-frequency 100 \
  --noise-amplitude 0.5 \
  --sample-rate 48000 \
  --duration 1 \
  --filter-length 64 \
  --learning-rate 0.001 \
  --secondary-delay 16 \
  --secondary-path delayed_lead
```

Export all simulated samples and metric summaries:

```bash
hybridanc-anc-dashboard \
  --noise-type multi_sine \
  --duration 1 \
  --csv-output docs/anc-comparison.csv \
  --json-output docs/anc-comparison.json
```

The dashboard compares fixed phase inversion, LMS, NLMS and FxLMS against one
shared original noise array. It displays waveforms, anti-noise samples,
residual waveforms and spectra, residual RMS, noise reduction, convergence
history and measured processing cost.

The results come only from the current Python simulation run. The dashboard
does not display physical measurement claims or open audio playback.

## Acoustic-Performance Evaluation

Evaluate prerecorded before/after microphone recordings:

```bash
hybridanc-anc-evaluation \
  --before baseline.wav \
  --after residual.wav \
  --trial-name controlled_trial_1 \
  --sample-rate 48000 \
  --measurement-location listening_position \
  --gain-setting unchanged \
  --before-trials 3 \
  --after-trials 3 \
  --csv-output docs/anc-evaluation.csv \
  --json-output docs/anc-evaluation.json
```

The module reports:

- Before/after RMS.
- Noise reduction in dB using
  `20 * log10(RMS_before / RMS_after)`.
- Frequency-band RMS and attenuation.
- Before/after FFT spectra.
- Convergence time.
- Stability and relative RMS drift.
- Repeat-trial status.
- CSV and JSON experiment records.

Before/after measurements must use the same microphone position, gain,
acoustic environment and recording format. Electrical RMS is not absolute
sound-pressure level unless a traceable microphone calibration factor and
reference are available.

Repeatability notes and confidence limitations are included with every
evaluation. A single before/after pair cannot establish repeatability.

## Audio Safety and Physical ANC Interlock

The safety controller provides:

- Maximum peak and RMS limits.
- Soft clipping followed by hard peak clipping.
- Linear output gain ramping.
- Immediate emergency mute.
- A physical-ANC enable/disable interlock.
- Adaptive-filter coefficient, error and output validation.
- NaN and infinity rejection.
- Callback-dropout detection.
- Invalid-device checks.
- Safe startup and shutdown.

ANC is disabled by default. Simulation and plotting never open physical
headphone output. Physical ANC can be enabled only after a validated
calibration ID is assigned and the caller explicitly confirms operator
activation with that ID.

### Emergency Shutdown

Call `emergency_mute()` or `safe_shutdown()` for immediate output muting.
Emergency mute disables the physical-ANC interlock, sets gain to zero and
latches state to `EMERGENCY_STOPPED`. Normal startup requires
`reset_emergency_stop()`.

Non-finite samples, unsafe gain, coefficient divergence, invalid output,
dropouts or unavailable devices can trigger emergency mute.

## Local Desktop Integration

Start the localhost Python engine:

```bash
python prototype/python/backend_service.py --host 127.0.0.1 --port 8765
```

The API contract is [API_CONTRACT.md](/Users/yasharthkesarwani/Downloads/ANC_Project/HybridANC/docs/API_CONTRACT.md).
Control endpoints use JSON; live metrics use SSE. Raw audio is not transmitted
through JSON/SSE.

Start the React frontend in HTTP mode:

```bash
cd apps/desktop/frontend
cp .env.example .env
npm run dev
```

Without `VITE_HYBRIDANC_API_URL`, the frontend continues using its mock service.

## Real-Time Stability Benchmark

Run a measured capture stability matrix:

```bash
hybridanc-stability \
  --input-device 2 \
  --duration 10 \
  --sample-rates 44100 48000 \
  --block-sizes 64 128 256 512 \
  --json-output docs/audio-stability.json
```

The harness reports callback interval average, jitter, minimum/maximum,
capture blocks, observed overflows, process CPU estimate and device/status
errors. It does not fabricate 10-minute or 30-minute continuous results.
Use `--duration 600` or `--duration 1800` only for an actual run.

Playback underruns and output clock mismatch require an actual duplex
capture/playback session and controlled environment. The existing latency
probe distinguishes digital routing timing from acoustic round trip.

## macOS Microphone Permissions

Microphone capture requires access in **System Settings > Privacy & Security >
Microphone**.

If an app or terminal cannot access the microphone:

1. Add or enable the terminal application, VS Code, or Python launcher under
   the macOS microphone privacy settings.
2. Quit and restart that application.
3. Unplug and reconnect a wired headset.
4. Confirm the selected input in the audio device output.
5. If the device entry is missing, reset the Terminal or VS Code microphone
   permission and reopen the application.

In some macOS and PortAudio combinations, a USB or wired microphone may require
Core Audio permissions, a matching sample rate, or a separate input-device
selection. The prototype reports unavailable devices without activating
playback or ANC.

### macOS Passthrough Testing

1. Run `hybridanc-devices` and select valid input/output IDs.
2. Start with `--gain 0.05`, the built-in microphone, and wired headphones.
3. Avoid placing the microphone near the headphones; even low gain can produce
   feedback in a closed acoustic loop.
4. Press Control-C for emergency stop. The CLI also stops streams during normal
   process shutdown.
5. Test the wired inline microphone with its own input ID after confirming that
   macOS exposes it separately.
6. Test switching IDs with `AudioPassthrough.switch_devices()`.
7. Run a 10-minute test and review overflow/underrun counters and device errors.

An inline microphone and wired output should not be assumed to share a clock,
channel layout, or simultaneous device topology. If macOS does not expose
independent input and output entries, run only a mode supported by the system.

## Controlled Physical ANC Experiment

HybridANC provides an offline experiment preparation and recording-analysis
workflow. It never starts audio playback or activates anti-noise.

Create a detailed report:

```bash
hybridanc-physical-anc template \
  --output docs/PHYSICAL_ANC_EXPERIMENT_REPORT.md
```

Analyze prerecorded controlled excitation and measurement WAV files:

```bash
hybridanc-physical-anc calibrate \
  --configuration B_future_measurement \
  --excitation calibration-excitation.wav \
  --measurement calibration-measurement.wav \
  --sample-rate 48000 \
  --fir-length 256 \
  --json-output docs/calibration.json \
  --csv-output docs/calibration.csv
```

Compare prerecorded baseline and residual recordings:

```bash
hybridanc-physical-anc residual \
  --before baseline.wav \
  --after residual.wav \
  --sample-rate 48000 \
  --json-output docs/residual.json
```

Configuration A uses only the laptop microphone and wired-earphone speakers.
It is not sufficient for a reliable physical residual measurement because
there is no separate, properly positioned error microphone. The inline
earphone microphone must not be assumed to measure the ear-canal or
listening-position residual.

Configuration B requires a reference microphone near the noise source,
a measured secondary speaker location and a separate measurement microphone
near the listening position, with synchronized audio and controlled acoustics.

Do not place uncalibrated anti-noise output near or on a person's ears.
The experiment module records only an offline estimate and report state.

## Safety

The prototype:

- does not activate audio playback;
- does not use an inline microphone as an assumed ANC error microphone;
- does not implement adaptive ANC algorithms;
- applies no uncalibrated anti-noise output;
- requires future physical ANC implementations to start with a separate,
  validated calibration state.

## Future C++ Integration

The configuration fields in `prototype/python/config.py` define stable initial
DSP contract values: sample rate, block size, channel count and FFT size.
Future C++ code under `core/` should expose matching interfaces and retain
platform-neutral frame buffers. Native bridges and desktop/mobile applications
must remain above that core rather than implementing DSP themselves.
