# Controlled Physical ANC Experiment Report

**Status:** Template only; no measured result is assumed.

## 1. Experiment Identity

- Experiment ID:
- Date and time:
- Operator:
- Repository/software commit:
- Operating system and audio host:
- Configuration: [ ] A existing hardware [ ] B future measurement setup
- Controlled environment:
- Hearing-safety supervisor:
- Emergency-stop method:

## 2. Explicit Safety Authorization

- [ ] Earphones are not worn on a person's ears during this experiment.
- [ ] Output is limited to a previously verified low-level test range.
- [ ] A calibrated measurement path is available before any output test.
- [ ] Operator can stop output immediately.
- [ ] Non-linear or unstable output is not permitted.
- [ ] Anti-noise output is disabled unless this form is completed.

**No result is valid until the measurement and output levels are documented.**

## 3. Hardware Inventory

| Item | Model/ID | Input or output | Microphone role | Validated? |
|---|---|---|---|---|
| Laptop | | | | |
| Laptop microphone | | Reference only for A | | |
| Wired earphones | | Secondary/output source | | |
| Inline earphone microphone | | Record only; not assumed error mic | | |
| External reference microphone | | Near noise source | | |
| External measurement microphone | | Listening position | | |
| Amplifier/interface | | | | |
| Measurement chain calibration | | | | |

## 4. Audio Routing

- Capture device:
- Playback device:
- Capture channels:
- Playback channels:
- Sample rate:
- Block size:
- Effective duplex latency:
- Input/output clock relationship:
- OS audio effects enabled/disabled:
- AGC enabled/disabled:
- Echo cancellation enabled/disabled:
- Doppler/beamforming effects enabled/disabled:

## 5. Configuration A: Existing Hardware

**Objective:** determine whether existing hardware can support a controlled
measurement and a meaningful simulation.

- Reference input: laptop microphone.
- Experimental output: wired-earphone speakers, not worn by an operator.
- Residual measurement: unavailable unless a separate calibrated microphone
  measures the controlled location.
- Inline microphone: not assumed to be the residual error sensor.
- Acoustic path: unknown and variable.
- Physical-ANC conclusion:
  [ ] Not sufficient for validation
  [ ] Inadequate for quantitative residual claim
  [ ] Not tested

Record why the earphone microphone cannot be assumed to measure residual
sound pressure:

## 6. Configuration B: Future Measurement Setup

- Reference mic location:
- Reference mic distance from source:
- Earphone speaker location:
- Measurement mic location:
- Earphone/measurement geometry:
- Noise-source location:
- Room dimensions:
- Sound-absorbing treatment:
- External calibration file:

## 7. Secondary-Path Calibration

1. Start with playback and capture mute.
2. Verify the measurement microphone and low-level output range.
3. Record at least 5 seconds of ambient noise.
4. Record 5–10 repetitions of deterministic low-level excitation.
5. Export the known excitation and simultaneous measurement WAV files.
6. Estimate the secondary impulse response offline.
7. Repeat the calibration 3 times.
8. Reject calibration with unstable delay or high repeatability error.

Recorded calibration files:

| Trial | Excitation WAV | Measurement WAV | FIR length | Estimated delay | MSE | Result |
|---|---|---|---:|---:|---:|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

## 8. FxLMS Parameter Calibration

- Controller FIR length:
- Learning rate:
- Initial weights:
- Estimated secondary FIR source:
- Step-size sweep:
- Weight limit:
- Output limit:
- Convergence threshold:
- Divergence stop condition:
- Filtered-X sign/phase review:

No parameter is validated until repeatable residual measurements are available.

## 9. Controlled Low-Level Test Sequence

Use controlled, low-level noise sources. Do not use a person's ears as the
measurement target.

| Trial | Noise type | Frequency | Noise level | Input file | Before WAV | After WAV | Result |
|---|---|---|---|---|---|---|---|
| 1 | 100 Hz sine | | | | | | |
| 2 | Multi-tone | | | | | | |
| 3 | White noise | | | | | | |
| 4 | Speech recording | | | | | | |

## 10. Residual Noise Measurement

- Baseline record before controller activation:
- Controller residual record:
- Fixed phase-inversion control record:
- LMS/NLMS/FxLMS records:
- Reference microphone record:
- Error microphone record:
- Measurement-microphone calibration factor:

Metrics:

| Method | Before RMS | After RMS | Reduction dB | Delay | Convergence | Divergence | Processing |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | | | 0.00 | | | | |
| Fixed inversion | | | | | | | |
| LMS | | | | | | | |
| NLMS | | | | | | | |
| FxLMS | | | | | | | |

## 11. Repeatability

| Trial | Date/time | RMS before | RMS after | Reduction dB | Delay | Stability | Notes |
|---|---|---:|---:|---:|---:|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |

Calculate repeatability:

- Mean reduction:
- Standard deviation:
- Maximum increase:
- Frequency bands with gain:

## 12. Failures and Stop Conditions

- [ ] Uncontrolled output level
- [ ] Diverging coefficients
- [ ] Non-finite samples
- [ ] Device disconnection
- [ ] Callback dropouts
- [ ] Excessive residual increase
- [ ] Acoustic feedback
- [ ] Missing or drifting error measurement
- [ ] Operator stop

## 13. Measured Versus Simulated Results

### Measured

- Hardware IDs:
- Calibrated levels:
- IR peaks and delay:
- Repeatable residual measurements:
- Stable frequency bands:

### Simulated

- Synthetic secondary path:
- Synthetic reference/error signals:
- LMS/NLMS/FxLMS result:
- Assumptions not verified on hardware:

## 14. Final Verdict

- [ ] Insufficient hardware for physical ANC validation
- [ ] Simulation only
- [ ] Calibration repeatable
- [ ] Residual reduction repeatable
- [ ] No repeatable benefit
- [ ] Not safe to continue

**Statement of limitations:**

