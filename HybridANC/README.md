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

Run the dependency validator separately:

```bash
python scripts/validate_dependencies.py
```

Run tests with coverage:

```bash
python -m pytest --cov=prototype/python --cov-report=term-missing
```

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
