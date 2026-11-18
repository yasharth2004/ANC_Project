"""Generated-sine demonstrations for foundational DSP concepts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np

try:
    from .normalization import normalize
    from .signal_metrics import peak_amplitude, rms, zero_crossing_rate
    from .filters import apply_sos_filter, butter_lowpass_sos
    from .windows import hann_window
    from .frequency_analysis import dominant_frequencies
except ImportError:
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    from prototype.python.dsp.normalization import normalize
    from prototype.python.dsp.signal_metrics import peak_amplitude, rms, zero_crossing_rate
    from prototype.python.dsp.filters import apply_sos_filter, butter_lowpass_sos
    from prototype.python.dsp.windows import hann_window
    from prototype.python.dsp.frequency_analysis import dominant_frequencies


LOGGER = logging.getLogger("hybridanc.dsp.demo")


def generate_sine(
    frequency_hz: float,
    sample_rate: float,
    duration_seconds: float,
) -> np.ndarray:
    """Generate a finite sine wave: x[n] = A sin(2*pi*f*n/fs)."""
    if frequency_hz < 0:
        raise ValueError("frequency_hz cannot be negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    nyquist = sample_rate / 2.0
    if frequency_hz >= nyquist:
        raise ValueError("frequency_hz must be below the Nyquist frequency")

    frames = int(sample_rate * duration_seconds)
    time = np.arange(frames, dtype=np.float64) / sample_rate
    return 0.5 * np.sin(2.0 * np.pi * frequency_hz * time)


def run_demo() -> None:
    sample_rate = 48_000
    sine = generate_sine(1_000.0, sample_rate, 0.1)

    normalized = normalize(sine)
    filtered = apply_sos_filter(
        butter_lowpass_sos(sample_rate, 2_000.0, order=4),
        normalized,
    )
    window = hann_window(len(filtered))
    windowed = filtered * window

    print(f"Samples: {len(sine)}")
    print(f"RMS: {rms(sine):.6f}")
    print(f"Peak amplitude: {peak_amplitude(sine):.6f}")
    print(f"Zero-crossing rate: {zero_crossing_rate(sine):.6f}")
    print(f"Dominant frequency: {dominant_frequencies(windowed, sample_rate):.1f} Hz")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demonstrate foundational HybridANC DSP operations"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    run_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
