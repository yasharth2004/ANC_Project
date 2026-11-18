"""Real-time environmental-noise analyzer using capture and Matplotlib.

The microphone callback only copies samples into the capture module's queue.
All FFT, band, waveform and plotting work occurs outside the audio callback.
RMS and dBFS values are relative digital levels, not calibrated SPL.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Protocol

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from .audio_capture import AudioCapture, AudioCaptureConfig
    from .dsp.signal_metrics import peak_amplitude, rms, rms_dbfs
except ImportError:
    from audio_capture import AudioCapture, AudioCaptureConfig
    from dsp.signal_metrics import peak_amplitude, rms, rms_dbfs


LOGGER = logging.getLogger("hybridanc.noise_analyzer")
SUPPORTED_SAMPLE_RATES = (44_100, 48_000)
SUPPORTED_FFT_SIZES = (256, 512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True)
class NoiseAnalyzerConfig:
    input_device: int | str | None = None
    sample_rate: int = 48_000
    block_size: int = 512
    fft_size: int = 2048
    refresh_interval_ms: int = 100
    waveform_seconds: float = 2.0
    spectrogram_columns: int = 120

    def __post_init__(self) -> None:
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError("sample_rate must be 44100 or 48000 Hz")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.fft_size not in SUPPORTED_FFT_SIZES:
            raise ValueError(
                "fft_size must be one of 256, 512, 1024, 2048, 4096 or 8192"
            )
        if self.refresh_interval_ms < 20:
            raise ValueError("refresh_interval_ms must be at least 20")
        if self.waveform_seconds <= 0:
            raise ValueError("waveform_seconds must be positive")
        if self.spectrogram_columns < 20:
            raise ValueError("spectrogram_columns must be at least 20")


@dataclass(frozen=True)
class FrequencyBand:
    name: str
    low_hz: float
    high_hz: float


FREQUENCY_BANDS = (
    FrequencyBand("Sub-bass", 20.0, 60.0),
    FrequencyBand("Bass", 60.0, 250.0),
    FrequencyBand("Low mids", 250.0, 500.0),
    FrequencyBand("Mids", 500.0, 2_000.0),
    FrequencyBand("High mids", 2_000.0, 4_000.0),
    FrequencyBand("Presence", 4_000.0, 8_000.0),
    FrequencyBand("Brilliance", 8_000.0, 20_000.0),
)


@dataclass(frozen=True)
class NoiseAnalysisSnapshot:
    rms: float
    rms_dbfs: float
    peak: float
    dominant_frequency_hz: float
    waveform: np.ndarray
    spectrum_db: np.ndarray
    frequencies_hz: np.ndarray
    band_levels_db: dict[str, float]
    spectrogram_db: np.ndarray
    overflow_count: int


class _CaptureLike(Protocol):
    is_running: bool
    overflow_count: int

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self, timeout: float | None = None) -> object: ...
    def raise_if_failed(self) -> None: ...


class NoiseAnalyzer:
    """Continuously capture and analyze audio without plotting in callbacks."""

    def __init__(
        self,
        config: NoiseAnalyzerConfig | None = None,
        *,
        capture: _CaptureLike | None = None,
    ) -> None:
        self.config = config or NoiseAnalyzerConfig()
        self.capture = capture or AudioCapture(
            AudioCaptureConfig(
                device=self.config.input_device,
                sample_rate=self.config.sample_rate,
                block_size=self.config.block_size,
                channels=1,
                queue_depth=32,
            )
        )
        waveform_frames = max(
            self.config.fft_size,
            int(self.config.sample_rate * self.config.waveform_seconds),
        )
        self._waveform = np.zeros(waveform_frames, dtype=np.float32)
        self._waveform_position = 0
        self._fft_frames = np.zeros(
            self.config.fft_size,
            dtype=np.float32,
        )
        fft_columns = self.config.spectrogram_columns
        self._spectrogram = np.full(
            (self.config.fft_size // 2 + 1, fft_columns),
            -120.0,
            dtype=np.float32,
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._running = False
        self._snapshot = NoiseAnalysisSnapshot(
            rms=0.0,
            rms_dbfs=-np.inf,
            peak=0.0,
            dominant_frequency_hz=0.0,
            waveform=self._waveform.copy(),
            spectrum_db=np.full(
                self.config.fft_size // 2 + 1,
                -120.0,
                dtype=np.float32,
            ),
            frequencies_hz=np.fft.rfftfreq(
                self.config.fft_size,
                d=1.0 / self.config.sample_rate,
            ),
            band_levels_db={band.name: -np.inf for band in FREQUENCY_BANDS},
            spectrogram_db=self._spectrogram.copy(),
            overflow_count=0,
        )

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self.capture.start()
        with self._lock:
            self._running = True
        self._worker = threading.Thread(
            target=self._analysis_loop,
            name="HybridANC-NoiseAnalyzer",
            daemon=True,
        )
        self._worker.start()
        LOGGER.info(
            "Started noise analyzer at %d Hz with FFT size %d",
            self.config.sample_rate,
            self.config.fft_size,
        )

    def stop(self) -> None:
        if not self.is_running:
            worker = self._worker
            self._worker = None
            if worker is not None:
                worker.join(timeout=1.0)
            return

        self._stop_event.set()
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.join(timeout=1.0)
        self.capture.stop()
        with self._lock:
            self._running = False
        LOGGER.info("Stopped noise analyzer")

    def emergency_stop(self) -> None:
        self.stop()

    def close(self) -> None:
        self.emergency_stop()

    def __enter__(self) -> "NoiseAnalyzer":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.emergency_stop()

    def snapshot(self) -> NoiseAnalysisSnapshot:
        with self._lock:
            current = self._snapshot
            return NoiseAnalysisSnapshot(
                rms=current.rms,
                rms_dbfs=current.rms_dbfs,
                peak=current.peak,
                dominant_frequency_hz=current.dominant_frequency_hz,
                waveform=current.waveform.copy(),
                spectrum_db=current.spectrum_db.copy(),
                frequencies_hz=current.frequencies_hz.copy(),
                band_levels_db=dict(current.band_levels_db),
                spectrogram_db=current.spectrogram_db.copy(),
                overflow_count=current.overflow_count,
            )

    def _append_waveform(self, samples: np.ndarray) -> None:
        count = len(samples)
        length = len(self._waveform)
        if count >= length:
            np.copyto(self._waveform, samples[-length:])
            self._waveform_position = 0
            return

        end = self._waveform_position + count
        if end <= length:
            np.copyto(self._waveform[self._waveform_position:end], samples)
        else:
            first = length - self._waveform_position
            np.copyto(self._waveform[self._waveform_position:], samples[:first])
            np.copyto(self._waveform[:count - first], samples[first:])
        self._waveform_position = end % length

    def _ordered_waveform(self) -> np.ndarray:
        position = self._waveform_position
        if position == 0:
            return self._waveform.copy()
        return np.concatenate(
            (self._waveform[position:], self._waveform[:position])
        )

    def analyze_window(self, samples: np.ndarray) -> NoiseAnalysisSnapshot:
        """Perform FFT and band analysis for a mono float32/float64 window."""
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            raise ValueError("analysis window must not be empty")

        fft_size = self.config.fft_size
        if mono.size >= fft_size:
            fft_input = mono[-fft_size:].copy()
        else:
            fft_input = np.zeros(fft_size, dtype=np.float32)
            fft_input[-mono.size:] = mono

        np.copyto(self._fft_frames, fft_input)
        window = np.hanning(fft_size).astype(np.float32)
        windowed = self._fft_frames * window
        spectrum_complex = np.fft.rfft(windowed)
        spectrum_magnitude = np.abs(spectrum_complex)
        positive_floor = np.maximum(
            spectrum_magnitude,
            np.finfo(np.float32).eps,
        )
        spectrum_db = 20.0 * np.log10(
            positive_floor / fft_size
        )

        frequencies = np.fft.rfftfreq(
            fft_size,
            d=1.0 / self.config.sample_rate,
        )
        minimum_bin = 1
        dominant_index = int(np.argmax(spectrum_magnitude[minimum_bin:])) + minimum_bin
        dominant_frequency = float(frequencies[dominant_index])

        band_levels: dict[str, float] = {}
        nyquist = self.config.sample_rate / 2.0
        for band in FREQUENCY_BANDS:
            high_hz = min(band.high_hz, nyquist)
            mask = (frequencies >= band.low_hz) & (frequencies <= high_hz)
            if not np.any(mask):
                band_levels[band.name] = -np.inf
                continue
            band_magnitude = spectrum_magnitude[mask]
            band_rms = float(
                np.sqrt(np.mean(np.square(band_magnitude, dtype=np.float64)))
            )
            band_levels[band.name] = float(
                20.0 * np.log10(
                    max(band_rms, np.finfo(np.float32).eps) / fft_size
                )
            )

        self._spectrogram = np.roll(self._spectrogram, -1, axis=1)
        self._spectrogram[:, -1] = spectrum_db

        waveform = self._ordered_waveform()
        peak = peak_amplitude(mono)
        rms_value = rms(mono)
        dbfs = rms_dbfs(mono)

        snapshot = NoiseAnalysisSnapshot(
            rms=float(rms_value),
            rms_dbfs=float(dbfs),
            peak=float(peak),
            dominant_frequency_hz=dominant_frequency,
            waveform=waveform,
            spectrum_db=spectrum_db.copy(),
            frequencies_hz=frequencies.copy(),
            band_levels_db=band_levels,
            spectrogram_db=self._spectrogram.copy(),
            overflow_count=int(getattr(self.capture, "overflow_count", 0)),
        )
        with self._lock:
            self._snapshot = snapshot
        return snapshot

    def _analysis_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                block = self.capture.read(timeout=0.05)
            except TimeoutError:
                try:
                    self.capture.raise_if_failed()
                except Exception as exc:
                    LOGGER.error("Noise analyzer capture failed: %s", exc)
                continue
            except Exception as exc:
                LOGGER.error("Noise analyzer read failed: %s", exc)
                continue

            try:
                samples = np.asarray(block.samples, dtype=np.float32).reshape(-1)
                self._append_waveform(samples)
                self.analyze_window(samples)
            except Exception as exc:
                LOGGER.error("Noise analysis failed: %s", exc)
            finally:
                block.release()


def create_sine_demo(
    sample_rate: int = 48_000,
    duration_seconds: float = 1.0,
) -> np.ndarray:
    """Generate a one-kHz sine for offline analyzer demonstrations."""
    frames = int(sample_rate * duration_seconds)
    time_axis = np.arange(frames, dtype=np.float32) / sample_rate
    return 0.25 * np.sin(2.0 * np.pi * 1000.0 * time_axis)


def run_visualization(analyzer: NoiseAnalyzer) -> None:
    """Run Matplotlib outside the audio callback."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
        constrained_layout=True,
    )
    waveform_axis = axes[0, 0]
    spectrum_axis = axes[0, 1]
    spectrogram_axis = axes[1, 0]
    level_axis = axes[1, 1]

    snapshot = analyzer.snapshot()
    total_seconds = len(snapshot.waveform) / analyzer.config.sample_rate
    time_axis = np.linspace(-total_seconds, 0.0, len(snapshot.waveform))
    waveform_line, = waveform_axis.plot(time_axis, snapshot.waveform, lw=0.7)
    waveform_axis.set_title("Time-domain waveform")
    waveform_axis.set_xlabel("Seconds")
    waveform_axis.set_ylabel("Relative amplitude")
    waveform_axis.set_ylim(-1.0, 1.0)
    waveform_axis.grid(True, alpha=0.2)

    spectrum_line, = spectrum_axis.plot(
        snapshot.frequencies_hz,
        snapshot.spectrum_db,
        lw=0.8,
    )
    spectrum_axis.set_title("FFT magnitude spectrum")
    spectrum_axis.set_xlabel("Frequency (Hz)")
    spectrum_axis.set_ylabel("Relative magnitude (dBFS-like)")
    spectrum_axis.set_xlim(20.0, analyzer.config.sample_rate / 2.0)
    spectrum_axis.set_ylim(-120.0, 0.0)
    spectrum_axis.grid(True, alpha=0.2)

    spectrogram_image = spectrogram_axis.imshow(
        snapshot.spectrogram_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[
            0.0,
            analyzer.config.spectrogram_columns,
            0.0,
            analyzer.config.sample_rate / 2.0,
        ],
        vmin=-100.0,
        vmax=0.0,
    )
    spectrogram_axis.set_title("Spectrogram")
    spectrogram_axis.set_xlabel("Recent FFT columns")
    spectrogram_axis.set_ylabel("Frequency (Hz)")
    spectrogram_axis.set_ylim(20.0, analyzer.config.sample_rate / 2.0)
    fig.colorbar(
        spectrogram_image,
        ax=spectrogram_axis,
        label="Relative magnitude",
    )

    level_text = level_axis.text(
        0.02,
        0.98,
        "",
        transform=level_axis.transAxes,
        va="top",
        family="monospace",
    )
    level_axis.set_title("Live meter")
    level_axis.set_axis_off()

    def update(frame: int) -> tuple:
        del frame
        snapshot = analyzer.snapshot()
        snapshot_count = len(snapshot.waveform)
        total_seconds = snapshot_count / analyzer.config.sample_rate
        time_axis = np.linspace(-total_seconds, 0.0, snapshot_count)
        waveform_line.set_data(time_axis, snapshot.waveform)

        spectrum_line.set_data(
            snapshot.frequencies_hz,
            snapshot.spectrum_db,
        )
        spectrogram_image.set_data(snapshot.spectrogram_db)

        db_text = (
            "not calibrated"
            if not np.isfinite(snapshot.rms_dbfs)
            else f"{snapshot.rms_dbfs:.1f}"
        )
        lines = [
            f"RMS: {snapshot.rms:.6f}",
            f"RMS relative: {db_text} dBFS-like",
            f"Peak: {snapshot.peak:.6f}",
            f"Dominant: {snapshot.dominant_frequency_hz:.1f} Hz",
            f"Capture overflows: {snapshot.overflow_count}",
            "",
            "Frequency bands:",
        ]
        for name, level in snapshot.band_levels_db.items():
            band_text = "not calibrated" if not np.isfinite(level) else f"{level:.1f}"
            lines.append(f"{name:<12} {band_text}")
        level_text.set_text("\n".join(lines))
        return (
            waveform_line,
            spectrum_line,
            spectrogram_image,
            level_text,
        )

    animation = FuncAnimation(
        fig,
        update,
        interval=analyzer.config.refresh_interval_ms,
        blit=False,
        cache_frame_data=False,
    )
    fig._hybridanc_animation = animation
    plt.show()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Display live environmental microphone analysis"
    )
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--sample-rate", type=int, choices=[44_100, 48_000], default=48_000)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--fft-size", type=int, choices=list(SUPPORTED_FFT_SIZES), default=2048)
    parser.add_argument("--refresh-ms", type=int, default=100)
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    if args.input_device is not None:
        try:
            input_device: int | str = int(args.input_device)
        except ValueError:
            input_device = args.input_device
    else:
        input_device = None

    config = NoiseAnalyzerConfig(
        input_device=input_device,
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        fft_size=args.fft_size,
        refresh_interval_ms=args.refresh_ms,
    )
    analyzer = NoiseAnalyzer(config)
    start_time = time.monotonic()

    if args.duration > 0:
        def close_after_duration() -> None:
            while time.monotonic() - start_time < args.duration:
                time.sleep(0.05)
            try:
                plt.close("all")
            except Exception:
                pass

        timer = threading.Thread(
            target=close_after_duration,
            name="HybridANC-NoiseAnalyzer-Timer",
            daemon=True,
        )
        timer.start()

    try:
        analyzer.start()
        run_visualization(analyzer)
    except KeyboardInterrupt:
        print()
    except Exception as exc:
        LOGGER.error("Noise analyzer failed: %s", exc)
        return 1
    finally:
        analyzer.emergency_stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
