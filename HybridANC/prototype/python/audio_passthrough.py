"""Real-time microphone-to-earphone passthrough.

This module joins capture and playback. It does not implement ANC, phase
inversion, filtering or feedback suppression.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Protocol

import numpy as np

try:
    from .audio_capture import AudioCapture, AudioCaptureConfig
    from .audio_playback import AudioPlayback, AudioPlaybackConfig
except ImportError:
    from audio_capture import AudioCapture, AudioCaptureConfig
    from audio_playback import AudioPlayback, AudioPlaybackConfig


LOGGER = logging.getLogger("hybridanc.audio_passthrough")
SUPPORTED_SAMPLE_RATES = (44_100, 48_000)


@dataclass(frozen=True)
class AudioPassthroughConfig:
    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: int = 48_000
    block_size: int = 512
    output_channels: int = 2
    gain: float = 0.05
    queue_depth: int = 32

    def __post_init__(self) -> None:
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError("sample_rate must be 44100 or 48000 Hz")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.output_channels not in (1, 2):
            raise ValueError("output_channels must be 1 or 2")
        if not 0.0 <= self.gain <= 0.25:
            raise ValueError("passthrough gain must be between 0.0 and 0.25")
        if self.queue_depth <= 0:
            raise ValueError("queue_depth must be positive")


@dataclass(frozen=True)
class PassthroughStatus:
    running: bool
    input_device: int | str | None
    output_device: int | str | None
    sample_rate: int
    block_size: int
    output_channels: int
    gain: float
    input_rms: float
    input_dbfs: float
    capture_overflow_count: int
    playback_underrun_count: int
    last_error: str | None


class _CaptureLike(Protocol):
    is_running: bool
    overflow_count: int

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self, timeout: float | None = None) -> object: ...
    def raise_if_failed(self) -> None: ...


class _PlaybackLike(Protocol):
    is_running: bool
    underrun_count: int

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def set_gain(self, gain: float) -> None: ...
    def write(self, samples: np.ndarray, timeout: float | None = None) -> None: ...
    def write_silence(self) -> None: ...
    def clear(self) -> None: ...
    def raise_if_failed(self) -> None: ...


class AudioPassthrough:
    """Coordinates low-gain microphone capture and headphone playback."""

    def __init__(
        self,
        config: AudioPassthroughConfig | None = None,
        *,
        capture: _CaptureLike | None = None,
        playback: _PlaybackLike | None = None,
    ) -> None:
        self.config = config or AudioPassthroughConfig()
        self._capture_device = self.config.input_device
        self._output_device = self.config.output_device
        self._gain = float(self.config.gain)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._running = False
        self._input_rms = 0.0
        self._last_error: str | None = None

        self.capture = capture or AudioCapture(
            AudioCaptureConfig(
                device=self.config.input_device,
                sample_rate=self.config.sample_rate,
                block_size=self.config.block_size,
                channels=1,
                queue_depth=self.config.queue_depth,
            )
        )
        self.playback = playback or AudioPlayback(
            AudioPlaybackConfig(
                device=self.config.output_device,
                sample_rate=self.config.sample_rate,
                block_size=self.config.block_size,
                channels=self.config.output_channels,
                gain=self.config.gain,
                queue_depth=self.config.queue_depth,
            )
        )

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def input_rms(self) -> float:
        with self._lock:
            return self._input_rms

    @property
    def input_dbfs(self) -> float:
        rms = self.input_rms
        if rms <= np.finfo(np.float32).eps:
            return -np.inf
        return float(20.0 * np.log10(rms))

    @property
    def gain(self) -> float:
        with self._lock:
            return self._gain

    def set_gain(self, gain: float) -> None:
        if not 0.0 <= gain <= 0.25:
            raise ValueError("passthrough gain must be between 0.0 and 0.25")
        with self._lock:
            self._gain = float(gain)
        self.playback.set_gain(gain)

    def start(self) -> None:
        if self.is_running:
            return

        with self._lock:
            self._last_error = None
            self._input_rms = 0.0
        self._stop_event.clear()

        try:
            self.playback.start()
            self.capture.start()
        except Exception as exc:
            LOGGER.error("Could not start passthrough: %s", exc)
            with self._lock:
                self._last_error = str(exc)
            self.playback.stop()
            self.capture.stop()
            raise

        with self._lock:
            self._running = True
        self._worker = threading.Thread(
            target=self._bridge_loop,
            name="HybridANC-Passthrough",
            daemon=True,
        )
        self._worker.start()
        LOGGER.info(
            "Started passthrough: input=%s output=%s rate=%d block=%d",
            self._capture_device,
            self._output_device,
            self.config.sample_rate,
            self.config.block_size,
        )

    def stop(self) -> None:
        """Stop playback first, then capture and the bridge worker."""
        if not self.is_running:
            worker = self._worker
            self._worker = None
            if worker is not None:
                worker.join(timeout=1.0)
            return

        self._stop_event.set()
        try:
            self.playback.clear()
        except Exception:
            pass
        try:
            self.playback.stop()
        except Exception as exc:
            LOGGER.warning("Playback stop returned an error: %s", exc)
        try:
            self.capture.stop()
        except Exception as exc:
            LOGGER.warning("Capture stop returned an error: %s", exc)

        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.join(timeout=1.0)

        with self._lock:
            self._running = False
            self._input_rms = 0.0
        LOGGER.info("Stopped audio passthrough")

    def emergency_stop(self) -> None:
        """Mute and stop both streams immediately."""
        try:
            self.set_gain(0.0)
            self.playback.clear()
        except Exception as exc:
            LOGGER.error("Emergency mute cleanup failed: %s", exc)
        self.stop()

    def close(self) -> None:
        self.emergency_stop()

    def __enter__(self) -> "AudioPassthrough":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.emergency_stop()

    def switch_devices(
        self,
        input_device: int | str | None,
        output_device: int | str | None,
    ) -> None:
        was_running = self.is_running
        gain = self.gain
        if was_running:
            self.emergency_stop()

        self._capture_device = input_device
        self._output_device = output_device
        if isinstance(self.capture, AudioCapture) and isinstance(
            self.playback,
            AudioPlayback,
        ):
            self.capture = AudioCapture(
                AudioCaptureConfig(
                    device=input_device,
                    sample_rate=self.config.sample_rate,
                    block_size=self.config.block_size,
                    channels=1,
                    queue_depth=self.config.queue_depth,
                )
            )
            self.playback = AudioPlayback(
                AudioPlaybackConfig(
                    device=output_device,
                    sample_rate=self.config.sample_rate,
                    block_size=self.config.block_size,
                    channels=self.config.output_channels,
                    gain=gain,
                    queue_depth=self.config.queue_depth,
                )
            )
        else:
            self.playback.set_gain(gain)
        with self._lock:
            self._gain = float(gain)
        if was_running:
            self.start()

    def status(self) -> PassthroughStatus:
        capture_overflows = getattr(self.capture, "overflow_count", 0)
        playback_underruns = getattr(self.playback, "underrun_count", 0)
        with self._lock:
            rms = self._input_rms
            dbfs = (
                -np.inf
                if rms <= np.finfo(np.float32).eps
                else float(20.0 * np.log10(rms))
            )
            return PassthroughStatus(
                running=self._running,
                input_device=self._capture_device,
                output_device=self._output_device,
                sample_rate=self.config.sample_rate,
                block_size=self.config.block_size,
                output_channels=self.config.output_channels,
                gain=self._gain,
                input_rms=rms,
                input_dbfs=dbfs,
                capture_overflow_count=int(capture_overflows),
                playback_underrun_count=int(playback_underruns),
                last_error=self._last_error,
            )

    def _bridge_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                block = self.capture.read(timeout=0.05)
            except TimeoutError:
                try:
                    self.capture.raise_if_failed()
                    self.playback.raise_if_failed()
                except Exception as exc:
                    LOGGER.error("Passthrough audio error: %s", exc)
                    with self._lock:
                        self._last_error = str(exc)
                continue
            except Exception as exc:
                LOGGER.error("Capture bridge read failed: %s", exc)
                with self._lock:
                    self._last_error = str(exc)
                continue

            try:
                samples = block.samples
                rms = float(
                    np.sqrt(
                        np.mean(
                            np.square(samples, dtype=np.float64),
                            dtype=np.float64,
                        )
                    )
                )
                with self._lock:
                    self._input_rms = rms

                if self.gain > 0.0:
                    self.playback.write(samples, timeout=0.05)
                else:
                    self.playback.write_silence()
            except TimeoutError:
                LOGGER.warning("Playback queue is full; clearing playback buffer")
                self.playback.clear()
            except Exception as exc:
                LOGGER.error("Passthrough bridge failed: %s", exc)
                with self._lock:
                    self._last_error = str(exc)
            finally:
                block.release()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Low-gain real-time microphone-to-earphone passthrough"
    )
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--sample-rate", type=int, choices=[44_100, 48_000], default=48_000)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--channels", type=int, choices=[1, 2], default=2)
    parser.add_argument("--gain", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    def parse_device(value: str | None) -> int | str | None:
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return value

    passthrough = AudioPassthrough(
        AudioPassthroughConfig(
            input_device=parse_device(args.input_device),
            output_device=parse_device(args.output_device),
            sample_rate=args.sample_rate,
            block_size=args.block_size,
            output_channels=args.channels,
            gain=args.gain,
        )
    )

    start_time = time.monotonic()
    try:
        passthrough.start()
        print("Passthrough active. Use low microphone-to-speaker separation.")
        while passthrough.is_running:
            time.sleep(0.5)
            status = passthrough.status()
            db_text = (
                "silent"
                if status.input_dbfs == -np.inf
                else f"{status.input_dbfs:.1f} dBFS"
            )
            print(
                f"\rRMS={status.input_rms:.6f} {db_text} "
                f"gain={status.gain:.3f}",
                end="",
                flush=True,
            )
            if args.duration > 0 and time.monotonic() - start_time >= args.duration:
                break
    except KeyboardInterrupt:
        print()
    except Exception as exc:
        LOGGER.error("Passthrough failed: %s", exc)
        return 1
    finally:
        passthrough.emergency_stop()

    print("\nPassthrough stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
