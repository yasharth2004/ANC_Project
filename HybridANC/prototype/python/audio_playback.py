"""Safe real-time mono/stereo playback using SoundDevice.

This module is independent of audio capture. It sends caller-provided blocks
to an output device and deliberately generates no ANC or phase-inverted signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
import logging
import threading
import time
from types import ModuleType
from typing import Callable

import numpy as np
import sounddevice as sd


LOGGER = logging.getLogger("hybridanc.audio_playback")

SUPPORTED_SAMPLE_RATES = (44_100, 48_000)
SUPPORTED_CHANNEL_COUNTS = (1, 2)
DEFAULT_LIMIT_DBFS = -1.0
DEFAULT_LIMIT = 10 ** (DEFAULT_LIMIT_DBFS / 20)


@dataclass(frozen=True)
class AudioPlaybackConfig:
    """Configuration for mono or stereo output."""

    sample_rate: int = 48_000
    block_size: int = 512
    channels: int = 2
    device: int | str | None = None
    queue_depth: int = 32
    gain: float = 0.05
    limiter_peak: float = DEFAULT_LIMIT
    dtype: str = "float32"
    silence_on_underrun: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError("sample_rate must be 44100 or 48000 Hz")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.channels not in SUPPORTED_CHANNEL_COUNTS:
            raise ValueError("channels must be 1 or 2")
        if self.queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        if not 0.0 <= self.gain <= 0.25:
            raise ValueError("gain must be between 0.0 and 0.25")
        if not 0.0 < self.limiter_peak <= 1.0:
            raise ValueError("limiter_peak must be greater than zero and at most 1.0")


class AudioPlaybackError(RuntimeError):
    """Base class for playback failures."""


class OutputDeviceDisconnectedError(AudioPlaybackError):
    """Raised when the selected output device is unavailable."""


class AudioPlayback:
    """Callback-based playback with gain and a conservative hard peak limiter."""

    def __init__(
        self,
        config: AudioPlaybackConfig | None = None,
        *,
        backend: ModuleType | None = None,
    ) -> None:
        self.config = config or AudioPlaybackConfig()
        self.backend = backend or sd
        self._stream: object | None = None
        self._lock = threading.Lock()
        self._running = False
        self._sequence = 0
        self._written_sequence = 0
        self._underrun_count = 0
        self._callback_error: BaseException | None = None
        self._last_callback_time_ns: int | None = None
        self._callback_intervals_ms = np.zeros(256, dtype=np.float64)
        self._callback_interval_index = 0
        self._callback_interval_count = 0
        self._gain = float(self.config.gain)
        self._limiter_peak = float(self.config.limiter_peak)
        self._input_queue: Queue[np.ndarray] = Queue(
            maxsize=self.config.queue_depth
        )
        self._silence = np.zeros(
            (self.config.block_size, self.config.channels),
            dtype=self.config.dtype,
        )
        self._scratch = np.empty_like(self._silence)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def underrun_count(self) -> int:
        with self._lock:
            return self._underrun_count

    @property
    def callback_error(self) -> BaseException | None:
        with self._lock:
            return self._callback_error

    @property
    def callback_interval_ms(self) -> float | None:
        if self._callback_interval_count == 0:
            return None
        index = (self._callback_interval_index - 1) % 256
        return float(self._callback_intervals_ms[index])

    @property
    def callback_interval_samples(self) -> np.ndarray:
        count = self._callback_interval_count
        if count == 0:
            return np.empty(0, dtype=np.float64)
        if count < 256:
            return self._callback_intervals_ms[:count].copy()
        return self._callback_intervals_ms.copy()

    @property
    def gain(self) -> float:
        with self._lock:
            return self._gain

    def set_gain(self, gain: float) -> None:
        """Set linear gain safely. Values outside 0.0-1.0 are rejected."""
        if not 0.0 <= gain <= 0.25:
            raise ValueError("gain must be between 0.0 and 0.25")
        with self._lock:
            self._gain = float(gain)

    def start(self) -> None:
        """Open and start the selected PortAudio output stream."""
        with self._lock:
            if self._running:
                return
            self._callback_error = None
            self._underrun_count = 0

        # Prefill a small silent output window so startup scheduling cannot
        # immediately underrun the callback. The window is not audio content.
        for _ in range(min(32, self.config.queue_depth)):
            try:
                self._input_queue.put_nowait(np.zeros_like(self._silence))
            except Full:
                break

        try:
            self.backend.check_output_settings(
                device=self.config.device,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                dtype=self.config.dtype,
            )
            stream = self.backend.OutputStream(
                device=self.config.device,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                blocksize=self.config.block_size,
                dtype=self.config.dtype,
                callback=self._audio_callback,
            )
            stream.start()
        except Exception as exc:
            LOGGER.error("Could not start audio playback: %s", exc)
            raise OutputDeviceDisconnectedError(
                f"Could not start output device: {exc}"
            ) from exc

        with self._lock:
            self._stream = stream
            self._running = True

        LOGGER.info(
            "Started %d-channel playback at %d Hz with block size %d",
            self.config.channels,
            self.config.sample_rate,
            self.config.block_size,
        )

    def stop(self, timeout: float = 2.0) -> None:
        """Immediately stop playback and close the output stream."""
        del timeout
        with self._lock:
            stream = self._stream
            self._stream = None
            self._running = False

        if stream is None:
            return

        try:
            stream.stop()
        except Exception as exc:
            LOGGER.warning("Output stream stop returned an error: %s", exc)
        try:
            stream.close()
        except Exception as exc:
            LOGGER.warning("Output stream close returned an error: %s", exc)

        LOGGER.info("Stopped audio playback")

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "AudioPlayback":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def write(self, samples: np.ndarray, timeout: float | None = None) -> None:
        """Copy a mono or stereo block into the bounded playback queue."""
        source = np.asarray(samples, dtype=np.float32)
        if source.ndim == 1:
            source = source.reshape(-1)
            if self.config.channels == 2:
                source = np.repeat(source[:, np.newaxis], 2, axis=1)
        elif source.ndim == 2:
            if source.shape[1] == 1 and self.config.channels == 2:
                source = np.repeat(source, 2, axis=1)
            elif source.shape[1] != self.config.channels:
                raise ValueError(
                    f"input has {source.shape[1]} channels; "
                    f"playback expects {self.config.channels}"
                )
        else:
            raise ValueError("samples must be a 1-D or 2-D NumPy array")

        frames = source.shape[0]
        if frames != self.config.block_size:
            raise ValueError(
                f"input must contain {self.config.block_size} frames; got {frames}"
            )

        block = np.empty(
            (self.config.block_size, self.config.channels),
            dtype=self.config.dtype,
        )
        if self.config.channels == 1:
            np.copyto(block[:, 0], source)
        else:
            np.copyto(block, source)

        try:
            self._input_queue.put(block, timeout=timeout)
        except Full as exc:
            raise TimeoutError("Playback queue is full") from exc

        with self._lock:
            self._written_sequence += 1

    def write_silence(self) -> None:
        """Best-effort queue insertion of one silence block."""
        try:
            block = np.zeros_like(self._silence)
            self._input_queue.put(block, timeout=0)
        except Full:
            pass

    def clear(self) -> None:
        """Remove all queued blocks, resulting in controlled silence."""
        while True:
            try:
                self._input_queue.get_nowait()
            except Empty:
                break

    def raise_if_failed(self) -> None:
        error = self.callback_error
        if error is not None:
            raise error

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Lightweight callback; copies, scales and limits only."""
        del time_info
        callback_time_ns = time.monotonic_ns()
        if self._last_callback_time_ns is not None:
            interval_ms = (
                callback_time_ns - self._last_callback_time_ns
            ) / 1_000_000.0
            index = self._callback_interval_index
            self._callback_intervals_ms[index] = interval_ms
            self._callback_interval_index = (index + 1) % 256
            self._callback_interval_count = min(
                self._callback_interval_count + 1,
                256,
            )
        self._last_callback_time_ns = callback_time_ns

        if status is not None:
            status_text = str(status)
            lowered = status_text.lower()
            if "underflow" in lowered:
                self._underrun_count += 1
            if "device" in lowered or "disconnect" in lowered:
                self._callback_error = OutputDeviceDisconnectedError(
                    f"Output device disconnected: {status_text}"
                )

        source: np.ndarray
        try:
            source = self._input_queue.get_nowait()
        except Empty:
            self._underrun_count += 1
            if self.config.silence_on_underrun:
                np.copyto(outdata, self._silence)
            else:
                np.copyto(outdata, self._silence)
            return

        np.copyto(self._scratch, source)
        gain = self._gain
        limiter_peak = self._limiter_peak

        np.multiply(self._scratch, gain, out=self._scratch)
        np.clip(
            self._scratch,
            -limiter_peak,
            limiter_peak,
            out=self._scratch,
        )
        np.copyto(outdata, self._scratch[:frames])


def main() -> int:
    """Run a short sine playback smoke test at a safe conservative level."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Play a short safe mono or stereo test tone"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-rate", type=int, choices=[44_100, 48_000], default=48_000)
    parser.add_argument("--channels", type=int, choices=[1, 2], default=2)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--gain", type=float, default=0.05)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    if args.device is not None:
        try:
            device: int | str = int(args.device)
        except ValueError:
            device = args.device
    else:
        device = None

    playback = AudioPlayback(
        AudioPlaybackConfig(
            device=device,
            sample_rate=args.sample_rate,
            block_size=args.block_size,
            channels=args.channels,
            gain=args.gain,
            queue_depth=64,
        )
    )

    blocks = int(args.duration * args.sample_rate / args.block_size)
    remainder = args.duration * args.sample_rate - blocks * args.block_size
    try:
        playback.start()
        print(
            f"Playing a conservative test tone at {args.sample_rate} Hz "
            f"with {args.channels} channel(s)"
        )
        for sequence in range(blocks):
            phase_block = np.arange(
                args.block_size,
                dtype=np.float32,
            ) + sequence * args.block_size
            tone = (
                0.25
                * np.sin(
                    2.0
                    * np.pi
                    * 440.0
                    * (phase_block / args.sample_rate),
                    dtype=np.float32,
                )
            )
            if args.channels == 2:
                tone = np.repeat(tone[:, np.newaxis], 2, axis=1)
            playback.write(tone, timeout=0.1)
            time.sleep(args.block_size / args.sample_rate)

        if remainder > 0:
            if args.channels == 1:
                remainder_block = np.zeros(
                    args.block_size,
                    dtype=np.float32,
                )
            else:
                remainder_block = np.zeros(
                    (args.block_size, 2),
                    dtype=np.float32,
                )
            playback.write(remainder_block, timeout=0.1)
        time.sleep(0.1)
    except (AudioPlaybackError, TimeoutError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print()
    finally:
        playback.stop()

    if playback.underrun_count:
        print(f"Underrun count: {playback.underrun_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
