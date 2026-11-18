"""Thread-safe real-time microphone capture using SoundDevice.

No plotting, disk I/O or DSP processing is performed in the audio callback.
Captured blocks are exposed to consumers through a bounded queue.
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


LOGGER = logging.getLogger("hybridanc.audio_capture")

SUPPORTED_SAMPLE_RATES = (44_100, 48_000)


@dataclass(frozen=True)
class AudioCaptureConfig:
    """Configuration for mono microphone capture."""

    sample_rate: int = 48_000
    block_size: int = 512
    channels: int = 1
    device: int | str | None = None
    queue_depth: int = 32
    buffer_count: int = 64
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                "sample_rate must be 44100 or 48000 Hz"
            )
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.channels != 1:
            raise ValueError("mono capture requires channels=1")
        if self.queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        if self.buffer_count < self.queue_depth:
            raise ValueError("buffer_count must be at least queue_depth")


class AudioCaptureError(RuntimeError):
    """Base class for capture failures."""


class DeviceDisconnectedError(AudioCaptureError):
    """Raised when the input device is lost or unavailable."""


class InputOverflowError(AudioCaptureError):
    """Recorded overflow condition reported by PortAudio."""


@dataclass
class CapturedBlock:
    """A reusable mono audio block returned to a consuming module."""

    samples: np.ndarray
    frames: int
    sample_rate: int
    sequence: int
    capture_time: float
    _owner: "AudioCapture | None" = None

    def release(self) -> None:
        """Return the backing array to the capture buffer pool."""
        if self._owner is not None:
            owner = self._owner
            self._owner = None
            owner._release_buffer(self.samples)
            owner._release_block(self)


class AudioCapture:
    """Callback-based, mono, real-time microphone capture."""

    def __init__(
        self,
        config: AudioCaptureConfig | None = None,
        *,
        backend: ModuleType | None = None,
    ) -> None:
        self.config = config or AudioCaptureConfig()
        self.backend = backend or sd
        self._stream: object | None = None
        self._lock = threading.Lock()
        self._running = False
        self._sequence = 0
        self._overflow_count = 0
        self._callback_error: BaseException | None = None
        self._last_callback_time_ns: int | None = None
        self._callback_intervals_ms = np.zeros(256, dtype=np.float64)
        self._callback_interval_index = 0
        self._callback_interval_count = 0
        self._blocks: Queue[CapturedBlock] = Queue(
            maxsize=self.config.queue_depth
        )
        self._buffer_pool: Queue[np.ndarray] = Queue(
            maxsize=self.config.buffer_count
        )
        self._block_pool: Queue[CapturedBlock] = Queue(
            maxsize=self.config.buffer_count
        )

        for _ in range(self.config.buffer_count):
            buffer = np.empty(
                self.config.block_size,
                dtype=self.config.dtype,
            )
            self._buffer_pool.put(buffer)
            self._block_pool.put(
                CapturedBlock(
                    samples=buffer,
                    frames=0,
                    sample_rate=self.config.sample_rate,
                    sequence=0,
                    capture_time=0.0,
                )
            )

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def overflow_count(self) -> int:
        with self._lock:
            return self._overflow_count

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

    def start(self) -> None:
        """Open and start the PortAudio input stream."""
        with self._lock:
            if self._running:
                return
            self._callback_error = None
            self._overflow_count = 0

        try:
            self.backend.check_input_settings(
                device=self.config.device,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                dtype=self.config.dtype,
            )
            stream = self.backend.InputStream(
                device=self.config.device,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                blocksize=self.config.block_size,
                dtype=self.config.dtype,
                callback=self._audio_callback,
            )
            stream.start()
        except Exception as exc:
            LOGGER.error("Could not start audio capture: %s", exc)
            raise DeviceDisconnectedError(
                f"Could not start input device: {exc}"
            ) from exc

        with self._lock:
            self._stream = stream
            self._running = True

        LOGGER.info(
            "Started mono capture at %d Hz with block size %d",
            self.config.sample_rate,
            self.config.block_size,
        )

    def stop(self, timeout: float = 2.0) -> None:
        """Stop and close the input stream without performing audio work."""
        with self._lock:
            stream = self._stream
            self._stream = None
            self._running = False

        if stream is None:
            return

        try:
            stream.stop()
        except Exception as exc:
            LOGGER.warning("Input stream stop returned an error: %s", exc)
        try:
            stream.close()
        except Exception as exc:
            LOGGER.warning("Input stream close returned an error: %s", exc)

        LOGGER.info("Stopped audio capture")

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def read(self, timeout: float | None = None) -> CapturedBlock:
        """Return the next captured block, blocking up to timeout seconds."""
        try:
            block = self._blocks.get(timeout=timeout)
        except Empty as exc:
            raise TimeoutError("No captured audio block became available") from exc
        return block

    def read_available(self) -> list[CapturedBlock]:
        """Return all currently queued blocks without blocking."""
        blocks: list[CapturedBlock] = []
        while True:
            try:
                blocks.append(self._blocks.get_nowait())
            except Empty:
                break
        return blocks

    def raise_if_failed(self) -> None:
        """Raise callback or PortAudio status errors once observed."""
        error = self.callback_error
        if error is not None:
            raise error

    def _release_buffer(self, buffer: np.ndarray) -> None:
        try:
            self._buffer_pool.put_nowait(buffer)
        except Full:
            pass

    def _release_block(self, block: CapturedBlock) -> None:
        try:
            self._block_pool.put_nowait(block)
        except Full:
            pass

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Lightweight SoundDevice callback; no I/O or expensive processing."""
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
            if "overflow" in lowered:
                self._overflow_count += 1
            if "device" in lowered or "disconnect" in lowered:
                self._callback_error = DeviceDisconnectedError(
                    f"Input device disconnected: {status_text}"
                )

        try:
            buffer = self._buffer_pool.get_nowait()
        except Empty:
            self._callback_error = InputOverflowError(
                "Capture consumer is too slow; preallocated buffers exhausted"
            )
            return
        try:
            block = self._block_pool.get_nowait()
        except Empty:
            self._release_buffer(buffer)
            return

        if frames == 0:
            self._release_buffer(buffer)
            return

        np.copyto(buffer, indata[:frames, 0])
        sequence = self._sequence
        self._sequence += 1

        block.samples = buffer
        block.frames = frames
        block.sample_rate = self.config.sample_rate
        block.sequence = sequence
        block.capture_time = time.monotonic()
        block._owner = self
        try:
            self._blocks.put_nowait(block)
        except Full:
            self._overflow_count += 1
            self._release_buffer(buffer)
            block._owner = None
            self._release_block(block)


def main() -> int:
    """Capture briefly and report block metadata without processing or writing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Capture real-time mono microphone blocks"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-rate", type=int, choices=[44_100, 48_000], default=48_000)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--duration", type=float, default=3.0)
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

    capture = AudioCapture(
        AudioCaptureConfig(
            device=device,
            sample_rate=args.sample_rate,
            block_size=args.block_size,
        )
    )

    deadline = time.monotonic() + args.duration
    block_count = 0
    try:
        capture.start()
        print(
            f"Capturing mono audio at {args.sample_rate} Hz "
            f"in blocks of {args.block_size} frames"
        )
        while time.monotonic() < deadline:
            try:
                block = capture.read(timeout=0.25)
            except TimeoutError:
                capture.raise_if_failed()
                continue
            block_count += 1
            peak = float(np.max(np.abs(block.samples))) if block.frames else 0.0
            print(
                f"\rBlock {block.sequence}: "
                f"frames={block.frames}, peak={peak:.6f}",
                end="",
                flush=True,
            )
            block.release()
    except (DeviceDisconnectedError, InputOverflowError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print()
    finally:
        capture.stop()

    print(f"\nCaptured {block_count} blocks.")
    if capture.overflow_count:
        print(f"Overflow count: {capture.overflow_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
