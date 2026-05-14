"""
Microphone capture for the live audio2video pipeline.

Wraps `sounddevice.InputStream` with:
  - a thread-safe numpy ring buffer the render loop can sample on demand
  - an optional simultaneous wav recorder for the post-run ffmpeg mux

Sample rate is fixed at 48000 to match the offline pipeline's SAMPLING_RATE.
Mono only; stereo devices get downmixed by averaging channels in the callback.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover - import-time deferred error
    sd = None
    _SD_IMPORT_ERROR = exc
else:
    _SD_IMPORT_ERROR = None

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover
    sf = None
    _SF_IMPORT_ERROR = exc
else:
    _SF_IMPORT_ERROR = None


def list_devices() -> str:
    """Return sounddevice's device table as a string (for the --list-devices flag)."""
    if sd is None:
        raise RuntimeError(
            f"sounddevice not installed ({_SD_IMPORT_ERROR}); run `pip install sounddevice`"
        )
    return str(sd.query_devices())


class MicCapture:
    """Non-blocking mic ring buffer with optional simultaneous wav recording.

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz. Locked to 48000 in the live pipeline.
    buffer_seconds:
        Length of the in-memory ring buffer. 2.0 s is plenty: the feature path
        only needs ~50 ms (1/FPS at 20 fps) and the UI taps ~250 ms.
    device:
        Pass-through to `sounddevice.InputStream(device=...)`. None = system default.
        Accepts int index or partial-string device name.
    blocksize:
        Frames per callback. 0 lets sounddevice pick — usually 256 or 512 on macOS.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        buffer_seconds: float = 2.0,
        device: Optional[object] = None,
        blocksize: int = 0,
    ) -> None:
        if sd is None:
            raise RuntimeError(
                f"sounddevice not installed ({_SD_IMPORT_ERROR}); run `pip install sounddevice`"
            )

        self.sample_rate = int(sample_rate)
        self.device = device
        self.blocksize = int(blocksize)

        self._buffer = np.zeros(int(buffer_seconds * self.sample_rate), dtype=np.float32)
        self._write_idx = 0  # next write position; wraps via modulo
        self._lock = threading.Lock()

        self._stream: Optional["sd.InputStream"] = None
        self._wav: Optional["sf.SoundFile"] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self, record_path: Optional[Path] = None) -> None:
        """Open the input stream (and optionally a wav recorder)."""
        if self._stream is not None:
            raise RuntimeError("MicCapture already started")

        if record_path is not None:
            if sf is None:
                raise RuntimeError(
                    f"soundfile not installed ({_SF_IMPORT_ERROR}); run `pip install soundfile`"
                )
            record_path = Path(record_path)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            self._wav = sf.SoundFile(
                str(record_path),
                mode="w",
                samplerate=self.sample_rate,
                channels=1,
                subtype="PCM_16",
            )

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,  # request mono; if device only supports stereo, we'd switch to 2 and average
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Close the stream and flush the wav file."""
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None
        if self._wav is not None:
            try:
                self._wav.flush()
            finally:
                self._wav.close()
                self._wav = None

    # ------------------------------------------------------------------ #
    # sounddevice callback
    # ------------------------------------------------------------------ #

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        # `indata` is float32, shape (frames, channels). Downmix to mono if stereo.
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata.mean(axis=1).astype(np.float32, copy=False)
        else:
            mono = indata[:, 0] if indata.ndim == 2 else indata
            mono = mono.astype(np.float32, copy=False)

        # Append to the on-disk wav (if recording).
        if self._wav is not None:
            self._wav.write(mono)

        # Append to the ring buffer.
        n = mono.shape[0]
        buf = self._buffer
        with self._lock:
            end = self._write_idx + n
            if end <= buf.size:
                buf[self._write_idx:end] = mono
            else:
                first = buf.size - self._write_idx
                buf[self._write_idx:] = mono[:first]
                buf[: n - first] = mono[first:]
            self._write_idx = end % buf.size

    # ------------------------------------------------------------------ #
    # Read API
    # ------------------------------------------------------------------ #

    def get_latest(self, n_samples: int) -> np.ndarray:
        """Return the most recent `n_samples` samples (chronological order).

        If `n_samples` exceeds the buffer length, returns the whole buffer.
        If the buffer hasn't filled yet, the returned array is still
        `min(n_samples, buffer.size)` long — the unfilled region reads as zeros,
        which is fine: the early frames just have a quiet pre-roll.
        """
        n = min(int(n_samples), self._buffer.size)
        with self._lock:
            end = self._write_idx
            start = end - n
            if start >= 0:
                out = self._buffer[start:end].copy()
            else:
                # Wrap: piece from tail of buffer + piece from head.
                out = np.concatenate(
                    (self._buffer[start:], self._buffer[:end]),
                    dtype=np.float32,
                )
        return out

    def get_waveform(self, n_samples: int) -> np.ndarray:
        """Alias kept separate so the UI tap is conceptually distinct from feature tap."""
        return self.get_latest(n_samples)
