"""
cv2 preview window + per-run frame recorder for the live pipeline.

LivePreview composites the latest diffusion frame, a scrolling mic waveform,
and a small HUD into a single cv2 window. The render loop calls
`update(frame_pil, waveform_np, hud_lines)` per frame and checks `should_quit()`
afterwards.

FrameRecorder writes PNGs to a per-run staging directory under
`live_video_outputs/.staging/{timestamp}/` and on `finalize(audio_wav_path)`
shells out to ffmpeg to mux the PNGs + mic.wav into
`live_video_outputs/{filename}.mp4`. Filename pattern mirrors the offline
`audio2video.py:360` so live and offline outputs sort intelligibly together.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image


_WINDOW_TITLE = "diffusion-bending-live"
_WAVEFORM_HEIGHT = 120
_WAVEFORM_BG = (16, 16, 16)
_WAVEFORM_LINE = (0, 220, 120)
_HUD_BG_ALPHA = 0.45
_HUD_FG = (240, 240, 240)


# ---------------------------------------------------------------------- #
# LivePreview
# ---------------------------------------------------------------------- #


@dataclass
class HudState:
    """Snapshot of HUD-relevant values for one frame."""

    frame_idx: int = 0
    measured_fps: float = 0.0
    raw_feature: float = 0.0
    smoothed_feature: float = 0.0
    prompt_segment: int = 0
    prompt_alpha: float = 0.0
    prompt_text: str = ""
    warming_up: bool = False
    extra: List[str] = field(default_factory=list)

    def to_lines(self) -> List[str]:
        lines = [
            f"frame {self.frame_idx:5d}   fps {self.measured_fps:5.2f}",
            f"feature raw={self.raw_feature:+.4f}  smoothed={self.smoothed_feature:+.4f}",
            f"prompt #{self.prompt_segment}  a={self.prompt_alpha:.2f}  '{self.prompt_text[:48]}'",
        ]
        if self.warming_up:
            lines.append("[warming up — bending suppressed]")
        lines.extend(self.extra)
        return lines


class LivePreview:
    """Single cv2 window showing (frame | waveform) with an HUD overlay."""

    def __init__(self, window_title: str = _WINDOW_TITLE) -> None:
        self.window_title = window_title
        self._last_key: int = -1
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)

    # ------------------------------------------------------------------ #
    # Per-frame update
    # ------------------------------------------------------------------ #

    def update(
        self,
        frame_pil: Image.Image,
        waveform: np.ndarray,
        hud: Optional[HudState] = None,
    ) -> None:
        frame_bgr = _pil_to_bgr(frame_pil)
        wave_canvas = _render_waveform(waveform, width=frame_bgr.shape[1])
        composite = np.vstack((frame_bgr, wave_canvas))
        if hud is not None:
            _draw_hud(composite, hud.to_lines())
        cv2.imshow(self.window_title, composite)
        self._last_key = cv2.waitKey(1) & 0xFF

    # ------------------------------------------------------------------ #
    # Quit detection
    # ------------------------------------------------------------------ #

    def should_quit(self) -> bool:
        # `q` or Escape pressed.
        if self._last_key in (ord("q"), 27):
            return True
        # User closed the window via the OS chrome.
        try:
            visible = cv2.getWindowProperty(self.window_title, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return True
        if visible < 1:
            return True
        return False

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window_title)
        except cv2.error:
            pass
        # destroyAllWindows for good measure — some OSs need it to actually release.
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    return arr[:, :, ::-1].copy()  # RGB → BGR


def _render_waveform(samples: np.ndarray, width: int) -> np.ndarray:
    canvas = np.full((_WAVEFORM_HEIGHT, width, 3), _WAVEFORM_BG, dtype=np.uint8)
    if samples.size < 2:
        return canvas
    # Downsample to one point per pixel column (max-abs in each bin) so loud
    # passages look like a proper waveform envelope rather than a tangle.
    s = samples.astype(np.float32, copy=False)
    bins = np.linspace(0, s.size, num=width + 1, dtype=np.int64)
    points = np.empty((width, 2), dtype=np.int32)
    mid = _WAVEFORM_HEIGHT // 2
    peak = float(np.max(np.abs(s))) if s.size else 0.0
    scale = (mid - 4) / peak if peak > 1e-6 else 0.0
    for x in range(width):
        lo, hi = bins[x], bins[x + 1]
        if hi <= lo:
            y = mid
        else:
            chunk = s[lo:hi]
            y = mid - int(chunk[np.argmax(np.abs(chunk))] * scale)
            y = max(2, min(_WAVEFORM_HEIGHT - 2, y))
        points[x, 0] = x
        points[x, 1] = y
    cv2.polylines(canvas, [points], isClosed=False, color=_WAVEFORM_LINE, thickness=1)
    return canvas


def _draw_hud(composite: np.ndarray, lines: Sequence[str]) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    pad = 6
    line_h = 16
    text_w = 0
    for ln in lines:
        (tw, _th), _ = cv2.getTextSize(ln, font, scale, thickness)
        text_w = max(text_w, tw)
    box_w = text_w + 2 * pad
    box_h = line_h * len(lines) + 2 * pad
    box_h = min(box_h, composite.shape[0])
    box_w = min(box_w, composite.shape[1])

    # Semi-transparent dark background.
    overlay = composite.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, _HUD_BG_ALPHA, composite, 1.0 - _HUD_BG_ALPHA, 0, composite)

    y = pad + 12
    for ln in lines:
        cv2.putText(composite, ln, (pad, y), font, scale, _HUD_FG, thickness, cv2.LINE_AA)
        y += line_h


# ---------------------------------------------------------------------- #
# FrameRecorder
# ---------------------------------------------------------------------- #


class FrameRecorder:
    """Writes per-frame PNGs to a staging dir, then ffmpeg-muxes on `finalize`."""

    def __init__(
        self,
        output_root: Path,
        fps: int = 20,
        width: int = 512,
        height: int = 512,
        run_tag: Optional[str] = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.run_tag = run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.staging_dir: Optional[Path] = None
        self._frame_count = 0

    @property
    def started(self) -> bool:
        return self.staging_dir is not None

    def start(self) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.staging_dir = self.output_root / ".staging" / self.run_tag
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        return self.staging_dir

    def write(self, frame_pil: Image.Image, frame_idx: int) -> None:
        if self.staging_dir is None:
            raise RuntimeError("FrameRecorder.start() must be called before write()")
        path = self.staging_dir / f"{frame_idx:05d}.png"
        frame_pil.save(path)
        self._frame_count = max(self._frame_count, frame_idx + 1)

    def finalize(
        self,
        audio_wav_path: Optional[Path],
        filename_parts: Sequence[str],
        keep_staging: bool = False,
    ) -> Optional[Path]:
        """Mux PNG sequence (+ optional wav) into mp4. Returns the output path or None."""
        if self.staging_dir is None or self._frame_count == 0:
            return None

        # Build the output filename, parallel to the offline pattern at
        # `audio2video.py:360`. Caller controls the descriptive parts.
        dim_string = f"_{self.height}x{self.width}" if (self.width, self.height) != (512, 512) else ""
        base = "_".join(["live", self.run_tag, *filename_parts]) + f"_{self.fps}fps{dim_string}"
        out_path = self.output_root / f"{base}.mp4"
        counter = 1
        while out_path.exists():
            out_path = self.output_root / f"{base}_{counter}.mp4"
            counter += 1

        ffmpeg_bin = os.environ.get("JAES_FFMPEG", "ffmpeg")
        if Path(ffmpeg_bin).exists():
            ffmpeg_exe = str(Path(ffmpeg_bin))
        else:
            ffmpeg_exe = shutil.which(ffmpeg_bin)
        if ffmpeg_exe is None:
            raise FileNotFoundError(
                "ffmpeg executable not found. Install ffmpeg or set JAES_FFMPEG to the full path."
            )

        cmd: List[str] = [
            ffmpeg_exe,
            "-y",
            "-framerate", str(self.fps),
            "-i", str(self.staging_dir / "%05d.png"),
        ]
        has_audio = audio_wav_path is not None and Path(audio_wav_path).exists()
        if has_audio:
            cmd += ["-i", str(audio_wav_path)]
        cmd += ["-vcodec", "libx264", "-pix_fmt", "yuv420p"]
        if has_audio:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += [str(out_path)]

        subprocess.run(cmd, check=True)

        if not keep_staging:
            shutil.rmtree(self.staging_dir, ignore_errors=True)
            staging_root = self.output_root / ".staging"
            if staging_root.exists() and not any(staging_root.iterdir()):
                staging_root.rmdir()

        return out_path
