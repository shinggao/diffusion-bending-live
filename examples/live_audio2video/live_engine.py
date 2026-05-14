"""
Persistent StreamDiffusion engine + per-frame hot path for the live pipeline.

Differences from the offline `audio2video.generate_video`:
  - Wrapper built once at startup; prompt embeddings encoded once via
    `PromptSchedule`. Per-frame `prepare()` is cheap because we pass
    `prompt_encoding=...` (CLIP encoder is bypassed in `pipeline.py:180`).
  - Noise walk advances by a fixed phase step per frame instead of being
    pre-computed across `num_frames` samples — the run length is unbounded.
  - Scalar features are rescaled online using a rolling-percentile window
    instead of the offline global min/max.
  - Matrix features (EnCodec) go through `LiveMatrixFeature` which returns
    None during its warmup; the engine skips bending for those frames.
"""

from __future__ import annotations

import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Optional

import numpy as np
import torch
from PIL import Image

# sys.path bootstrap so `utils.*` and `examples.audio2video.audio2video` resolve
# when this file is run/imported directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils.bending as util  # noqa: E402
from utils.wrapper import StreamDiffusionWrapper  # noqa: E402

# Reuse the offline operator→range dict so we don't duplicate that table.
from examples.audio2video.audio2video import bending_functions_range  # noqa: E402

from examples.live_audio2video.live_features import LiveMatrixFeature  # noqa: E402
from examples.live_audio2video.live_output import HudState  # noqa: E402
from examples.live_audio2video.live_prompts import PromptSchedule  # noqa: E402
from examples.live_audio2video.live_smoothing import StreamingSmoother  # noqa: E402


# Sentinel layer value the offline batch.py uses to disable the bending hook
# while keeping the rest of the pipeline identical (see `project_design_choices`).
LAYER_DISABLED = 420


class LiveEngine:
    """Owns the StreamDiffusionWrapper and renders one frame at a time.

    Parameters
    ----------
    prompt_file:
        Path to the timestamped `MM:SS : prompt` file (same format as offline).
    bend_function:
        Factory in the `utils.bending` style: takes one parameter, returns a
        latent-tensor → latent-tensor closure.
    feature_fn:
        Either a scalar feature `(audio, sr) -> float` from `utils.bending`
        (rms / centroid / spread / skewness / kurtosis / flux) OR a
        `LiveMatrixFeature` instance for the matrix branch.
    smoother:
        StreamingSmoother for the scalar branch. Ignored when feature_fn is a
        LiveMatrixFeature (matrix smoothing left to a future iteration; see
        the comment near `_step_matrix`).
    layer:
        Bending layer index. Use `LAYER_DISABLED` to keep bending machinery off
        for an A/B baseline. Must be in [0, len(t_index_list)) OR ==LAYER_DISABLED.
    """

    SAMPLING_RATE = 48000
    FPS = 20
    GUIDANCE_SCALE = 1.2
    NEGATIVE_PROMPT_DEFAULT = "low quality, text, words, letters"

    def __init__(
        self,
        prompt_file: Path,
        bend_function: Callable,
        feature_fn,
        smoother: Optional[StreamingSmoother],
        layer: int,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
        t_index_list: Optional[list] = None,
        noise_period_frames: int = 200,
        negative_prompt: Optional[str] = None,
        feature_history_seconds: float = 4.0,
        feature_warmup_frames: int = 40,
        rescale_percentile: float = 5.0,
    ) -> None:
        self.bend_function = bend_function
        self.feature_fn = feature_fn
        self.smoother = smoother
        self.layer = int(layer)
        self.seed = int(seed)
        self.width = int(width)
        self.height = int(height)
        self.negative_prompt = (
            negative_prompt if negative_prompt is not None else self.NEGATIVE_PROMPT_DEFAULT
        )

        # Pre-set seeds. The wrapper sets its own torch seed via __init__, but
        # we still want random/np/torch seeded for the noise walk basis.
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # ---------- Build the persistent wrapper ----------
        if t_index_list is None:
            t_index_list = [22, 32, 45]  # shorter than the offline [0,16,32,45] for live
        self.t_index_list = list(t_index_list)
        if not (0 <= self.layer < len(self.t_index_list) or self.layer == LAYER_DISABLED):
            raise ValueError(
                f"--layer must be 0..{len(self.t_index_list) - 1} (or {LAYER_DISABLED} to disable). "
                f"Got {self.layer}."
            )

        self.wrapper = StreamDiffusionWrapper(
            model_id_or_path="CompVis/stable-diffusion-v1-4",
            lora_dict=None,
            t_index_list=self.t_index_list,
            frame_buffer_size=1,
            width=self.width,
            height=self.height,
            warmup=10,
            acceleration="xformers",
            mode="txt2img",
            use_denoising_batch=False,
            cfg_type="none",
            seed=self.seed,
            bending_fn=None,  # set per-frame
        )

        # ---------- Prompt timeline ----------
        self.prompts = PromptSchedule(
            prompt_file=prompt_file,
            wrapper=self.wrapper,
            negative_prompt=self.negative_prompt,
            guidance_scale=self.GUIDANCE_SCALE,
        )

        # ---------- Noise walk basis (A, B) ----------
        latent_h = self.wrapper.stream.latent_height
        latent_w = self.wrapper.stream.latent_width
        noise_shape = (1, 4, latent_h, latent_w)
        self._A = torch.normal(mean=0.0, std=1.0, size=noise_shape, dtype=torch.float64)
        self._B = torch.normal(mean=0.0, std=1.0, size=noise_shape, dtype=torch.float64)
        self.phase = 0.0
        self.phase_step = (2.0 * math.pi) / max(1, int(noise_period_frames))

        # ---------- Scalar-branch rescale state ----------
        history_size = int(feature_history_seconds * self.FPS)
        self._feature_history: Deque[float] = deque(maxlen=max(1, history_size))
        self.feature_warmup_frames = int(feature_warmup_frames)
        self.rescale_percentile = float(rescale_percentile)
        self._matrix_mode = isinstance(self.feature_fn, LiveMatrixFeature)
        if not self._matrix_mode:
            rng = bending_functions_range.get(self.bend_function, ())
            self._bend_min, self._bend_max = (rng if rng else (0.0, 1.0))
            self._bend_midpoint = 0.5 * (self._bend_min + self._bend_max)
        else:
            self._bend_min = self._bend_max = self._bend_midpoint = 0.0

        # ---------- HUD scratch ----------
        self._frame_idx = 0
        self._last_raw: float = 0.0
        self._last_smoothed: float = 0.0
        self._last_warmup_flag: bool = True
        self._t_last_frame: Optional[float] = None
        self._fps_ema: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate_frame(self, audio_slice: np.ndarray, t_seconds: float) -> Image.Image:
        """Render one frame using the current audio slice and wall-clock time."""
        if self._matrix_mode:
            bend = self._step_matrix(audio_slice)
        else:
            bend = self._step_scalar(audio_slice)

        # Advance the noise walk regardless of bending.
        noise = self._A * math.sin(self.phase) + self._B * math.cos(self.phase)
        self.phase += self.phase_step

        # Cached embedding for this wall-clock; CLIP encoder is bypassed.
        embeds = self.prompts.embedding_at(t_seconds)

        self.wrapper.prepare(
            prompt_encoding=embeds,
            num_inference_steps=50,
            guidance_scale=self.GUIDANCE_SCALE,
            bending_fn=bend,
            bending_layer=self.layer,
            input_noise=noise,
        )
        out = self.wrapper()

        # The wrapper's txt2img returns a PIL.Image (single frame_buffer_size=1).
        if isinstance(out, list):
            out = out[0]
        self._tick_fps()
        self._frame_idx += 1
        return out

    def last_hud(self, prompts_hud_text: Optional[str] = None) -> HudState:
        seg, alpha, text = self.prompts.hud_state()
        return HudState(
            frame_idx=self._frame_idx,
            measured_fps=self._fps_ema,
            raw_feature=self._last_raw,
            smoothed_feature=self._last_smoothed,
            prompt_segment=seg,
            prompt_alpha=alpha,
            prompt_text=prompts_hud_text or text,
            warming_up=self._last_warmup_flag,
        )

    # ------------------------------------------------------------------ #
    # Per-frame branches
    # ------------------------------------------------------------------ #

    def _step_scalar(self, audio_slice: np.ndarray):
        raw = float(self.feature_fn(audio_slice, self.SAMPLING_RATE))
        smoothed = self.smoother.step(raw) if self.smoother is not None else raw

        self._feature_history.append(raw)
        self._last_raw = raw
        self._last_smoothed = float(smoothed)

        warming = self._frame_idx < self.feature_warmup_frames
        self._last_warmup_flag = warming

        if self.layer == LAYER_DISABLED:
            return None

        if warming:
            # Stable midpoint while min/max accumulates: avoids huge swings on frame 0.
            return self.bend_function(self._bend_midpoint) if self.bend_function else None

        # Rolling-percentile rescale (less noisy than true min/max).
        buf = np.fromiter(self._feature_history, dtype=np.float32)
        lo = float(np.percentile(buf, self.rescale_percentile))
        hi = float(np.percentile(buf, 100.0 - self.rescale_percentile))
        if hi - lo < 1e-9:
            scaled = self._bend_midpoint
        else:
            scaled = util.scale_range(smoothed, lo, hi, self._bend_min, self._bend_max)
            scaled = float(np.clip(scaled, self._bend_min, self._bend_max))
        return self.bend_function(scaled) if self.bend_function is not None else None

    def _step_matrix(self, audio_slice: np.ndarray):
        # Note: matrix-branch smoothing is not applied in v1. The offline path
        # smooths each (i, j) cell of the t-SNE-reduced 4×4 independently; doing
        # the same online would mean keeping 16 StreamingSmoother instances and
        # re-blending after partial_fit. Left as a follow-up — current run will
        # still look reasonable because the EnCodec encoder + PCA already act
        # as a low-pass over the input.
        mat = self.feature_fn.step(audio_slice, self.SAMPLING_RATE)
        self._last_warmup_flag = mat is None
        if mat is None:
            self._last_raw = 0.0
            self._last_smoothed = 0.0
            return None

        # Cast to wrapper device/dtype so tensor_multiply doesn't bounce CPU↔GPU.
        mat = mat.to(self.wrapper.stream.device, self.wrapper.stream.dtype)
        self._last_raw = float(mat.abs().mean().item())
        self._last_smoothed = self._last_raw

        if self.layer == LAYER_DISABLED or self.bend_function is None:
            return None
        return self.bend_function(mat)

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #

    def _tick_fps(self) -> None:
        now = time.monotonic()
        if self._t_last_frame is not None:
            dt = now - self._t_last_frame
            if dt > 1e-6:
                inst = 1.0 / dt
                alpha = 0.1
                self._fps_ema = (
                    inst if self._fps_ema == 0.0 else (alpha * inst + (1 - alpha) * self._fps_ema)
                )
        self._t_last_frame = now
