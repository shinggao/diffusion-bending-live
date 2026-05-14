"""
Audio feature extraction for the live pipeline.

Scalar features (`rms`, `centroid`, `spread`, `skewness`, `kurtosis`, `flux`)
are re-used from `utils.bending` directly — they're already pure functions of
(audio_slice, sr) and live-safe.

The matrix path (EnCodec → (4, 4)) requires substituting for sklearn t-SNE,
which can't `.transform()` new points. Three replacements implemented here:
  - `pca`     IncrementalPCA, fit during warmup, transform per frame
  - `slice`   first 16 EnCodec dims reshaped to (4, 4); zero warmup
  - `opentsne` openTSNE.TSNEEmbedding.transform() (lazy import)

All three return torch tensors of shape (4, 4) ready for `tensor_multiply`
during the post-warmup phase. While warming up, `.step(...)` returns None — the
engine treats None as "skip bending this frame".

See `project_encodec_live_alternatives.md` in the memory store for the
why-not-just-tsne discussion.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


# Default sizing tuned to FPS=20 in the offline pipeline.
_DEFAULT_WARMUP_FRAMES = 60       # 3 seconds at 20 fps
_DEFAULT_REFIT_FRAMES = 30        # additional batches for partial_fit after warmup
_TARGET_DIMS = 16
_MATRIX_SHAPE = (4, 4)


class LiveMatrixFeature:
    """EnCodec encoder + online dim-reduction → (4, 4) tensor per audio slice."""

    def __init__(
        self,
        encodec_model,
        encodec_processor,
        mode: str = "pca",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        warmup_frames: int = _DEFAULT_WARMUP_FRAMES,
        refit_frames: int = _DEFAULT_REFIT_FRAMES,
    ) -> None:
        if mode not in ("pca", "slice", "opentsne"):
            raise ValueError(f"Unknown matrix-mode: {mode!r}")
        self.model = encodec_model
        self.processor = encodec_processor
        self.mode = mode
        self.device = device
        self.dtype = dtype
        self.warmup_frames = int(warmup_frames)
        self.refit_frames = int(refit_frames)

        # Mode-specific state, populated lazily on first use.
        self._buffer: list = []         # warmup buffer of raw EnCodec vectors
        self._batch: list = []          # post-warmup batch for partial_fit
        self._ipca = None               # sklearn IncrementalPCA
        self._opentsne = None           # openTSNE.TSNEEmbedding
        self._frame_idx = 0

        # `slice` mode has no warmup.
        if self.mode == "slice":
            self.warmup_frames = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return f"encodec_{self.mode}"

    @property
    def warmed_up(self) -> bool:
        return self._frame_idx >= self.warmup_frames

    def step(self, audio_slice: np.ndarray, sr: int) -> Optional[torch.Tensor]:
        """Return a (4, 4) torch tensor for this audio slice, or None during warmup."""
        encoding = self._encode_audio(audio_slice, sr)  # (150,) numpy

        if self.mode == "slice":
            self._frame_idx += 1
            return self._reshape_to_matrix(encoding[:_TARGET_DIMS])

        if self.mode == "pca":
            return self._step_pca(encoding)

        if self.mode == "opentsne":
            return self._step_opentsne(encoding)

        # Unreachable: validated in __init__.
        raise RuntimeError(f"Unexpected mode {self.mode}")

    # ------------------------------------------------------------------ #
    # EnCodec encoding (shared across modes)
    # ------------------------------------------------------------------ #

    def _encode_audio(self, audio_slice: np.ndarray, sr: int) -> np.ndarray:
        # EnCodec expects sampling_rate == 48000 (facebook/encodec_48khz) and stereo.
        assert sr == self.processor.sampling_rate, (
            f"Input SR {sr} != EnCodec SR {self.processor.sampling_rate}"
        )
        audio = audio_slice
        if audio.ndim == 1:
            audio = np.stack((audio, audio), axis=0)
        inputs = self.processor(
            raw_audio=audio,
            sampling_rate=self.processor.sampling_rate,
            return_tensors="pt",
        )
        encoder_outputs = self.model.encode(inputs["input_values"], inputs["padding_mask"])
        # Shape (batch, codebook, time) → squeeze and take first codebook timestep
        # to match the offline path in `utils.bending.encodec`.
        matrix = encoder_outputs[0].squeeze()
        vec = matrix[0]  # one 150-D EnCodec vector for this slice
        return vec.detach().cpu().numpy().astype(np.float32, copy=False)

    # ------------------------------------------------------------------ #
    # Mode implementations
    # ------------------------------------------------------------------ #

    def _step_pca(self, encoding: np.ndarray) -> Optional[torch.Tensor]:
        from sklearn.decomposition import IncrementalPCA

        self._frame_idx += 1

        if not self.warmed_up:
            self._buffer.append(encoding)
            if len(self._buffer) == self.warmup_frames:
                # First fit on the warmup buffer.
                self._ipca = IncrementalPCA(n_components=_TARGET_DIMS)
                self._ipca.partial_fit(np.stack(self._buffer, axis=0))
                self._buffer.clear()
            return None

        # Post-warmup: transform this point, then accumulate for periodic refit.
        assert self._ipca is not None
        reduced = self._ipca.transform(encoding.reshape(1, -1))[0]
        self._batch.append(encoding)
        if len(self._batch) >= self.refit_frames:
            self._ipca.partial_fit(np.stack(self._batch, axis=0))
            self._batch.clear()
        return self._reshape_to_matrix(reduced)

    def _step_opentsne(self, encoding: np.ndarray) -> Optional[torch.Tensor]:
        # Lazy import: only pay the dependency cost if this mode is selected.
        from openTSNE import TSNE  # type: ignore

        self._frame_idx += 1

        if not self.warmed_up:
            self._buffer.append(encoding)
            if len(self._buffer) == self.warmup_frames:
                tsne = TSNE(
                    n_components=_TARGET_DIMS,
                    perplexity=min(15, max(2, self.warmup_frames // 4)),
                    n_jobs=-1,
                )
                # `.fit` returns a TSNEEmbedding which exposes `.transform(...)`
                # for out-of-sample projection.
                self._opentsne = tsne.fit(np.stack(self._buffer, axis=0))
                self._buffer.clear()
            return None

        assert self._opentsne is not None
        reduced = self._opentsne.transform(encoding.reshape(1, -1))[0]
        return self._reshape_to_matrix(np.asarray(reduced, dtype=np.float32))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _reshape_to_matrix(self, vec: np.ndarray) -> torch.Tensor:
        arr = np.asarray(vec, dtype=np.float32).reshape(_MATRIX_SHAPE)
        return torch.from_numpy(arr).to(self.device, self.dtype)
