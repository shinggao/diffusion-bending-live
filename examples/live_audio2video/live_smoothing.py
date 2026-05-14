"""
Causal/streaming replacement for `utils.bending.smooth(...)`.

The offline `smooth(kernel, alpha, threshold)` chains:
  median filter (scipy.signal.medfilt, centered)
  → envelope follower
  → noise gate

The median filter is acausal (it needs the full series) so we replace it with
a one-sided sliding-window median. Envelope follower is already causal — its
formula in `bending.py:81-92` only references prev → we just apply it per-sample
and keep `prev` as instance state. Noise gate is stateless.

The one-sided median introduces a lag of roughly kernel_size/2 frames vs. the
offline centered median. Default kernel here is 21 instead of the offline 41
to compensate; callers can override.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np


class StreamingSmoother:
    """Per-sample causal smoother. Call `.step(value) -> smoothed`."""

    def __init__(
        self,
        kernel_size: int = 21,
        envelope_alpha: float = 0.7,
        noise_threshold: float = 0.0,
    ) -> None:
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if not (0.0 <= envelope_alpha <= 1.0):
            raise ValueError("envelope_alpha must be in [0, 1]")
        self.kernel_size = int(kernel_size)
        self.envelope_alpha = float(envelope_alpha)
        self.noise_threshold = float(noise_threshold)

        self._window: Deque[float] = deque(maxlen=self.kernel_size)
        self._env_prev: float = 0.0
        self._env_initialised: bool = False

    @property
    def name(self) -> str:
        """Filename-style descriptor for the parameter set (mirrors the offline tag)."""
        return (
            f"smedian{self.kernel_size}"
            f"envelope{self.envelope_alpha}"
            f"gate{self.noise_threshold}"
        )

    def step(self, value: float) -> float:
        # Stage 1: one-sided sliding median.
        self._window.append(float(value))
        med = float(np.median(self._window))

        # Stage 2: envelope follower (causal). Same formula as bending.py:81-92.
        if not self._env_initialised:
            self._env_prev = med
            self._env_initialised = True
            env = med
        else:
            prev = self._env_prev
            if abs(med) > abs(prev):
                env = med
            else:
                a = self.envelope_alpha
                env = (1.0 - a) * med + a * prev
            self._env_prev = env

        # Stage 3: noise gate (stateless threshold).
        gated = 0.0 if abs(env) < self.noise_threshold else env
        return float(gated)

    def reset(self) -> None:
        """Clear smoother state. Useful for unit-testing or restarting a session."""
        self._window.clear()
        self._env_prev = 0.0
        self._env_initialised = False


def smooth_series(
    values, kernel_size: int = 21, envelope_alpha: float = 0.7, noise_threshold: float = 0.0
) -> Tuple[np.ndarray, str]:
    """Convenience: apply `StreamingSmoother` to an entire 1-D series at once.

    Mirrors the offline `bending.smooth(...)` return shape `(smoothed, name)` so
    existing test scripts that compare offline vs. streaming smoothing can swap
    one for the other.
    """
    s = StreamingSmoother(kernel_size, envelope_alpha, noise_threshold)
    out = np.fromiter((s.step(v) for v in values), dtype=np.float32, count=len(values))
    return out, s.name
