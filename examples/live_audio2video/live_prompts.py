"""
Prompt timeline encoded once at startup; wall-clock lerp at render time.

Same `MM:SS : prompt text` file format the offline `audio2video.py` consumes
(see `audio2video.py:184-189`). The schedule is fixed when this class is built;
no UI for editing prompts mid-run in v1.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch


def parse_prompt_file(prompt_file: Path) -> Tuple[List[str], List[float]]:
    """Parse `MM:SS : prompt` lines into prompts[] + seconds[].

    Lines that are blank are skipped. The split allows colons inside the prompt
    body because we only split on the first two colons (MM, SS, rest).
    """
    prompt_file = Path(prompt_file)
    with prompt_file.open("r", encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    prompts: List[str] = []
    seconds: List[float] = []
    for ln in lines:
        parts = ln.split(":", 2)
        if len(parts) < 3:
            raise ValueError(
                f"Bad prompt line in {prompt_file.name}: {ln!r}. "
                "Expected `MM:SS : prompt text`."
            )
        mm, ss, body = parts
        seconds.append(float(mm) * 60.0 + float(ss))
        prompts.append(body.strip())
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")
    return prompts, seconds


def encode_prompt(wrapper, prompt: str, negative_prompt: str, do_cfg: bool) -> torch.Tensor:
    """Encode a single prompt → CLIP embedding via the StreamDiffusion pipeline.

    Mirrors the helper inlined in `audio2video.py:122-131` so the offline and
    live paths use the same encoding contract.
    """
    encoder_output = wrapper.stream.pipe.encode_prompt(
        prompt=prompt,
        device=wrapper.stream.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
    )
    prompt_embeds = encoder_output[0].repeat(wrapper.stream.batch_size, 1, 1)
    return prompt_embeds


class PromptSchedule:
    """Fixed prompt timeline + wall-clock-driven embedding lerp."""

    def __init__(
        self,
        prompt_file: Path,
        wrapper,
        negative_prompt: str = "",
        guidance_scale: float = 1.2,
    ) -> None:
        self.prompts, self.seconds = parse_prompt_file(prompt_file)
        do_cfg = guidance_scale > 1.0
        self.embeddings: List[torch.Tensor] = [
            encode_prompt(wrapper, p, negative_prompt, do_cfg) for p in self.prompts
        ]
        # Track which segment the last lerp landed in, useful for HUD reporting.
        self._last_segment: int = 0
        self._last_alpha: float = 0.0

    # ------------------------------------------------------------------ #
    # Lerp API
    # ------------------------------------------------------------------ #

    def embedding_at(self, t_seconds: float) -> torch.Tensor:
        """Return the CLIP embedding for wall-clock time `t_seconds`."""
        if len(self.embeddings) == 1:
            self._last_segment = 0
            self._last_alpha = 0.0
            return self.embeddings[0]

        # Before the first timestamp → first embedding.
        if t_seconds <= self.seconds[0]:
            self._last_segment = 0
            self._last_alpha = 0.0
            return self.embeddings[0]

        # After the last timestamp → last embedding (clamp).
        if t_seconds >= self.seconds[-1]:
            self._last_segment = len(self.embeddings) - 1
            self._last_alpha = 1.0
            return self.embeddings[-1]

        # Find bracketing segment.
        i = 0
        for k in range(len(self.seconds) - 1):
            if self.seconds[k] <= t_seconds < self.seconds[k + 1]:
                i = k
                break

        start_t, end_t = self.seconds[i], self.seconds[i + 1]
        span = end_t - start_t
        alpha = 0.0 if span <= 0 else (t_seconds - start_t) / span
        a = self.embeddings[i]
        b = self.embeddings[i + 1]
        self._last_segment = i
        self._last_alpha = float(alpha)
        return (1.0 - alpha) * a + alpha * b

    # ------------------------------------------------------------------ #
    # HUD introspection
    # ------------------------------------------------------------------ #

    def hud_state(self) -> Tuple[int, float, str]:
        """Return (segment_index, lerp_alpha, current_prompt_text) for the HUD overlay."""
        # The "current" prompt is whichever side has more weight, for display.
        idx = self._last_segment
        if self._last_alpha >= 0.5 and idx + 1 < len(self.prompts):
            idx = idx + 1
        return self._last_segment, self._last_alpha, self.prompts[idx]
