"""
Entry point for the live audio2video pipeline.

Run:
    python examples/live_audio2video/live_audio2video.py \
        --prompt-file inputs/test.txt \
        --bend multiply --feature rms --layer 1 --seed 42

Press `q` or Esc in the preview window to stop. On exit, the rendered frames
+ captured mic audio get muxed into `live_video_outputs/live_*.mp4` (unless
`--no-save` was set).

See `examples/live_audio2video/README` (or the project README's Live Mode
section) for the full argument reference. Use `--list-devices` to inspect
sounddevice's view of the system's audio inputs without loading the model.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure local imports (e.g. `utils`, `examples.audio2video`) resolve when
# this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # match audio2video.py's matplotlib workaround

import utils.bending as util  # noqa: E402
from examples.audio2video.audio2video import bending_functions_range  # noqa: E402

from examples.live_audio2video.live_audio import MicCapture, list_devices  # noqa: E402
from examples.live_audio2video.live_engine import LAYER_DISABLED, LiveEngine  # noqa: E402
from examples.live_audio2video.live_features import LiveMatrixFeature  # noqa: E402
from examples.live_audio2video.live_output import FrameRecorder, LivePreview  # noqa: E402
from examples.live_audio2video.live_smoothing import StreamingSmoother  # noqa: E402


# ---------------------------------------------------------------------- #
# Argument resolution
# ---------------------------------------------------------------------- #


_SCALAR_FEATURES = {"rms", "centroid", "spread", "skewness", "kurtosis", "flux"}


def _resolve_bend(name: str):
    """Look up a bending operator by name from `utils.bending` and whitelist it."""
    if not hasattr(util, name):
        raise SystemExit(f"--bend {name!r}: no such function in utils.bending")
    fn = getattr(util, name)
    if fn not in bending_functions_range:
        raise SystemExit(
            f"--bend {name!r} is not registered in bending_functions_range. "
            f"Allowed: {sorted(f.__name__ for f in bending_functions_range)}"
        )
    return fn


def _resolve_feature(name: str, args) -> object:
    """Resolve --feature to either a scalar callable or a LiveMatrixFeature instance."""
    if name in _SCALAR_FEATURES:
        return getattr(util, name)

    if name == "encodec":
        try:
            from transformers import EncodecModel, AutoProcessor  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                f"--feature encodec requires transformers>=4.31.0 ({exc})"
            ) from exc
        from transformers import EncodecModel, AutoProcessor  # type: ignore  # repeated import for clarity
        model = EncodecModel.from_pretrained("facebook/encodec_48khz")
        processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")
        return LiveMatrixFeature(
            encodec_model=model,
            encodec_processor=processor,
            mode=args.matrix_mode,
            warmup_frames=args.matrix_warmup_frames,
        )

    raise SystemExit(
        f"--feature {name!r}: unknown. Allowed: {sorted(_SCALAR_FEATURES)} or 'encodec'."
    )


def _parse_audio_device(value):
    if value is None:
        return None
    # Allow either int index or partial-name string.
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


# ---------------------------------------------------------------------- #
# Main loop
# ---------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live audio→video diffusion-bending pipeline."
    )
    parser.add_argument("--prompt-file", type=Path, help="MM:SS : prompt text timeline file.")
    parser.add_argument("--bend", default="multiply", help="Bending operator name (default: multiply).")
    parser.add_argument(
        "--feature",
        default="rms",
        help="Audio feature: rms | centroid | spread | skewness | kurtosis | flux | encodec",
    )
    parser.add_argument(
        "--matrix-mode",
        default="pca",
        choices=("pca", "slice", "opentsne"),
        help="Online dim-reduction for --feature encodec (default: pca).",
    )
    parser.add_argument(
        "--matrix-warmup-frames",
        type=int,
        default=60,
        help="Warmup frames for the matrix-feature path (default: 60).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=1,
        help=f"Bending layer index. Use {LAYER_DISABLED} to disable bending (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--smooth-kernel", type=int, default=21)
    parser.add_argument("--smooth-alpha", type=float, default=0.7)
    parser.add_argument("--smooth-gate", type=float, default=0.0)
    parser.add_argument(
        "--audio-device",
        default=None,
        help="sounddevice device int index or partial-name string (default: system default).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print sounddevice device table and exit (no model load).",
    )
    parser.add_argument(
        "--noise-period-frames",
        type=int,
        default=200,
        help="Frames for one full sin/cos noise walk cycle (default: 200 = 10s @ 20fps).",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt. Default: 'low quality, text, words, letters'.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Preview only; skip wav/png/ffmpeg pipeline.",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(list_devices())
        return 0

    if args.prompt_file is None:
        parser.error("--prompt-file is required (unless --list-devices).")
    if not args.prompt_file.exists():
        parser.error(f"--prompt-file not found: {args.prompt_file}")

    # Resolve bend operator + audio feature into actual callables/objects.
    bend_function = _resolve_bend(args.bend)
    feature_fn = _resolve_feature(args.feature, args)

    # Scalar smoother — only used in the scalar branch (engine ignores it for matrix).
    smoother = StreamingSmoother(
        kernel_size=args.smooth_kernel,
        envelope_alpha=args.smooth_alpha,
        noise_threshold=args.smooth_gate,
    )

    print(">>> Loading StreamDiffusion + encoding prompt timeline (this takes a moment)...")
    engine = LiveEngine(
        prompt_file=args.prompt_file,
        bend_function=bend_function,
        feature_fn=feature_fn,
        smoother=smoother,
        layer=args.layer,
        seed=args.seed,
        width=args.width,
        height=args.height,
        noise_period_frames=args.noise_period_frames,
        negative_prompt=args.negative_prompt,
    )
    print(">>> Engine ready. Press 'q' or Esc in the preview window to stop.")

    SR = LiveEngine.SAMPLING_RATE
    FPS = LiveEngine.FPS
    samples_per_frame = SR // FPS         # ~50 ms feature window
    waveform_window = SR // 4             # ~250 ms for the UI tap

    mic = MicCapture(
        sample_rate=SR,
        buffer_seconds=2.0,
        device=_parse_audio_device(args.audio_device),
    )

    recorder = None
    audio_wav_path = None
    if not args.no_save:
        recorder = FrameRecorder(
            output_root=PROJECT_ROOT / "live_video_outputs",
            fps=FPS,
            width=args.width,
            height=args.height,
        )
        staging = recorder.start()
        audio_wav_path = staging / "mic.wav"

    preview = LivePreview()

    mic.start(record_path=audio_wav_path)

    t0 = time.monotonic()
    frame_idx = 0
    try:
        while True:
            t_now = time.monotonic() - t0
            audio_slice = mic.get_latest(samples_per_frame)
            wave_view = mic.get_waveform(waveform_window)

            frame_img = engine.generate_frame(audio_slice, t_seconds=t_now)
            preview.update(frame_img, wave_view, hud=engine.last_hud())

            if recorder is not None:
                recorder.write(frame_img, frame_idx)
            frame_idx += 1

            if preview.should_quit():
                break
    finally:
        mic.stop()
        preview.close()

        if recorder is not None:
            # Build the descriptive filename parts mirroring audio2video.py:360.
            feature_name = getattr(feature_fn, "name", None) or args.feature
            parts = [
                bend_function.__name__,
                feature_name,
                f"layer{args.layer}",
                smoother.name,
                f"seed{args.seed}",
            ]
            try:
                out_path = recorder.finalize(audio_wav_path, filename_parts=parts)
            except Exception as exc:
                print(f"!!! ffmpeg mux failed: {exc}")
                out_path = None
            if out_path is not None:
                print(f">>> Saved video as {out_path}")
            else:
                print(">>> No frames written; nothing to save.")

    elapsed = time.monotonic() - t0
    print(f">>> Generated {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-6):.2f} fps avg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
