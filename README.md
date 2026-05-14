# Generating Music Reactive Videos by Applying Network Bending to Stable Diffusion
Luke Dzwonczyk, Carmine-Emanuele Cella, and David Ban | Journal of the Audio Engineering Society, June 2025

Abstract: In this paper we present the first steps towards the creation of a tool which enables artists to create music visualizations using pre-trained, generative, machine learning models. First, we investigate the application of network bending, the process of applying transforms within the layers of a generative network, to image generation diffusion models by utilizing a range of point-wise, tensor-wise, and morphological operators. We identify a number of visual effects that result from various operators, including some that are not easily recreated with standard image editing tools. We find that this process allows for continuous, fine-grain control of image generation which can be helpful for creative applications. Next, we generate music-reactive videos using Stable Diffusion by passing audio features as parameters to network bending operators. Finally, we comment on certain transforms which radically shift the image and the possibilities of learning more about the latent space of Stable Diffusion based on these transforms. This paper is an extended version of the paper ``Network Bending of Diffusion Models" which appeared in the 27th International Conference on Digital Audio Effects.


The code in this repository began as a fork of [Stream Diffusion](https://github.com/cumulo-autumn/StreamDiffusion).
## Installation instructions 

### Windows 

You must first clone the StreamDiffusion repo. This is only for the purpose of creating the necessary conda environment.
```
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
cd StreamDiffusion

conda create -n streamdiffusion python=3.10
conda activate streamdiffusion

pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
python setup.py develop easy_install streamdiffusion[tensorrt]

# some fixes to the StreamDiffusion install
conda install "numpy<2"
pip install transformers==4.28.0
pip install huggingface-hub==0.25.2

# optional: for EnCodecModel support in examples/audio2video/batch.py
pip install transformers==4.31.0

python -m streamdiffusion.tools.install-tensorrt

```
A simple way to see if StreamDiffusion was properly installed:
```
# while in the StreamDiffusion repo
cd examples
python benchmark/multi.py
```

Once you confirm that StreamDiffusion is properly running on your system, then you can install the necessary dependencies for this repository. Start by cloning this repo:
```
git clone https://github.com/dzluke/JAES2025.git
cd JAES2025

conda activate streamdiffusion
conda install -c conda-forge librosa
conda install scipy
```
Now you can run the code in this repo using the `streamdiffusion` conda environment.

If you get the error `OSError: cannot load library 'libsndfile.dll': error 0x7e`, try:
``` 
pip uninstall soundfile
pip install soundfile
```

## File Documentation

### `examples/audio2video/audio2video.py`

This file contains the main function `generate_video`, which creates music-reactive videos by applying network bending techniques to Stable Diffusion. It uses audio features to control the bending functions applied to the generative model, resulting in dynamic visualizations.

#### Key Function:
- **`generate_video(audio_path, prompt_file_path, layer, bend_function, audio_feature, smoothing_fn, seed, width=512, height=512)`**:
  - Generates a video based on the input audio and prompts.
  - Parameters:
    - `audio_path`: Path to the input audio file.
    - `prompt_file_path`: Path to the text file containing prompts.
    - `layer`: Layer of the model to apply bending.
    - `bend_function`: Function defining the bending operation.
    - `audio_feature`: Function to extract features from the audio.
    - `smoothing_fn`: Function to smooth the audio features.
    - `seed`: Random seed for reproducibility.
    - `width`, `height`: Dimensions of the output video.
  - Outputs:
    - A video file saved in the `video_outputs` directory.

### `examples/live_audio2video/` (experimental, live-input branch)

A streaming variant of `audio2video.py` that drives Stable Diffusion + network bending from **live microphone input** instead of a pre-recorded `.wav`. Audio captured at 48 kHz feeds the bending operator inside the U-Net frame-by-frame; a preview window shows the current diffusion frame stacked on top of a live waveform plus an HUD with the measured FPS and current feature values. Press `q` or `Esc` to stop. On exit, the rendered frames plus the captured mic audio are muxed into an mp4 under `live_video_outputs/`.

Extra dependency:
```
pip install sounddevice
# (soundfile is already pulled in by librosa, but if missing: pip install soundfile)
# For --matrix-mode opentsne only: pip install openTSNE
```

List audio inputs (no model load):
```
python examples/live_audio2video/live_audio2video.py --list-devices
```

End-to-end run with a single startup prompt (one line `0:00 : prompt text` in a file):
```
python examples/live_audio2video/live_audio2video.py \
    --prompt-file inputs/test.txt \
    --bend multiply --feature rms --layer 1 --seed 42
```

Files under `examples/live_audio2video/`:
- `live_audio.py` — mic capture, ring buffer, simultaneous wav recorder.
- `live_smoothing.py` — causal/streaming replacement for the offline `bending.smooth(...)`.
- `live_features.py` — scalar features (pass-through to `utils.bending`) + EnCodec matrix path with online dim-reduction (`pca` / `slice` / `opentsne`).
- `live_prompts.py` — `MM:SS : prompt` schedule, encoded once at startup, wall-clock-lerped per frame.
- `live_engine.py` — persistent `StreamDiffusionWrapper` + per-frame hot path (CLIP encoding bypassed via cached embeddings).
- `live_output.py` — `cv2.imshow` preview window + `FrameRecorder` (PNG staging + ffmpeg mux on quit).
- `live_audio2video.py` — argparse entry point.

Use `--layer 420` to disable the bending hook for an A/B baseline (matches the offline trick from `batch.py`). Use `--no-save` for preview-only runs that don't write to `live_video_outputs/`.

### `examples/audio2video/batch.py`

This file allows batch processing of multiple video generations by calling the `generate_video` function with different configurations.

#### Key Usage:
- Define a list of inputs, each specifying parameters for a video generation task.
- Call `audio2video.generate_video` for each input.

#### Example Input:
```python
inputs.append(
    {
         "audio": audio_path,
         "prompt": prompt_file_path,
         "layer": layer,
         "bend": bending_function,
         "feature": audio_feature,
         "smoothing": util.smooth(median_filter_kernel, envelope_follower_alpha, noise_gate_threshold),
         "seed": seed,
         "width": width,
         "height": height
    }
)
```

#### Execution:
Run the script to process all inputs sequentially and generate videos.






