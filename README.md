# av-denoise

Fast and efficient NLMEANS video denoising using CubeCL.

This project is heavily inspired by [KNLmeansCL](https://github.com/Khanattila/KNLMeansCL) alongside FFmpeg's nlmeans 
implementation but is built to be a more standalone tool and also make use or more modern tooling to better 
leverage modern hardware instead of relying on the now rather outdated OpenCL.

## Features

- Library and binary offering.
  * The binary supports both STDIN (y4m) and FFMS2 ingestion and emits y4m frames to STDOUT.
- **Spatial** and **Temporal** support
- **Luma**, **Chroma** and **YUV*** specific denoising kernels.
   * YUV is for YUV 4:4:4 only.
- Adjustable nlmeans tuning paramters with sensible defaults.
- **Prefilter support** for more accurate denoising with less detail loss.
   * Includes an on-gpu bilateral filter out of the box
   * You can specify the reference frame yourself using the library rather than CLI.
- _**Fast!**_ - Around **2x** faster than FFmpeg's OpenCL implementation.
   * Be aware that the `STDIN` mode for the binary cannot fully utilise larger modern GPUs, it will
     likely be just as fast as FFmpeg using much less GPU compute, but we cannot parallelize across scenes.

## Hardware support

The project supports the following accelerators/gpus:

- **AMD GPUs** (via the `rocm` or `vulkan` features)
- **Intel GPUs** (via the `vulkan` feature)
- **Nvidia GPUs** (via the `cuda` or `vulkan` features)
- **Apple Silicon** (via the `metal` feature)
- **CPU** (via the `cpu` feature)
  * _WARNING! The CPU backend within CubeCL is still very new, and is not as optimised as a manually written kernel.<br/>
    As such, I do not recommend using this backend outside of testing._

### Notes about the JIT

It is important to note that `av-denoise` internally uses a JIT (Just In Time) compiler for its kernels; this means
that the kernels are compiled and optimised for your specific hardware _at runtime._ As such, the first a couple of
calls will have significant overhead as the system compiles, optimises and caches the kernels.

Additionally, because the kernels are compiled at runtime, whatever environment you run the tool in,
must also provide access to the hardware specific headers and compilers.

This primarily has the following impacts:

- The `rocm` backend requires the AMD HIP compiler and headers, typically vendored via the ROCm dev SDK.
- The `cuda` backend requires the NVIDIA CUDA headers and nvcc, typically vendored via the CUDA devel toolkit.
- The `cpu` backend should not require any special dependencies directly, as it should already be vendored.
- The `vulkan` and `metal` backends should "just work" on non-containerised hosts. If you are building for
  docker, then the vulkan backend requires `vulkan-icd-loader` and then the relevant GPU specific driver,
  i.e. `vulkan-radeon` or `vulkan-intel`.

Since both the CUDA and ROCm backends are very heavy in terms of dependencies, I recommend just using the `vulkan`
backend for those devices. It should be more or less the same performance, without all the library headache.

## Binary Usage

```angular2html
Fast and efficient video denoising

Usage: av-denoise [OPTIONS] <COMMAND>

Commands:
  file   Denoise a video file with scene-aware parallel processing
  stdin  Denoise a y4m stream from stdin and emit y4m to stdout
  help   Print this message or the help of the given subcommand(s)

Options:
  -a, --algorithm <ALGORITHM>
          Denoising algorithm.
          
          Only `nlmeans` is currently implemented.
          
          [default: nlmeans]

  -A, --accelerators <ACCELERATORS>
          Hardware accelerator priority list (comma-delimited).
          
          The runtime is selected by probing each accelerator in order and taking the first one that initialises successfully. If none work, the binary exits with an error.
          
          Defaults to every backend the binary was compiled with.
          
          [default: vulkan cpu]

  -d, --device <DEVICE>
          Specific device to bind to on the selected accelerator.
          
          Accepted forms:
          
          `default` — backend-chosen default device.
          
          `discrete[:N]` — discrete GPU at ordinal N (default 0). Honoured by CUDA, ROCm, and wgpu.
          
          `integrated[:N]` — integrated GPU at ordinal N. wgpu only.
          
          `virtual[:N]` — virtual GPU at ordinal N. wgpu only.
          
          `cpu` — software/CPU device.
          
          [default: default]

      --channel-mode <CHANNEL_MODE>
          Which channels of each frame to denoise (comma-delimited).
          
          `luma` denoises only Y; `chroma` only U/V at the source's native subsampled resolution. `luma,chroma` runs both as two independent denoisers (full-res Y + subsampled UV).
          
          `yuv` invokes the library's fused 3-channel kernel in one pass. It requires a YUV444 source and cannot be combined with any other mode.

          Possible values:
          - luma:   Denoise only the luma (Y) plane. Chroma is passed through
          - chroma: Denoise only the chroma (U, V) planes. Luma is passed through
          - yuv:    Single-pass fused YUV denoising via the library's 3-channel kernel. Requires a YUV444 source and cannot be combined with other modes
          
          [default: luma]

      --prefilter <PREFILTER>
          Reference clip used for NLM weight calculation.
          
          `none` disables prefiltering and uses the noisy input directly for both weight calculation and pixel accumulation.
          
          `bilateral:<sigma_s>,<sigma_r>` runs an on-GPU bilateral prefilter; `sigma_s` is the spatial sigma in pixels and `sigma_r` is the range sigma in `[0, 1]` intensity units. A sensible starting point is `bilateral:3.0,0.02`.
          
          [default: none]

      --temporal-radius <TEMPORAL_RADIUS>
          Temporal radius for temporal-aware denoising.
          
          `0` (default) runs spatial-only denoising — each output frame depends only on the matching input frame. Values `> 0` enable temporal denoising over a `2 * radius + 1` frame window centred on the current frame; higher values give stronger noise reduction at the cost of latency and memory.
          
          In `file` mode, temporal context is reset at every scene boundary detected by av-scenechange, so increasing the radius never blends frames across cuts.
          
          [default: 0]

      --search-radius <SEARCH_RADIUS>
          Override NLM search-window radius. Library default: 2.
          
          Higher values find more candidate patches at the cost of work quadratic in this value. Bounded by the library's `MAX_SEARCH_RADIUS`.

      --patch-radius <PATCH_RADIUS>
          Override NLM patch radius. Library default: 4.
          
          Patch is `(2*patch_radius + 1)` square. Larger patches preserve structure better at the cost of higher GPU memory. Bounded by the library's `MAX_PATCH_RADIUS`.

      --strength <STRENGTH>
          Override NLM filter strength (sigma). Library default: 1.2.
          
          Higher = more smoothing. Must be finite and > 0. Acts as the shared default for both planes; `--luma-strength` and `--chroma-strength` take precedence when set.

      --luma-strength <LUMA_STRENGTH>
          Override strength for the luma denoiser only. Falls back to `--strength` (or the library default) when unset. Ignored when luma isn't being denoised or when `--channel-mode yuv` is used (the fused kernel can't tune planes independently)

      --chroma-strength <CHROMA_STRENGTH>
          Override strength for the chroma denoiser only. Falls back to `--strength` (or the library default) when unset. Ignored when chroma isn't being denoised or when `--channel-mode yuv` is used

      --self-weight <SELF_WEIGHT>
          Override the centre pixel's self-weight in NLM averaging. Library default: 1.0. Must be finite and >= 0

  -h, --help
          Print help (see a summary with '-h')
```