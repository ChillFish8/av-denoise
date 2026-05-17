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
- **Motion compensation** for high-quality temporal denoising on heavy motion.
   * MVTools-inspired hierarchical block matching, fully on-GPU.
   * Warps temporal neighbours into spatial alignment with the centre frame
     before NLM weighting — preserves detail on anime, fast pans, and action.
   * Enabled with `--motion-compensation`; tuned via `--mc-blksize`,
     `--mc-overlap`, `--mc-search`, `--mc-pyramid-levels`.
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

## Installing

`av-denoise` is available both in library _and_ binary format, by default only the `cpu` and `vulkan` features
are enabled, since they are typically the default accelerators you will want to use.

When compiling the binary, you want to enable the `binary` feature at minimum, but I recommend for most users
to enable the `binary-full` feature instead if you are ever unsure about how you are going to be ingesting frames.

The following (non-accelerator) features are available:

- `binary` - Enables the dependencies and code required to compile `av-denoise` as a binary.
   * This pulls in `ffms2` as hard dependencies. This means you must install `ffms2` before you can compile and link
     the binary.

### Cargo install

```bash
cargo install --locked av-denoise --features binary
```

### From source

```bash
git clone https://github.com/ChillFish8/av-denoise.git
cargo build --release --features binary
cp ./target/release/av-denoise ./av-denoise
```

### As a library

```bash
cargo add av-denoise
```

## Example commands

**Y/UV Denoise - ROCm/Vulkan - On GPU 1 - Light Denoise - Spatial - strength=luma:1.2,choma:1.2**
```bash
av-denoise file \
  --accelerators rocm,vulkan \
  --device discrete:1 \
  --channel-mode luma,chroma \
  --strength 1.2 \
  --input ./sample.mkv \
    | ffmpeg -hide_banner -loglevel info -y -f yuv4mpegpipe -i - -c:v ffv1 ./output.mkv
```

**Y/UV Denoise - Vulkan - On iGPU 0 - Split Denoise - Temporal (radius=1) - strength=luma:2.0,choma:1.5**
```bash
av-denoise file \
  --accelerators vulkan \
  --device integrated:0 \
  --channel-mode luma,chroma \
  --temporal-radius 1 \
  --luma-strength 2.0 \
  --chroma-strength 1.5 \
  --input ./sample.mkv \
    | ffmpeg -hide_banner -loglevel info -y -f yuv4mpegpipe -i - -c:v ffv1 ./output.mkv
```

**Y-Only Denoise - Metal - On GPU 0 - Heavy Denoise - Spatial - strength=luma:3.0**
```bash
av-denoise file \
  --accelerators metal \
  --device discrete:0 \
  --channel-mode luma \
  --strength 3.0 \
  --input ./sample.mkv \
    | ffmpeg -hide_banner -loglevel info -y -f yuv4mpegpipe -i - -c:v ffv1 ./output.mkv
```

**YUV Fused Denoise - Vulkan - On Default GPU - Medium Denoise - Spatial - strength=yuv:2.0**
```bash
av-denoise file \
  --accelerators vulkan \
  --channel-mode yuv \
  --strength 2.0 \
  --input ./sample.mkv \
    | ffmpeg -hide_banner -loglevel info -y -f yuv4mpegpipe -i - -c:v ffv1 ./output.mkv
```

**Y/UV Denoise - Vulkan - On GPU 0 - Temporal (radius=2) + Motion Compensation - Anime / Heavy Motion**
```bash
av-denoise file \
  --accelerators vulkan \
  --device discrete:0 \
  --channel-mode luma,chroma \
  --temporal-radius 2 \
  --motion-compensation \
  --strength 1.5 \
  --input ./anime.mkv \
    | ffmpeg -hide_banner -loglevel info -y -f yuv4mpegpipe -i - -c:v ffv1 ./output.mkv
```

## Binary usage

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

      --motion-compensation
          Enable MVTools-style motion compensation for temporal denoising.

          Estimates per-block motion between the centre frame and each temporal neighbour, then warps neighbours into spatial alignment before NLM weighting. Greatly improves quality on heavy-motion content (anime, fast pans, action) where the temporal path would otherwise smear edges or collapse to spatial-only.

          No-op when `--temporal-radius 0`.

      --mc-blksize <MC_BLKSIZE>
          Motion-compensation block size in pixels (must be even).

          [default: 16]

      --mc-overlap <MC_OVERLAP>
          Motion-compensation block overlap in pixels. Must satisfy `overlap < blksize` so the step (`blksize - overlap`) is positive.

          [default: 8]

      --mc-search <MC_SEARCH>
          Motion-compensation search radius at the finest pyramid level (in pixels). The coarse pass uses the same radius on the `/2` image so the effective reach is doubled.

          [default: 4]

      --mc-pyramid-levels <MC_PYRAMID_LEVELS>
          Pyramid levels for hierarchical motion estimation. `1` disables the coarse pass; `2` adds a `/2` coarse pass that seeds the fine refinement.

          [default: 2]

  -h, --help
          Print help (see a summary with '-h')
```