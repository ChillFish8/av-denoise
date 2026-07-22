# av-denoise

Fast and efficient NLMEANS video denoising using CubeCL.

This project is heavily inspired by [KNLmeansCL](https://github.com/Khanattila/KNLMeansCL) alongside FFmpeg's nlmeans 
implementation but is built to be a more standalone tool and also make use or more modern tooling to better 
leverage modern hardware instead of relying on the now rather outdated OpenCL.

## Table of contents

- [Features](#features)
- [Benchmarks](#benchmarks)
  - [Apples-to-apples spatial NL-means (strength 1.0)](#apples-to-apples-spatial-nl-means-strength-10)
  - [av-denoise feature cost (strength 1.0, default patch/search)](#av-denoise-feature-cost-strength-10-default-patchsearch)
- [Hardware support](#hardware-support)
  - [Notes about the JIT](#notes-about-the-jit)
    - [Configure compilation cache directory](#configure-compilation-cache-directory)
- [Installing](#installing)
  - [Cargo install](#cargo-install)
  - [From source](#from-source)
  - [As a library](#as-a-library)
- [Example commands](#example-commands)
- [Binary usage](#binary-usage)

## Features

- **`--preset` speed/quality ladder** (`veryfast`, `fast`, `base`, `slow`, `veryslow`) bundling
  algorithm, temporal radius, and search radius into a single dial.
   * `base` is the default. `veryfast` matches this tool's original default behavior.
- Library and binary offering.
  * The binary supports both STDIN (y4m) and FFMS2 ingestion and emits y4m frames to STDOUT.
- **Spatial** and **Temporal** support
- **Luma**, **Chroma** and **YUV*** specific denoising kernels.
   * YUV is for YUV 4:4:4 only.
- Adjustable nlmeans tuning paramters with sensible defaults.
- **Prefilter support** for more accurate denoising with less detail loss.
   * Includes an on-gpu bilateral filter out of the box
   * You can specify the reference frame yourself using the library rather than CLI.
- **`nlmeans-hq` algorithm** (`--algorithm nlmeans-hq`) for quality-focused denoising.
   * Measures the noise level automatically per scene (EMA-smoothed across frames) and scales its
     strength to it, a multiplier whose default is calibrated per temporal radius and per plane
     (luma or chroma). Override with `--hq-sigma` if the automatic estimate misjudges a source, or
     fall back to an absolute strength with `--hq-no-auto-strength`.
   * Subtracts the expected noise floor from patch distances so matching isn't skewed by grain
     (`--hq-no-noise-floor` to keep it in).
   * Gates each temporal neighbour's contribution by how well it block-matches the centre frame, so
     occluded or changed content doesn't get blurred in (`--hq-no-temporal-confidence`,
     `--hq-thsad-scale`).
   * Adds an opt-in NLM-spatial pilot prefilter (`--prefilter nlm[:<strength_scale>]`) alongside the
     bilateral one.
- **Motion compensation** for high-quality temporal denoising on heavy motion.
   * MVTools-inspired hierarchical block matching, fully on-GPU.
   * Enabled with `--motion-compensation`, tuned via `--mc-blksize`,
     `--mc-overlap`, `--mc-search`, `--mc-pyramid-levels`.
   * The motion-tracking strategy adapts automatically to `--temporal-radius`.
- _**Fast!**_ - Around **2x** faster than FFmpeg's OpenCL implementation.
   * Be aware that the `STDIN` mode for the binary cannot fully utilise larger modern GPUs, it will
     likely be just as fast as FFmpeg using much less GPU compute, but we cannot parallelize across scenes.

## Benchmarks

Numbers below come from `scripts/bench_runs.py` (`just compare-perf`), which pipes
each tool to `ffmpeg -f null -` so the encoder is not measured. Throughput is
total frames divided by wall-clock elapsed.

- Input is a 3,450-frame 1080p FFV1 clip.
- `av-denoise` using the `vulkan` backend.
- Running on a `AMD AI Pro R9700` (AMD 9070XT equivalent) GPU.
- These tables measure the `nlmeans` algorithm at `veryfast`-preset settings, not the
  `base` default.

### Apples-to-apples spatial NL-means (strength 1.0)

Matched patch and search sizes on both tools, av-denoise uses radii compared to
ffmpeg which takes the absolute size.

| patch / search | av-denoise (fps) | ffmpeg nlmeans_opencl (fps) | speedup |
|----------------|-----------------:|----------------------------:|--------:|
| p=5, r=11      |        **72.57** |                       30.25 |  ~2.40x |
| p=7, r=15      |        **42.41** |                       16.33 |  ~2.60x |
| p=9, r=15      |        **41.84** |                       16.26 |  ~2.57x |

### av-denoise feature cost (strength 1.0, default patch/search)

All luma+chroma. Spatial baseline is the reference. _Lower fps = more work._

| run                              |   fps | notes                               |
|----------------------------------|------:|-------------------------------------|
| spatial baseline                 | 97.25 | `--temporal-radius 0`               |
| spatial + bilateral prefilter    | 93.50 | adds one on-GPU pass per frame      |
| temporal r=1                     | 72.73 | 3-frame window                      |
| temporal r=2                     | 62.07 | 5-frame window                      |
| temporal r=1 + motion comp       | 64.03 | hierarchical block matching enabled |
| temporal r=2 + motion comp       | 54.29 |                                     |
| temporal r=1 + prefilter         | 69.58 |                                     |
| full r=1 (temporal+MC+prefilter) | 60.97 |                                     |
| full r=2 (temporal+MC+prefilter) | 52.18 |                                     |

Reproduce with `just compare-perf` (config: `scripts/bench_runs.toml`).

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

#### Configure compilation cache directory

Set `AV_DENOISE_COMPILATION_CACHE=/some/dir` to redirect the compiled-kernel and autotune caches to a specific
directory (overrides whatever is in `cubecl.toml`). 
Library users can call `av_denoise::apply_compilation_cache_env()` before `Denoiser::create` to honor the same 
env var from their own binary.

## Installing

`av-denoise` is available both in library _and_ binary format, by default only the `vulkan` feature
is enabled, since that is typically the default accelerators you will want to use.

When compiling the binary, enable the `binary` feature. It pulls in both ingestion paths (FFMS2 for
`file` mode, y4m for `stdin` mode), so there's nothing else to pick between.

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

These use `--preset veryfast` to pin the plain `nlmeans` algorithm, so `--strength`
stays an absolute value rather than the `nlmeans-hq` noise multiplier `base` (the
default preset) would apply.

**Y/UV Denoise - ROCm/Vulkan - On GPU 1 - Light Denoise - Spatial - strength=luma:1.2,choma:1.2**
```bash
av-denoise file \
  --preset veryfast \
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
  --preset veryfast \
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
  --preset veryfast \
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
  --preset veryfast \
  --accelerators vulkan \
  --channel-mode yuv \
  --strength 2.0 \
  --input ./sample.mkv \
    | ffmpeg -hide_banner -loglevel info -y -f yuv4mpegpipe -i - -c:v ffv1 ./output.mkv
```

**Y/UV Denoise - Vulkan - On GPU 0 - Temporal (radius=2) + Motion Compensation - Anime / Heavy Motion**
```bash
av-denoise file \
  --preset veryfast \
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

```text
Fast and efficient video denoising

Usage: av-denoise [OPTIONS] <COMMAND>

Commands:
  file   Denoise a video file, splitting work by scene
  stdin  Denoise a y4m stream coming in on stdin, writing y4m on stdout
  help   Print this message or the help of the given subcommand(s)

Options:
      --preset <PRESET>
          Speed vs quality dial.
          
          `veryfast` is the fastest and lowest-quality end of the dial. It runs the plain `nlmeans` algorithm with no temporal window and matches this tool's original default behavior.
          
          `fast`, `base`, `slow`, and `veryslow` all run `nlmeans-hq` and widen the temporal window going up the list, from a 1-frame radius at `fast` to an 8-frame radius at `veryslow`. `slow` and `veryslow` also widen the search radius.
          
          `base` is the default.
          
          [default: base]

  -a, --algorithm <ALGORITHM>
          Denoising algorithm to run.
          
          `nlmeans` is the fast path. `nlmeans-hq` is a quality-focused variant that calibrates its weighting to the noise level, measured automatically per frame (see `--hq-sigma` to override).
          
          Defaults to whatever `--preset` selects.

  -A, --accelerators <ACCELERATORS>
          Which hardware backends to try, in order of preference.
          
          The first backend that initialises is used. If none work the program exits with an error.
          
          The list is comma-separated, for example `vulkan,cpu`.
          
          [default: vulkan]

  -d, --device <DEVICE>
          Which device to use on the chosen backend.
          
          Accepted values:
          
          `default` lets the backend pick.
          
          `discrete[:N]` picks the Nth discrete GPU (default 0). Works on CUDA, ROCm, and Vulkan.
          
          `integrated[:N]` picks the Nth integrated GPU. Vulkan only.
          
          `virtual[:N]` picks the Nth virtual GPU. Vulkan only.
          
          `cpu` uses the software backend.
          
          [default: default]

      --channel-mode <CHANNEL_MODE>
          Which planes of the video to clean (comma-separated).
          
          `luma` cleans only the brightness plane.
          
          `chroma` cleans only the colour planes at their native size.
          
          `luma,chroma` cleans both as two independent passes, which is usually what you want for noisy footage.
          
          `yuv` cleans all three planes in one fused pass.
          
          `yuv` needs a YUV444 source and cannot be combined with the other modes.

          Possible values:
          - luma:   Clean only the brightness plane (Y). Colour passes through
          - chroma: Clean only the colour planes (U, V). Brightness passes through
          - yuv:    Clean all three planes together in one pass. Needs a YUV444 source and cannot be combined with the other modes
          
          [default: luma]

      --prefilter <PREFILTER>
          Reference image used when comparing patches.
          
          Omitted (the default) means no prefilter, for both `nlmeans` and `nlmeans-hq`.
          
          `none` forces the noisy input directly (the cheapest option). This is the same as leaving the flag unset.
          
          `nlm` or `nlm:<strength_scale>` runs a windowed spatial NLM pass first and compares patches against that cleaner image. `strength_scale` multiplies the main pass strength for the pilot pass. Bare `nlm` uses the calibrated default.
          
          `bilateral:<sigma_s>,<sigma_r>` runs a quick on-GPU bilateral blur first, then compares patches against that cleaner image.
          
          `sigma_s` is the spatial blur radius in pixels.
          
          `sigma_r` is the colour-similarity threshold in `[0, 1]`.
          
          A good starting point is `bilateral:3.0,0.02`.
          
          Prefiltering keeps more detail at the cost of one extra GPU pass per frame.

      --temporal-radius <TEMPORAL_RADIUS>
          How many neighbouring frames to look at on each side when cleaning a frame.
          
          `0` means no temporal blending. Each frame is cleaned on its own.
          
          Values above `0` look at that many frames before and after the current one.
          
          Larger values give stronger cleanup but use more memory and add latency.
          
          In `file` mode this is reset at every scene change, so raising it never causes blending across cuts.
          
          Defaults to whatever `--preset` selects.

      --search-radius <SEARCH_RADIUS>
          How far away to look for similar patches inside a frame.
          
          Larger values find more matches but cost quadratically more work.
          
          Defaults to whatever `--preset` selects.

      --patch-radius <PATCH_RADIUS>
          Size of each patch being compared. The patch is `(2*patch_radius + 1)` pixels square.
          
          Larger patches preserve fine structure better but cost more GPU memory. Library default is 4.

      --strength <STRENGTH>
          Cleaning strength. Higher numbers smooth more.
          
          Must be a finite number greater than 0.
          
          The default depends on the algorithm. `nlmeans` defaults to 1.2. `nlmeans-hq` interprets strength as a multiplier on the measured noise level. Its default is calibrated automatically, adapting to the temporal radius and to which plane (luma or chroma) is being denoised, so lower and higher radii each get their own measured value.
          
          This value applies to both planes unless `--luma-strength` or `--chroma-strength` is set.

      --luma-strength <LUMA_STRENGTH>
          Strength override for the brightness plane only.
          
          Falls back to `--strength` (or the library default) when not set.
          
          Ignored when luma is not being denoised, or when `--channel-mode yuv` is used.

      --chroma-strength <CHROMA_STRENGTH>
          Strength override for the colour planes only.
          
          Falls back to `--strength` (or the library default) when not set.
          
          Ignored when chroma is not being denoised, or when `--channel-mode yuv` is used.

      --self-weight <SELF_WEIGHT>
          How much weight to give the centre pixel itself when averaging.
          
          Library default is 1.0. Must be a finite number `>= 0`.
          
          Setting to 0 gives pure NLM (centre pixel only counts if a similar patch was found nearby).

      --hq-sigma <HQ_SIGMA>
          How noisy the source is. Leave it unset for almost all uses.
          
          The noise level is measured automatically per scene when this is not set. Set it only when the automatic estimate misjudges a source and you want to pin the value.
          
          Small values mean light grain and larger values mean heavier noise. `3` is subtle grain, `6` is clearly visible grain, `12` and up is heavy noise.

      --hq-no-auto-strength
          Treat `--strength` as an absolute value instead of a multiplier on `--hq-sigma`

      --hq-no-noise-floor
          Keep the expected-noise floor inside patch distances instead of subtracting it

      --hq-no-temporal-confidence
          Disable per-block temporal confidence weighting for `nlmeans-hq`.
          
          By default HQ block-matches each temporal neighbour against the centre frame and lets a poor match suppress that neighbour's contribution, instead of blurring in occluded or changed content. Setting this applies temporal weights uniformly no matter how well a neighbour matches.
          
          Only takes effect when `--temporal-radius` is above 0.

      --hq-thsad-scale <HQ_THSAD_SCALE>
          Multiplier on the per-block mismatch threshold temporal confidence weighting tolerates before a neighbour's contribution starts dropping.
          
          Higher values tolerate larger mismatches. Library default is 1.0. Ignored when `--hq-no-temporal-confidence` is set.

      --motion-compensation
          Turn on motion compensation for temporal denoising.
          
          When the camera or content moves between frames, the brightness at the same `(x, y)` is different content in each frame.
          
          Without help, temporal cleanup will blur moving edges.
          
          Motion compensation looks at where each block of pixels moved between frames, then shifts neighbour frames to line up with the current frame before cleaning.
          
          This keeps detail sharp on anime, fast pans, and action footage.
          
          The tracking strategy adapts automatically to `--temporal-radius`.
          
          Has no effect when `--temporal-radius 0`.

      --mc-blksize <MC_BLKSIZE>
          Size of each motion-search block, in pixels. Must be even.
          
          Larger blocks are more stable but track motion less accurately on small details.
          
          [default: 16]

      --mc-overlap <MC_OVERLAP>
          How many pixels neighbouring motion blocks may overlap.
          
          Must be less than `--mc-blksize`. Higher overlap smooths the transitions between blocks but does more work.
          
          [default: 8]

      --mc-search <MC_SEARCH>
          How many pixels of motion to search for at the finest level.
          
          The coarse pyramid pass reaches further (search radius times 2 for a 2-level pyramid), so for typical content the default is fine.
          
          Raise it for very fast motion.
          
          [default: 4]

      --mc-pyramid-levels <MC_PYRAMID_LEVELS>
          How many levels the motion-search pyramid uses.
          
          `1` does a single full-resolution search (cheaper, weaker on large motion).
          
          `2` (default) does a coarse pass on a half-size image first, then refines at full resolution.
          
          This handles much larger motion at modest extra cost.
          
          [default: 2]

  -h, --help
          Print help (see a summary with '-h')
```
