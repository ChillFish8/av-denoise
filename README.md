# av-denoise

Faster and higher quality denoising for all.

This project was originally heavily inspired by [KNLmeansCL](https://github.com/Khanattila/KNLMeansCL) alongside 
FFmpeg's NLMeans implementation but is built to be a more standalone tool built and provide a more advanced denoising
experience and eventually growing beyond NLMeans.

av-denoise features **NLMeans**, **NLMeans-HQ** and **NL4D** algorithms offering significant advantages over existing
denoising tools.

## Table of contents

- [Features](#features)
- [Guide](#guide)
  - [Choosing an algorithm](#choosing-an-algorithm)
  - [Piped input and high bit depth](#piped-input-and-high-bit-depth)
- [NL4D tuning guide](#nl4d-tuning-guide)
  - [The NL4D dials](#the-nl4d-dials)
  - [What not to touch in NL4D](#what-not-to-touch-in-nl4d)
- [NLMeans tuning guide](#nlmeans-tuning-guide)
  - [Still too noisy](#still-too-noisy)
  - [Losing detail](#losing-detail)
  - [Common situations](#common-situations)
  - [What not to touch in nlmeans](#what-not-to-touch-in-nlmeans)
- [Benchmarks](#benchmarks)
  - [Algorithm defaults](#algorithm-defaults)
  - [Apples-to-apples spatial NL-means (strength 1.0)](#apples-to-apples-spatial-nl-means-strength-10)
  - [av-denoise feature cost (strength 1.0, default patch/search)](#av-denoise-feature-cost-strength-10-default-patchsearch)
  - [Bit depth cost](#bit-depth-cost)
- [Hardware support](#hardware-support)
  - [Notes about the JIT](#notes-about-the-jit)
    - [Configure compilation cache directory](#configure-compilation-cache-directory)
- [Installing](#installing)
  - [Cargo install](#cargo-install)
  - [From source](#from-source)
  - [As a library](#as-a-library)
- [Example commands](#example-commands)
- [Binary usage](#binary-usage)
  - [Global options](#global-options)
  - [Algorithm - `nl4d`](#nl4d-options)
  - [Algorithm - `nlmeans`](#nlmeans-options)

## Features

- **Simple tuning presets** - the `--preset` ladder (`veryfast` → `veryslow`) automatically adjusts denoiser settings
  without requiring you to modify many sets of parameters for every input.
- **NL4D Algorithm** - Offers best in class noise removal and detail retention while being faster than more standard
  V-BM3D algorithms and without the artefacts.
- **NLMeans-HQ Algorithm** - A smarter NLMeans denoiser able to process motion and detail extraction far better than
  traditional NLMeans.
- **Automatic noise handling** - Both _NL4D_ and _NLMeans-HQ_ offer automatic noise estimation removing the need to
  manually specify a tuned `sigma` parameter for every source, offering a simple to use `sigma-scale` flag for increasing
  or decreasing the relative denoise strength.
- **Temporal denoising with motion awareness** - up to 17-frame windows, per-neighbour
  block-match confidence, and opt-in on-GPU motion compensation.
- **Luma, chroma, and YUV444 kernels** - spatial or temporal, each plane individually tunable.
- **Prefilters** - on-GPU bilateral and NLM-pilot reference clips, or supply your own guide via
  the library.
  - *NLMeans family of denoisers only, not applicable to NL4D.
- **Library and binary** - y4m over a pipe, or direct file ingestion via FFMS2 with
  scene-parallel workers.
- **8, 10, and 12-bit** - depth is detected from the source and preserved on output. Tuning parameters are normalized
  across bit-depth.
- _**Fast!**_ - around **2x** FFmpeg's `nlmeans_opencl` at matched settings and **~1.4x** faster than V-BM3DHIP.
  - Piped input can't parallelise across scenes, so file input makes the best use of big GPUs.

## Guide

The defaults are measured, not guessed. Start with them, judge the result by eye, and change one
knob at a time.

```bash
av-denoise nl4d --input noisy.mkv | ffmpeg -f yuv4mpegpipe -i - -c:v libsvtav1 clean.mkv
```

That runs `nl4d` at the `base` preset: a 5-frame temporal window (radius of `2`), motion tracking
on, and fully automatic noise handling. The noise level and the grain's spatial correlation are
measured per scene, so most sources need nothing else.

Your main dial is `--preset`. Go up the ladder (`slow`, `veryslow`) for noisier sources or when
quality matters more than time. Go down (`fast`, `veryfast`) when speed is more important. Each
algorithm fills in its own knobs from it.

### Choosing an algorithm

**Use `nl4d` unless you need the speed.** It gives the best noise removal and the best detail
retention of anything here, on any kind of source. The only thing it asks for is time.

Both algorithms clean a frame using patches from itself and its neighbours. They differ in what
they do with the matches once they have them.

**`nl4d`** stacks the matching patches, transforms the whole stack together, and zeroes only the
coefficients that look like noise. Detail shared across the stack survives that, which is why it
keeps texture the alternative averages away. It always tracks motion and always needs a temporal
window, so it has no spatial-only mode.

**`nlmeans`** averages the patches that look alike. The averaging is what removes the noise, and
it is also what softens fine texture, because detail that is not repeated closely enough
somewhere else gets averaged away along with the grain.

Reach for `nlmeans` when one of these applies:

| Situation                                       | Reach for                     |
|-------------------------------------------------|-------------------------------|
| Throughput matters more than the result         | `nlmeans --variant fast`      |
| You want no temporal window at all              | `nlmeans --temporal-radius 0` |
| You need a prefilter, or per-plane `--strength` | `nlmeans`                     |

See [Algorithm defaults](#algorithm-defaults) for what each costs.

> [!IMPORTANT]
> You may _think_ you want a prefilter or no temporal window, but that is probably incorrect or operating on an
> assumption that does not apply to av-denoise. It is **highly** advised to ignore any advice recommending a prefilter
> for the NL4D or NLMeans-HQ algorithms in av-denoise, it is going to _hurt_ detail retention, not improve it.

### Piped input and high bit depth

`--input` also takes `-` (or `pipe:0`) to read a y4m stream from standard input, and `pipe:N` for
an inherited file descriptor:

```bash
ffmpeg -i noisy.mkv -pix_fmt yuv420p -f yuv4mpegpipe - \
  | av-denoise nlmeans --input - \
  | ffmpeg -f yuv4mpegpipe -i - -c:v libsvtav1 clean.mkv
```

Piped input has no scene detection, so the temporal window slides across the whole stream and
`--workers` does not apply.

ffmpeg will not write 10 or 12-bit y4m without `-strict -1`, so high-depth pipelines need it on
the *producing* side:

```bash
ffmpeg -i noisy.mkv -pix_fmt yuv420p10le -strict -1 -f yuv4mpegpipe - \
  | av-denoise nlmeans --input - \
  | ffmpeg -f yuv4mpegpipe -i - -c:v libsvtav1 clean.mkv
```

File input via `--input noisy.mkv` needs no such flag.

## NL4D tuning guide

```bash
av-denoise nl4d --input noisy.mkv | ffmpeg -f yuv4mpegpipe -i - -c:v libsvtav1 clean.mkv
```

`--preset` sets how many neighbouring frames are searched, from a 3-frame window at `fast` to a
17-frame one at `veryslow`. `veryfast` keeps the 3-frame window and narrows the spatial search
instead, since NL4D has nothing to group without neighbours.

### The NL4D dials

| Symptom                               | First thing to try          | Second                    |
|---------------------------------------|-----------------------------|---------------------------|
| Grain or noise still visible          | one preset higher           | `--lambda-ht` up a little |
| Fine texture getting scrubbed         | `--lambda-ht` down a little | `--sigma-scale 0.9`       |
| Result looks under-cleaned everywhere | `--sigma-scale 1.1`         | one preset higher         |
| Smearing or ghosting on motion        | one preset lower            | `--refine` up             |
| Too slow                              | `--spatial-radius` down     | one preset lower          |

**`--lambda-ht` is the main dial.** It sets how many standard deviations of estimated noise a
transform coefficient has to clear to survive, so raising it removes more noise and takes more
fine detail with it. The defaults, 5.3 for luma and 4.2 for chroma, were picked by eye on real
grain and deliberately biased toward keeping detail. Move in steps of about 0.3 and judge by eye.
`--luma-lambda-ht` and `--chroma-lambda-ht` pin one plane without touching the other.

**`--sigma-scale` is the other one**, and it does something different. `--lambda-ht` decides how
aggressive to be at a given noise level; `--sigma-scale` corrects the noise level itself. That
estimate also feeds the motion confidence scoring, so when the whole result reads uniformly
under- or over-cleaned, correcting the level fixes the cause rather than the symptom. When you
are happy with the level and just want a different trade, use `--lambda-ht`.

**`--spatial-radius` is the speed dial.** The centre-frame search covers `(2 * radius + 1)^2`
positions, so it dominates the work. Dropping it from 9 to 6 roughly halves the candidates, which
is exactly what `--preset veryfast` does.

### What not to touch in NL4D

- **`--sigma`** pins the noise level to a fixed value and turns the per-scene measurement off
  entirely. `--sigma-scale` keeps the measurement and nudges it, which is almost always what you
  actually want.
- **`--c-min`** only decides how much compute a frame costs. It never changes which patches are
  admitted once they are scored, so it is not a quality dial.
- **`--no-confidence-variance`** stops a poorly matched patch from being trusted less than a 
  well-matched one. It exists to isolate that mechanism in testing and calibration, not to improve output.
- **`--thsad-scale`, `--mc-blksize`, `--mc-overlap`, `--mc-search`, `--mc-pyramid-levels`** tune
  the motion machinery's internals, changing any of these will likely invalidate all other defaults.

## NLMeans tuning guide

```bash
av-denoise nlmeans --input noisy.mkv | ffmpeg -f yuv4mpegpipe -i - -c:v libsvtav1 clean.mkv
```

`--preset` picks the variant as well as the window: `veryfast` runs the `fast` variant with no
temporal window at all, and everything above it runs `hq` with a window from 3 frames at `fast`
to 17 at `veryslow`.

| Symptom                        | First thing to try                 | Second                      |
|--------------------------------|------------------------------------|-----------------------------|
| Grain or noise still visible   | one preset higher                  | `--hq-sigma-scale 1.1`      |
| Fine texture getting scrubbed  | `--hq-sigma-scale 0.9`             | lower `--strength` slightly |
| Smearing or ghosting on motion | `--motion-compensation`            | one preset lower            |
| Colour speckle survives        | `--chroma-strength` up             | -                           |
| Too slow                       | check the GPU is actually selected | one preset lower            |

### Still too noisy

Work through these in order.

1. **Go up a preset.** Noise and grain are independent frame to frame, so a deeper temporal
   window removes them more effectively than any strength increase. This is the strongest lever in
   the tool.
2. **Enable `--motion-compensation`** on footage with real movement, so the deeper window keeps
   finding usable matches instead of falling back to the current frame.
    * For **Anime** sources, you may not want to enable this option. Anime sources typically cope with
      high motion much more effectively, and introducing motion compensation can work against you.
3. **Nudge `--hq-sigma-scale` up** (try 1.1, then 1.2). This tells the denoiser the noise is a
   little stronger than it measured, and everything downstream (strength, patch matching, motion
   confidence) adapts together. Increase in small steps, judging by eye each time.

### Losing detail

The same dial works downward: `--hq-sigma-scale 0.9` tells it the source is cleaner than measured.
If texture is still being scrubbed after that, lower `--strength` a little.

The rule of thumb for choosing between the two:

- `--hq-sigma-scale` says *how noisy the source really is*
- `--strength` says *how aggressively to clean at that noise level*. 

**Prefer the sigma scale first.** The noise level also steers patch matching and motion confidence, so correcting it
fixes the cause rather than the symptom.

### Common situations

- **Old live action, heavy film grain.** Real grain is spatially correlated and hides from naive
  estimators. The estimator here measures it from frame-to-frame residuals, so start with plain
  `--preset slow --motion-compensation` before reaching for any manual value. If it still reads
  slightly weak to your eye, `--hq-sigma-scale 1.1` is the intended fix.
- **Mostly clean sources.** `fast` or `veryfast` is usually enough, and over-denoising a clean
  source only costs detail. If you only want the light grain layer gone, stay at a low preset and
  let the automatic strength do its thing.
- **Colour speckle.** Chroma already gets its own measured strength, but stubborn colour noise can
  take `--chroma-strength` above the default without touching luma.
- **Fast motion looks smeary.** Enable `--motion-compensation` first. If a scene still trails,
  drop one preset (a shallower window has less material to mis-blend).

### What not to touch in NLMeans

These exist for debugging, calibration work, and unusual sources. Reaching for them first usually
makes things worse.

- **`--hq-sigma`** pins the noise level to a fixed value, which disables the per-scene measurement
  entirely. `--hq-sigma-scale` keeps the measurement and nudges it, which is almost always what
  you actually want.
- **Raising `--strength` to fight leftover grain.** Grain that survives means the noise level read
  low, and extra strength scrubs detail before it removes grain. Fix the level
  (`--hq-sigma-scale`) instead.
- **`--hq-no-noise-floor`, `--hq-no-auto-strength`, `--hq-no-temporal-confidence`** switch off
  measured machinery. They are comparison and debugging switches, not quality options.
- **`--hq-thsad-scale`, `--mc-blksize`, `--mc-overlap`, `--mc-search`, `--mc-pyramid-levels`**
  tune the motion machinery's internals. The defaults are calibrated together.
- **`--prefilter`** (the NLM pilot and bilateral modes) changes what patch matching sees, and
  under the `hq` variant's calibrated automatic handling both modes measured neutral at best on
  default settings. They exist for experimentation and for library users supplying their own
  reference clip, not as a default quality upgrade.
- **`--search-radius` and `--patch-radius`** reshape the whole matching problem, and every other
  default is tuned around them. Cost grows quadratically with the search radius, and the presets
  already adjust these parameters based on exhaustive tuning.
- **`--device cpu`** selects a software device where the platform offers one, such as lavapipe
  under Vulkan. It is for testing the pipeline, not for real encodes.

## Benchmarks

Numbers below come from `scripts/bench_runs.py` (`just compare-perf`), which pipes
each tool to `ffmpeg -f null -` so the encoder is not measured. Throughput is
total frames divided by wall-clock elapsed.

- Input is a 3,450-frame 1080p FFV1 clip.
- `av-denoise` using the `vulkan` backend.
- Running on a `AMD AI Pro R9700` (AMD 9070XT equivalent) GPU.
- Elapsed time is measured around the whole process, so the one-off scene detection
  pass is inside every number.
- Absolute fps varies between benchmarking sessions (thermal state, background load,
  driver version). Treat comparisons within a table as meaningful; treat the same
  config's absolute fps across different tables as not directly comparable.

### Algorithm defaults

Every row uses `--channel-mode luma,chroma` and no tuning beyond the preset, so the
rows are directly comparable. nl4d always tracks motion, which is why the
motion-compensated `hq` row is here — that is the like-for-like comparison, not the
plain `hq` one.

| run                                          | preset |       fps | denoising   | detail retention | notes                                                                 |
|----------------------------------------------|--------|----------:|-------------|------------------|-----------------------------------------------------------------------|
| `nlmeans --variant fast`                     | `base` | **58.91** | medium      | low              | Traditional nlmeans algorithm                                         |
| `nl4d --preset fast`                         | `fast` |     48.04 | higher      | higher           | Better detail retention compared to V-BM3D (r=1)                      |
| `nlmeans --variant hq`                       | `base` |     47.34 | medium      | medium           | nlmeans with adaptive noise estimation and motion confidence (NLM-HQ) |
| `nlmeans --variant hq --motion-compensation` | `base` |     42.48 | high        | high             | NLM-HQ + block matching motion compensation                           |
| `nl4d`                                       | `base` |     42.39 | **highest** | **highest**      | Better detail retention compared to V-BM3D (r=2) and all NLM variants |

The two quality columns are judged by eye on real grain, not computed. They rank the
rows against each other and mean nothing outside this table.

Grouping patches across the temporal window costs about what motion-compensated
`nlmeans hq` costs at the same window size. One rung down the ladder, `nl4d --preset
fast` halves the window to 3 frames and lands on plain `nlmeans hq` throughput while
still tracking motion.

All five ran back to back in one session. Repeat passes agreed within 2% on every row
except `nlmeans --variant fast`, the least GPU-bound run of the five, which came in 12%
low on one pass out of four under background load.

Reproduce with (add `--device discrete:N` to pin a particular GPU):

```bash
just compare-perf -- --accelerators vulkan \
  --only av_default_nlmeans_fast,av_default_nlmeans_hq,av_default_nlmeans_hq_mc,av_fast_nl4d,av_default_nl4d
```

### Apples-to-apples spatial NL-means (strength 1.0)

The two tables below pin `--variant fast` at `veryfast`-preset settings, not the `base`
default, so they isolate one feature at a time rather than measuring a shipping config.

Matched patch and search sizes on both tools, av-denoise uses radii compared to
ffmpeg which takes the absolute size.

| patch / search | av-denoise (fps) | ffmpeg nlmeans_opencl (fps) | speedup |
|----------------|-----------------:|----------------------------:|--------:|
| p=5, r=11      |        **72.57** |                       30.25 |  ~2.40x |
| p=7, r=15      |        **42.41** |                       16.33 |  ~2.60x |
| p=9, r=15      |        **41.84** |                       16.26 |  ~2.57x |

> [!NOTE]
> av-denoise uses more sensible defaults compared to ffmpeg and enables the high-quality modes by default so
> the numbers you see here will not map directly to your own experience unless you explicitly configure it to
> match the settings to ffmpeg. (_NOT ADVISED_)

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

### Bit depth cost

Same clip, same settings, differing only in source depth. 10-bit moves twice
the bytes through decode, conversion, and the y4m output, so some of the gap is
I/O rather than denoising.

| source depth |       fps |
|--------------|----------:|
| 8-bit        | **91.19** |
| 10-bit       |     73.57 |

## Hardware support

The project supports the following accelerators/gpus:

- **AMD GPUs** (via the `rocm` or `vulkan` features)
- **Intel GPUs** (via the `vulkan` feature)
- **Nvidia GPUs** (via the `cuda` or `vulkan` features)
- **Apple Silicon** (via the `metal` feature)

There is no software backend. The collaborative filter aggregates its filtered patches through
atomic floating-point adds, and CubeCL's CPU runtime does not implement atomics. A software
*device* is still reachable with `--device cpu` where the platform provides one, such as lavapipe
under Vulkan.

### Notes about the JIT

It is important to note that `av-denoise` internally uses a JIT (Just In Time) compiler for its kernels; this means
that the kernels are compiled and optimised for your specific hardware _at runtime._ As such, the first a couple of
calls will have significant overhead as the system compiles, optimises and caches the kernels.

Additionally, because the kernels are compiled at runtime, whatever environment you run the tool in,
must also provide access to the hardware specific headers and compilers.

This primarily has the following impacts:

- The `rocm` backend requires the AMD HIP compiler and headers, typically vendored via the ROCm dev SDK.
- The `cuda` backend requires the NVIDIA CUDA headers and nvcc, typically vendored via the CUDA devel toolkit.
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
file input, y4m for piped input), so there's nothing else to pick between.

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

Two dials do almost everything: `--preset` for how hard to work, and a sigma scale to nudge the
measured noise level when your eye disagrees with it. Each example below changes one thing from
the defaults.

**Clean up a noisy file.** `nl4d` measures the noise per scene and picks its own threshold.

```bash
av-denoise nl4d --input noisy.mkv | ffmpeg -f yuv4mpegpipe -i - -c:v libsvtav1 clean.mkv
```

**Keep more detail, or take out more grain.** `--lambda-ht` is nl4d's main dial. Lower keeps more
detail, higher removes more noise. The default differs between luma and chroma, and either plane
can be pinned on its own with `--luma-lambda-ht` / `--chroma-lambda-ht`.

```bash
av-denoise nl4d --lambda-ht 4.5 --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v libsvtav1 clean.mkv
```

**A noisier source.** Go up the preset ladder. A deeper temporal window is the strongest lever in
the tool, for either algorithm.

```bash
av-denoise nl4d --preset slow --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v libsvtav1 clean.mkv
```

**Trade quality for throughput.** `nlmeans` is the faster family. Its `fast` variant does no noise
measurement at all.

```bash
av-denoise nlmeans --variant fast --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v libsvtav1 clean.mkv
```

**Still grainy under `nlmeans`.** Tell it the noise is a little stronger than it measured. Move in
steps of 0.1 and judge by eye.

```bash
av-denoise nlmeans --preset slow --hq-sigma-scale 1.1 --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v libsvtav1 clean.mkv
```

**Fine texture getting scrubbed.** The same dial works downward.

```bash
av-denoise nlmeans --hq-sigma-scale 0.9 --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v libsvtav1 clean.mkv
```

**Live action with real movement.** Motion compensation keeps the deeper window finding usable
matches instead of smearing. Anime is often better off without it.

```bash
av-denoise nlmeans --preset slow --motion-compensation --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v libsvtav1 clean.mkv
```

**Brightness only, for speed.** Both planes are cleaned by default. Narrow to luma when the colour
is already clean and you want the time back.

```bash
av-denoise nl4d --channel-mode luma --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v libsvtav1 clean.mkv
```

**Pick a specific GPU.** Both flags are global, so they work either side of the subcommand.

```bash
av-denoise --accelerators vulkan --device discrete:1 nl4d --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v libsvtav1 clean.mkv
```

<details>
<summary><b>Advanced examples</b> — fixed variant, manual per-plane strength, explicit backends</summary>

These pin `--preset veryfast` to select the `fast` variant, which makes `--strength` an absolute
value rather than the noise multiplier the `hq` variant applies. That trades away the per-scene
measurement, so treat them as calibration and debugging recipes rather than a starting point. If
your goal is a better-looking result, the dials above are the ones to reach for first — see
[What not to touch in nlmeans](#what-not-to-touch-in-nlmeans).

**Y/UV Denoise - ROCm/Vulkan - On GPU 1 - Light Denoise - Spatial - strength=luma:1.2,choma:1.2**
```bash
av-denoise nlmeans \
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
av-denoise nlmeans \
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
av-denoise nlmeans \
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
av-denoise nlmeans \
  --preset veryfast \
  --accelerators vulkan \
  --channel-mode yuv \
  --strength 2.0 \
  --input ./sample.mkv \
    | ffmpeg -hide_banner -loglevel info -y -f yuv4mpegpipe -i - -c:v ffv1 ./output.mkv
```

**Y/UV Denoise - Vulkan - On GPU 0 - Temporal (radius=2) + Motion Compensation - Anime / Heavy Motion**
```bash
av-denoise nlmeans \
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

</details>

## Binary usage

Two algorithms share one set of global flags. `nl4d` is the one to reach
for first, and gives up a spatial-only mode to do it. `nlmeans` is there
for when throughput matters more than the result, or when you need
something `nl4d` does not carry.

The tables below cover the flags worth reaching for. Each subcommand's
`--help` carries every flag with the full explanation.

### Global options

These are global, so they work on either side of the subcommand.

| Flag                               | What it does                                                                                                     | Default       |
|------------------------------------|------------------------------------------------------------------------------------------------------------------|---------------|
| `--preset <veryfast..veryslow>`    | Speed vs quality. Each algorithm fills in its own knobs from it, see the ladders below.                          | `base`        |
| `--channel-mode <luma,chroma,yuv>` | Which planes to clean. `yuv` is one fused pass and needs a YUV444 source.                                        | `luma,chroma` |
| `-A, --accelerators <list>`        | Backends to try, in order. Comma-separated, for example `cuda,vulkan`.                                           | `vulkan`      |
| `-d, --device <spec>`              | `default`, `discrete[:N]`, `integrated[:N]`, `virtual[:N]`, or `cpu`.                                            | `default`     |
| `--progress`                       | Draws a progress bar for file input. Off by default, because anything else writing to the terminal scrambles it. | off           |

Both subcommands also take:

| Flag                    | What it does                                                                                     | Default  |
|-------------------------|--------------------------------------------------------------------------------------------------|----------|
| `-i, --input <path\|->` | A path is opened with ffms2 and split by scene. `-` or `pipe:0` reads y4m from stdin.            | required |
| `-W, --workers <N>`     | How many scenes to clean in parallel. Trades GPU memory for throughput. Ignored for piped input. | `2`      |

### `nl4d` options

Groups matching 8x8 patches from across the whole temporal window into
one stack and shrinks the stack's transform coefficients together, rather
than filtering with non-local means first. Patches are searched both
inside the centre frame and around where each neighbour frame's motion
predicts they moved to.

Motion tracking is always on, and every preset keeps a temporal window,
which this algorithm needs. There is no NLM weighting pass, so none of
the NLM knobs above exist here.

What `--preset` fills in:

| Preset     | Temporal radius | Spatial radius |
|------------|-----------------|----------------|
| `veryfast` | 1               | 6              |
| `fast`     | 1               | 9              |
| `base`     | 2               | 9              |
| `slow`     | 4               | 9              |
| `veryslow` | 8               | 9              |

| Flag                    | What it does                                                                                                                                                                                                         | Default              |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| `--temporal-radius <N>` | How many neighbouring frames to search on each side, in 1..=8. More frames means more patches to group.                                                                                                              | from `--preset`      |
| `--lambda-ht <f>`       | How aggressively small transform coefficients are zeroed out. Higher removes more noise and more fine detail with it. This is the main quality dial. `--luma-lambda-ht` and `--chroma-lambda-ht` override one plane. | 5.3 luma, 4.2 chroma |
| `--spatial-radius <N>`  | Half-width of the candidate search inside the centre frame, in 1..=16. Most of the search work goes here, since the window covers `(2N+1)^2` positions.                                                              | from `--preset`      |
| `--sigma-scale <f>`     | Nudges the measured noise level, the same dial `nlmeans` spells `--hq-sigma-scale`.                                                                                                                                  | `1.0`                |

<details>
<summary><b>Expert flags</b> — calibration and debugging, not everyday tuning</summary>

| Flag                                                                 | What it does                                                                                                                                                                   | Default             |
|----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| `--refine <N>`                                                       | Half-width of the window searched around each neighbour frame's motion-predicted position, in 1..=4. Raise it when motion tracking lands close but not exact.                  | `2`                 |
| `--sigma <f>`                                                        | Pins the noise level in 8-bit units, turning the per-scene measurement off entirely.                                                                                           | measured            |
| `--thsad-scale <f>`                                                  | How badly a neighbour frame may match before its patches stop being trusted.                                                                                                   | `1.0`               |
| `--c-min <f>`                                                        | Confidence floor below which a whole neighbour block is skipped rather than scored. Only changes how much compute a frame costs, never which patches are admitted once scored. | `0.05`              |
| `--no-confidence-variance`                                           | Gives every patch the same noise estimate, instead of trusting a poorly matched one less.                                                                                      | off                 |
| `--mc-blksize`, `--mc-overlap`, `--mc-search`, `--mc-pyramid-levels` | Motion-search geometry. nl4d always tracks motion, so these are always live.                                                                                                   | `16`, `8`, `4`, `2` |

</details>

### `nlmeans` options

Compares small patches of pixels and averages the ones that look alike,
either inside a single frame or across a temporal window.

What `--preset` fills in:

| Preset     | Variant | Temporal radius | Search radius |
|------------|---------|-----------------|---------------|
| `veryfast` | `fast`  | 0               | 2             |
| `fast`     | `hq`    | 1               | 2             |
| `base`     | `hq`    | 2               | 2             |
| `slow`     | `hq`    | 4               | 4             |
| `veryslow` | `hq`    | 8               | 4             |

| Flag                    | What it does                                                                                                                                                                                   | Default         |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| `--variant <fast\|hq>`  | `fast` uses fixed weighting and is the cheapest option. `hq` measures the noise level per scene and matches its weighting to it.                                                               | from `--preset` |
| `--temporal-radius <N>` | How many neighbouring frames to use on each side. `0` cleans each frame on its own. This is the strongest lever in the tool.                                                                   | from `--preset` |
| `--hq-sigma-scale <f>`  | Nudges the measured noise level. Raise it when the result still looks noisy, lower it when texture is going. Move in steps of 0.1.                                                             | `1.0`           |
| `--strength <f>`        | How hard to filter. Under `hq` this multiplies the measured noise level, and its default is calibrated per plane and per radius. `--luma-strength` and `--chroma-strength` override one plane. | calibrated      |
| `--motion-compensation` | Tracks where blocks moved, so a deep window lines frames up instead of smearing them. Usually worth it on live action, often not on anime.                                                     | off             |
| `--prefilter <mode>`    | Compares patches against a cleaned reference instead of the noisy input. `nlm[:<scale>]` or `bilateral:<sigma_s>,<sigma_r>`. Costs one extra GPU pass per frame.                               | `none`          |

<details>
<summary><b>Expert flags</b> — calibration and debugging, not everyday tuning</summary>

| Flag                                                                 | What it does                                                                                                                                                                       | Default             |
|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| `--search-radius <N>`                                                | How far to look for matching patches inside a frame. Costs quadratically.                                                                                                          | from `--preset`     |
| `--patch-radius <N>`                                                 | Half-width of a compared patch, which covers `(2N+1)^2` pixels.                                                                                                                    | `4`                 |
| `--self-weight <f>`                                                  | How much weight the centre pixel gets in the average. `0` is pure NLM.                                                                                                             | `1.0`               |
| `--hq-sigma <f>`                                                     | Pins the noise level in 8-bit units, turning the per-scene measurement off entirely. `--hq-sigma-scale` keeps the measurement and nudges it, which is almost always what you want. | measured            |
| `--hq-thsad-scale <f>`                                               | How badly a neighbour may match before its contribution starts dropping.                                                                                                           | `1.0`               |
| `--hq-no-auto-strength`                                              | Reads `--strength` as an absolute value instead of a multiplier on the measured noise.                                                                                             | off                 |
| `--hq-no-noise-floor`                                                | Keeps the expected noise floor inside patch distances instead of subtracting it.                                                                                                   | off                 |
| `--hq-no-temporal-confidence`                                        | Weights every neighbour equally, however badly it matches.                                                                                                                         | off                 |
| `--mc-blksize`, `--mc-overlap`, `--mc-search`, `--mc-pyramid-levels` | Motion-search geometry. Only used with `--motion-compensation`.                                                                                                                    | `16`, `8`, `4`, `2` |

</details>
