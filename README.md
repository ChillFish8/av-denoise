# av-denoise

Faster and higher quality denoising for all.

This project was originally heavily inspired by [KNLmeansCL](https://github.com/Khanattila/KNLMeansCL) alongside 
FFmpeg's NLMeans implementation but is built to be a more standalone tool built and provide a more advanced denoising
experience and eventually growing beyond NLMeans.

av-denoise features **NLMeans**, **NLMeans-HQ** and **NL4D** algorithms offering significant advantages over existing
denoising tools.

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
- **Temporal denoising with motion awareness** - up to 17-frame windows, per-neighbour block-match confidence, 
  and opt-in on-GPU motion compensation.
- **Luma, chroma, and YUV444 kernels** - spatial or temporal, each plane individually tunable.
- **Library and binary** - y4m over a pipe, or direct file ingestion via FFMS2 with
  scene-parallel workers.
- **8, 10, and 12-bit** - depth is detected from the source and preserved on output. Tuning parameters are normalized
  across bit-depth.
- _**Fast!**_ - around **2x** FFmpeg's `nlmeans_opencl` at matched settings and **~1.3x** faster than V-BM3DHIP.
  - Piped input can't parallelize across scenes, so file input makes the best use of big GPUs.

---


## Tutorials

### [Tutorial for CLI](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/TUTORIAL-CLI.md)

##### [Example commands](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/TUTORIAL-CLI.md#example-commands)

##### [Binary usage and flag reference](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/TUTORIAL-CLI.md#binary-usage)


### [Tutorial for VapourSynth](https://github.com/ChillFish8/av-denoise/blob/HEAD/av-denoise-vs/README.md)


## Installing Summary

`av-denoise` is available both in library, VapourSynth plugin and binary format.

### [Installation for CLI](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/TUTORIAL-CLI.md#installation)

### [Installation for VapourSynth](https://github.com/ChillFish8/av-denoise/blob/HEAD/av-denoise-vs/README.md#installing)

### [Container images](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/TUTORIAL-CLI.md#container-images)

Images are published to GHCR as `ghcr.io/chillfish8/av-denoise:<backend>-<version>`, one per accelerator
backend (`vulkan`, `cuda`, `rocm`).

### As a library

```bash
cargo add av-denoise
```

## [Choosing an algorithm](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/CHOOSING_AN_ALGORITHM.md)

A guide to help you understand what each algorithm offers and pick which one is best for you.


## ["Don't do this!"](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/DONT_DO_THIS.md)

Common footguns to avoid and why.


## Tuning guides

There are dedicated docs for how to adjust each algorithm and tune it for your tastes, assuming the defaults
don't already do what you want.

#### [Tuning guide for CLI](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/TUNING-CLI.md)

#### [Tuning guide for VapourSynth](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/TUNING-VS.md)


## Benchmarks

Numbers below come from `scripts/bench_runs.py` (`just compare-perf`), which pipes
each tool to `ffmpeg -f null -` so the encoder is not measured. Throughput is
total frames divided by wall-clock elapsed.

- Input is a 3,450-frame 1080p FFV1 clip.
- `av-denoise` using the `vulkan` backend.
- Running on a `AMD AI Pro R9700` (AMD 9070XT equivalent) GPU.
- Elapsed time is measured around the whole process, so the one-off scene detection
  pass is inside every number.
- Take these numbers with a pinch of salt.

---

### Algorithm defaults

Every row uses `--channel-mode luma,chroma` and no tuning beyond the preset, so the
rows are directly comparable. NL4D always tracks motion, which is why the
motion-compensated NLMeans-HQ row is here — that is the like-for-like comparison, not
the plain one.

| run                                          | preset |       fps | denoising   | detail retention | notes                                                                     |
|----------------------------------------------|--------|----------:|-------------|------------------|---------------------------------------------------------------------------|
| `nlmeans --variant fast`                     | `base` | **58.91** | low         | low              | Traditional NLMeans algorithm                                             |
| `nl4d --preset fast`                         | `fast` |     48.04 | high        | higher           | Better detail retention compared to V-BM3D (r=1)                          |
| `nlmeans --variant hq`                       | `base` |     47.34 | medium      | medium           | NLMeans with adaptive noise estimation and motion confidence (NLMeans-HQ) |
| `nlmeans --variant hq --motion-compensation` | `base` |     42.48 | medium      | medium           | NLMeans-HQ + block matching motion compensation                           |
| `nl4d`                                       | `base` |     42.39 | **highest** | **highest**      | Better detail retention compared to V-BM3D (r=2) and all NLMeans variants |

The two quality columns are not objective, they exist to give you an idea more of what sort of configuration
fits your situation best.

Grouping patches across the temporal window costs about what motion-compensated
NLMeans-HQ costs at the same window size. One rung down the ladder, `nl4d --preset
fast` halves the window to 3 frames and lands on plain NLMeans-HQ throughput while
still tracking motion.

All five ran back to back in one session. Repeat passes agreed within 2% on every row
except `nlmeans --variant fast`, the least GPU-bound run of the five, which came in 12%
low on one pass out of four under background load.

Reproduce with (add `--device discrete:N` to pin a particular GPU):

```bash
just compare-perf -- --accelerators vulkan \
  --only av_default_nlmeans_fast,av_default_nlmeans_hq,av_default_nlmeans_hq_mc,av_fast_nl4d,av_default_nl4d
```

### NL4D vs V-BM3D

NL4D groups patches across the temporal window the way V-BM3D does, so the closest
external reference is a real V-BM3D. This runs [V-BM3DHIP](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA)
on the GPU through VapourSynth (the `vapoursynth-bm3dhip` package), at NL4D's own
window size so both search five frames.

| run                  |       fps | vs NL4D      |
|----------------------|----------:|--------------|
| NL4D (`base` preset) | **38.65** | —            |
| V-BM3DHIP (radius 2) |     28.69 | 1.35x slower |

> [!NOTE]
> V-BM3DHIP has no automatic noise estimation, so its sigma is pinned. That changes what the 
> result looks like, not how much work it does. 

```bash
just compare-perf -- --accelerators vulkan --only av_default_nl4d,bm3dhip_r2
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

---

## Hardware support

The project supports the following accelerators/gpus:

- **AMD GPUs** (via the `rocm` or `vulkan` features)
- **Intel GPUs** (via the `vulkan` feature)
- **Nvidia GPUs** (via the `cuda` or `vulkan` features)
- **Apple Silicon** (via the `metal` feature)

Run `av-denoise list-devices` to see which of these your machine offers and what to pass to
`--device`.

There is no software backend. The collaborative filter aggregates its filtered patches through
atomic floating-point adds, and CubeCL's CPU runtime does not implement atomics. A software
*device* is still reachable with `--device cpu` where the platform provides one, such as lavapipe
under Vulkan.

### Notes about the JIT

It is important to note that `av-denoise` internally uses a JIT (Just In Time) compiler for its kernels. This means
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

#### Compiled kernel cache

Compiling the kernels takes about ten seconds when you first start the denoising pipeline. 
These compiled kernels get cached on disk, which makes that a cost paid once per machine rather than once per run.

By default, the cache lives in `av-denoise` inside the platform cache directory, which is `$XDG_CACHE_HOME` or
`~/.cache` on Linux and macOS, and `%LOCALAPPDATA%` on Windows. With no platform cache directory at all, it falls
back to `av-denoise` inside the temporary directory and warns.

- `AV_DENOISE_COMPILATION_CACHE=/some/dir` puts the compiled-kernel and autotune caches somewhere else, which is
  what CI runs and containers use to keep the cache on a mounted volume. It overrides whatever is in `cubecl.toml`.
- `AV_DENOISE_COMPILATION_CACHE=off` disables caching entirely. Use this when benchmarking, because a warm cache
  hides the compilation cost a first run pays.

If the cache directory cannot be created, `av-denoise` logs a warning and carries on without a cache.

Library users can call `av_denoise::install_compilation_cache()` before `Denoiser::create` to get the same
behaviour in their own binary. It has to run before the first `Denoiser` exists, because building a CubeCL client
locks the global config. An embedder that wants to choose the cache directory itself can call
`av_denoise::default_cache_dir()` to get the same default this crate uses, and
`av_denoise::install_compilation_cache_at()` to install it, or any other directory, directly.
