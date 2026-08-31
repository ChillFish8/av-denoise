# av-denoise VapourSynth plugin (`vsavd`)

This is the home of [av-denoise](https://github.com/ChillFish8/av-denoise) exposed as a VapourSynth plugin to fit within existing filtering 
pipelines. With caveats.

We provide typed interfaces for all the [supported algorithms](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/CHOOSING_AN_ALGORITHM.md): 

- `vsavd.Nl4d(...)` for our best in class _NL4D_ denoiser offering higher quality and more effective denoising over V-BM3D.
- `vsavd.NlmHQ(...)` for our _NLMeans-HQ_ algorithm which provides a higher quality and smarter denoising experience over base NLMeans.
- `vsavd.Nlm(...)` for quick and dirt NLMeans mirroring FFmpeg's strength scaling.

_All_ algorithms are temporal aware and have inbuilt motion compensation kernels.

## Table of contents

- [Installing](#installing)
- [Choosing an algorithm](#choosing-an-algorithm)
- [Nl4d](#nl4d)
- [NlmHQ](#nlmhq-nlmeans-hq)
- [Nlm](#nlm-nlmeans)
- [Presets](#presets)
- [Channel modes](#channel-modes)
- [Devices and accelerators](#devices-and-accelerators)
- [Shared conventions](#shared-conventions)
- [Tuning guide](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/TUNING-VS.md)

## Installing

Pre-built wheels are available for Linux, macOS and Windows, published to PyPI.

_Please note that the wheels for **Linux** and **Windows** are compiled for `vulkan` and `cuda` only. **macOS** is compiled for `metal`.
ROCm is not recommended and requires manual compilation._

```bash
pip install vsavd
```

or, with `uv`:

```bash
uv add vsavd
```

Installing `vsavd` bundles the compiled plugin inside the wheel, so there is nothing else to build or load.

## [Choosing an algorithm](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/CHOOSING_AN_ALGORITHM.md)

Please have a read of the linked page for info about what each algorithm does and its tradeoffs.

**TL;DR: Use NL4D if you're unsure**

## [_Don't do this!_](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/DONT_DO_THIS.md)

We recommend reading this before deciding how to integrate this into your existing filtering pipeline
as some things differ quiet heavily to what you are likely used to.

## [Nl4d](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/CHOOSING_AN_ALGORITHM.md#nl4d)

`Nl4d` is the spatio-temporal denoiser, wrapping `avd.NL4D`. It gives the best noise
removal and detail retention of the three, at the cost of time.

```python
import vapoursynth as vs
import vsavd as avd

core = vs.core
clip = core.lsmas.LWLibavSource("noisy.mkv")
clean = avd.Nl4d(clip)
clean.set_output()
```

| Parameter         | Type            | CLI equivalent          | Recommended            |
|-------------------|-----------------|-------------------------|------------------------|
| `lambda_ht_scale` | float           | `--lambda-ht-scale`     | yes, the main dial     |
| `sigma_scale`     | float           | `--sigma-scale`         | yes                    |
| `preset`          | string          | `--preset`              | yes                    |
| `refine`          | int             | `--refine`              | yes                    |
| `spatial_radius`  | int             | `--spatial-radius`      | yes, the speed dial    |
| `lambda_ht`       | float           | `--lambda-ht`           | situational, see below |
| `channel_mode`    | string          | channel-mode flags      | situational            |
| `device`          | string          | `--device`              | situational            |
| `accelerators`    | list of strings | `-A`, `--accelerators`  | situational            |

`lambda_ht_scale` is the threshold multiplier a transform coefficient's estimated-noise
standard deviations must clear to survive. Raising it removes more noise and takes more
fine detail with it. Try it in steps of about 0.05 before reaching for `lambda_ht`,
which pins luma and chroma's thresholds (5.3 and 4.2 by default) to the same absolute
number and loses that separation.

`spatial_radius` is the speed dial. `preset` already resolves it, so setting
`spatial_radius` explicitly overrides whatever the preset would have picked. The
centre-frame search covers `(2 * radius + 1)^2` positions, so it dominates the work.
Dropping it is the fastest way to speed a run up.

`refine` is the half-width of the window searched around each neighbour frame's
motion-predicted position. Raise it when motion tracking lands close but not exact.

`sigma_scale` keeps the per-scene noise measurement and nudges it, which is almost
always what you actually want.

## [NlmHQ (NLMeans-HQ)](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/CHOOSING_AN_ALGORITHM.md#nlmeans-hq-nlmeans-high-quality)

`NlmHQ` is the high-quality NLMeans variant, wrapping `avd.NLMeans` with `variant="hq"`.
It runs a per-scene noise estimator and reads its result into `sigma_scale` rather than
leaving noise level to a hand-set `strength`. Use it over `Nlm` when you want the
estimator to pick the noise level for you.

```python
import vapoursynth as vs
import vsavd as avd

core = vs.core
clip = core.lsmas.LWLibavSource("noisy.mkv")
clean = avd.NlmHQ(clip)
clean.set_output()
```

| Parameter             | Type            | CLI equivalent          | Recommended            |
|-----------------------|-----------------|-------------------------|------------------------|
| `preset`              | string          | `--preset`              | yes                    |
| `motion_compensation` | bool            | `--motion-compensation` | yes                    |
| `sigma_scale`         | float           | `--hq-sigma-scale`      | yes, the main dial     |
| `chroma_strength`     | float           | `--chroma-strength`     | yes                    |
| `channel_mode`        | string          | channel-mode flags      | situational            |
| `strength`            | float           | `--strength`            | situational, see below |
| `luma_strength`       | float           | `--luma-strength`       | situational            |
| `device`              | string          | `--device`              | situational            |
| `accelerators`        | list of strings | `-A`, `--accelerators`  | situational            |

Raising `strength` to fight leftover grain is the wrong move. Grain that survives means
the noise level read low, and extra strength scrubs detail before it removes grain.
Correcting `sigma_scale` instead is the right move.

## [Nlm (NLMeans)](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/CHOOSING_AN_ALGORITHM.md#nlmeans)

`Nlm` is the fast NLMeans variant, wrapping `avd.NLMeans` with `variant="fast"`. It has no
noise estimator, so it runs quickly and expects you to set `strength` yourself. Reach for
it when speed matters more than squeezing out the last bit of noise.

```python
import vapoursynth as vs
import vsavd as avd

core = vs.core
clip = core.lsmas.LWLibavSource("noisy.mkv")
clean = avd.Nlm(clip, strength=1.2)
clean.set_output()
```

| Parameter             | Type            | CLI equivalent          | Recommended |
|-----------------------|-----------------|-------------------------|-------------|
| `preset`              | string          | `--preset`              | yes         |
| `motion_compensation` | bool            | `--motion-compensation` | yes         |
| `chroma_strength`     | float           | `--chroma-strength`     | yes         |
| `channel_mode`        | string          | channel-mode flags      | situational |
| `strength`            | float           | `--strength`            | yes         |
| `luma_strength`       | float           | `--luma-strength`       | situational |
| `device`              | string          | `--device`              | situational |
| `accelerators`        | list of strings | `-A`, `--accelerators`  | situational |

`Nlm` has no noise estimator, so `strength` is the dial that sets the noise level. Use
`NlmHQ` if you would rather have it measured for you.

## Presets

`preset` is the main quality-versus-speed dial. It takes one of five values, from
fastest to slowest and best-quality: `veryfast`, `fast`, `base`, `slow`, `veryslow`.
Each preset resolves every unset numeric and string parameter to a value tuned for that speed tier.

Any parameter you set explicitly overrides what the preset would have chosen for it,
other fields still fall back to the preset's values. This lets you take a preset as a
starting point and adjust just the dial you care about, as the examples above do.

> [!TIP]
> Presets exist to be easy levers to adjust, but you can probably still get the quality you want
> using the `base` preset on NLMeans-HQ and NL4D by tweaking the `*-scale` parameters.

## Channel modes

`channel_mode` selects which planes get denoised, and takes one of four values:

- `luma` denoises the luma plane only, leaving chroma untouched.
- `chroma` denoises the chroma planes only, leaving luma untouched.
- `lumachroma` denoises both, luma and chroma independently.
- `yuv` denoises luma and chroma together as a single pass. It needs a 4:4:4 source,
  since it requires the chroma planes to be full resolution.

## Devices and accelerators

`device` is typed on all three functions and selects which GPU device runs the filter.
`device="cpu"` selects a software device where the platform offers one, such as
lavapipe under Vulkan. It is for testing the pipeline, not for real encodes.

`accelerators` selects which GPU backend to use, for example `["vulkan"]` or `["cuda"]`. 
Multiple accelerators can be provided and the system will try each accelerator in the order 
provided, choosing the first accelerator which can work on the host hardware.

> [!IMPORTANT]
> You can only use accelerators the plugin was compiled with, for example the wheels for
> Linux and Windows only support `"vulkan"` and `"cuda"`, requesting `"rocm"` would result
> in an error.

```python
clean = avd.Nl4d(clip, preset="slow", accelerators=["cuda", "vulkan"])
```

## Shared conventions

- Every numeric script argument is optional, an unset one falls back to the algorithm's  own default, or to
  whatever `preset` resolves for that field.
- `device` and `accelerators` are unset by default rather than pinned to a literal string.

## [Tuning guide](https://github.com/ChillFish8/av-denoise/blob/HEAD/docs/TUNING-VS.md)

For more information about how to adjust the algorithms to best fit your needs.