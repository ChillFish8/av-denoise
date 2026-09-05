# Tuning guide for VapourSynth

Even though we provide `presets` as a convenient way to adjust quality VS speed, you can get a lot more performance
for a given quality by making some adjustments if you're willing to put in the time to tune it manually. Assuming
the defaults don't give you want out of the box.

Each algorithm wants you to use a slightly different set of options for tuning, so it is important to make sure
you're following the guide for the algorithm you're using specifically.

> [!CAUTION]
> Tuning guides for other algorithms like BM3D, etc... Will **not** transfer over. 

> [!IMPORTANT]
> Each algorithm has limitations, NLMeans can only ever really do a very light denoise without producing perceptible
> detail loss. Often it is better to just go up to NLMeans-HQ, or from NLMeans-HQ to NL4D rather than spending
> lots of additional time trying to tune an algorithm that has hit is fundamental limitations.
>
> For more info consider reading the "How it works" sections of our
> **[Choosing an algorithm](CHOOSING_AN_ALGORITHM.md)** docs.

## How the plugin differs from the CLI

Two things change what tuning feels like here, both worth knowing before you start.

**`preset` does not pick the algorithm.** On the CLI, `--preset veryfast` selects the plain NLMeans
variant and everything above it selects NLMeans-HQ. In Python you pick the algorithm by calling
`Nlm`, `NlmHQ` or `Nl4d`, so `preset` only resolves the numeric dials, mainly the temporal window.
`NlmHQ(clip, preset="veryfast")` is a spatial-only NLMeans-HQ run, and `Nlm(clip, preset="slow")` is
the fast variant with a 9-frame window, neither of which the CLI can express through `--preset`.

**There is no scene detection.** The plugin estimates the noise level fresh for each frame from that
frame's own temporal window rather than smoothing it over a scene, so a frame denoises to the same
pixels no matter what order VapourSynth asks for frames in. Tuning still works the same way, but if
you have wildly different-looking scenes in one clip, trimming and denoising them separately gives
each one its own dials.

Every numeric parameter is optional. An unset one falls back to what `preset` resolves for it, and
anything you set explicitly wins over the preset.

## Nl4d

```python
import vapoursynth as vs
import vsavd as avd

core = vs.core
clip = core.lsmas.LWLibavSource("noisy.mkv")
clean = avd.Nl4d(clip)
clean.set_output()
```

`preset` sets how many neighbouring frames are searched, from a 3-frame window at `fast` to a
17-frame one at `veryslow`. `veryfast` keeps the 3-frame window and narrows the spatial search
instead, since NL4D has nothing to group without neighbours.

### The NL4D dials

| Symptom                               | First thing to try              | Second            |
|---------------------------------------|---------------------------------|-------------------|
| Grain or noise still visible          | `lambda_ht_scale` up a little   | one preset higher |
| Fine texture getting scrubbed         | `lambda_ht_scale` down a little | `sigma_scale=0.9` |
| Result looks under-cleaned everywhere | `sigma_scale=1.1`               | one preset higher |
| Smearing or ghosting on motion        | one preset lower                | `refine` up       |
| Too slow                              | `spatial_radius` down           | one preset lower  |

**`lambda_ht_scale` is the main dial.** The threshold it scales is how many standard deviations
of estimated noise a transform coefficient has to clear to survive, so raising the scale removes
more noise and takes more fine detail with it. Move in steps of **0.05 to 0.1** and judge by eye,
before you consider going up a preset. You should try this parameter before touching the absolute
values, since luma and chroma start from different defaults and the scale keeps that separation.

```python
clean = avd.Nl4d(clip, lambda_ht_scale=1.1)
```

**`lambda_ht` sets those thresholds outright.** The defaults are 4.24 for luma and 3.36 for chroma.
Luma's value was tuned and deliberately biased toward keeping detail. Chroma's carries that same
bias over rather than being tuned on its own. A single value here flattens both planes onto the
same number, so prefer the scale unless you have a figure you want. `luma_lambda_ht` and
`chroma_lambda_ht`, both reachable by name, pin one plane without touching the other, and
`lambda_ht_scale` still applies on top of whatever is pinned.

```python
clean = avd.Nl4d(clip, luma_lambda_ht=5.0, lambda_ht_scale=1.05)
```

**`sigma_scale` is the other one**, and it does something different. The lambda dials decide how
aggressive to be at a given noise level. `sigma_scale` corrects the noise level itself. That
estimate also feeds the motion confidence scoring, so when the whole result reads uniformly
under- or over-cleaned, correcting the level fixes the cause rather than the symptom. When you
are happy with the level and just want to adjust how much noise is removed vs detail, use
`lambda_ht_scale`.

**`spatial_radius` is the speed dial.** The centre-frame search covers `(2 * radius + 1)^2`
positions, so it dominates the work. Dropping it from 9 to 6 roughly halves the candidates, which
is exactly what `preset="veryfast"` does.

### What not to touch in NL4D

- **`sigma`** pins the noise level to a fixed value and turns the per-frame measurement off
  entirely. `sigma_scale` keeps the measurement and nudges it, which is almost always what you
  actually want.
- **`luma_mismatch_scale` and `chroma_mismatch_scale`** set how much less a poorly matched patch is
  trusted, rather than whether it is, judged by the patch's own match residual rather than the
  motion block's score. The variance they control grows with the square of the value,
  so `2` distrusts a bad match four times as much. The effect saturates. It saturates sooner the
  worse the patch matched, because the variance the mechanism derives is capped at 64 times the
  channel's own variance, and `0` turns the mechanism off.
- **`temporal_radius`** is what `preset` mostly exists to resolve. Setting it by hand is fine, but
  it is the same lever the preset ladder pulls, so reach for the ladder first.

`strength`, `luma_strength`, `chroma_strength`, `search_radius`, `patch_radius`, `prefilter` and
`motion_compensation` all belong to the NLMeans weighting pass, which NL4D does not have. Passing
one raises `vs.Error` rather than being silently ignored, so a misplaced dial fails loudly.

## NLMeans-HQ (NLMeans High Quality)

```python
import vapoursynth as vs
import vsavd as avd

core = vs.core
clip = core.lsmas.LWLibavSource("noisy.mkv")
clean = avd.NlmHQ(clip)
clean.set_output()
```

`preset` sets the temporal window and the search radius, from no window at all at `veryfast`
through 3 frames at `fast` to 17 at `veryslow`. The variant is already fixed to HQ by calling
`NlmHQ`, so unlike the CLI, `preset="veryfast"` here still runs the noise estimator, just with
nothing but the current frame to work from.

| Symptom                        | First thing to try                 | Second                     |
|--------------------------------|------------------------------------|----------------------------|
| Grain or noise still visible   | `sigma_scale=1.1`                  | one preset higher          |
| Fine texture getting scrubbed  | `sigma_scale=0.9`                  | lower `strength` slightly  |
| Smearing or ghosting on motion | `motion_compensation=True`         | one preset lower           |
| Colour speckle survives        | `chroma_strength` up               | -                          |
| Too slow                       | check the GPU is actually selected | one preset lower           |

### Still too noisy

Work through these in order.

1. **Nudge `sigma_scale` up** (try 1.1, then 1.2). This tells the denoiser the noise is a little
   stronger than it measured, and everything downstream (strength, patch matching, motion
   confidence) adapts together. Increase in steps of 0.1, judging by eye each time. It costs
   nothing in throughput, which is why it comes first.
2. **Set `motion_compensation=True`** on footage with fast movement, so the temporal window keeps
   finding usable matches instead of falling back to the current frame.
   * For **Anime** sources, you may not want to enable this option. Anime sources typically cope with
    high motion much more effectively, and introducing motion compensation can work against you.
3. **Go up a preset.** Noise and grain are independent frame to frame, so a deeper temporal
   window removes them more effectively than any strength increase. This is the strongest lever in
   the tool, and the most expensive, so save it for once the scale has stopped helping.

```python
clean = avd.NlmHQ(clip, sigma_scale=1.1)
```

### Losing detail

The same dial works the other way: `sigma_scale=0.9` tells NLMeans-HQ the source is cleaner than
measured.

The rule of thumb for choosing between the two:

- `sigma_scale` says *how noisy the source really is*
- `strength` / `luma_strength` / `chroma_strength` says *how aggressively to clean at that noise level*.
    * Remember that strength for luma and chroma planes are different, and under `NlmHQ` the value
      is a multiplier on the measured noise level rather than an absolute figure. You should use
      this as a last resort.

**Prefer the sigma scale first.** The noise level also steers patch matching and motion confidence, so correcting it
fixes the cause rather than the symptom.

### Common situations

> [!IMPORTANT]
> These points assume that you have already tried with the base defaults.

- **Old live action, heavy film grain.** Real grain is spatially correlated and hides from naive
  estimators. The estimator here measures it from frame-to-frame residuals, so if the result reads
  slightly weak to your eye, `sigma_scale=1.1` is the intended fix. Add
  `motion_compensation=True` next, and only then go up a preset.
- **Colour speckle.** Chroma already gets its own measured strength, but stubborn colour noise can
  take `chroma_strength` above the default without touching luma.
- **Fast motion looks smeary.** Set `motion_compensation=True` first. If a scene still trails,
  drop one preset (a shallower window has less material to mis-blend).
- **Mixed content in one clip.** With no scene detection the estimate is per frame, so a clip that
  swings between very clean and very grainy material is better served by trimming it and giving
  each part its own call.

### What not to touch in NLMeans-HQ

These exist for debugging, calibration work, and unusual sources. Reaching for these usually
makes things worse.

- **`sigma`** pins the noise level to a fixed value, which disables the per-frame measurement
  entirely. `sigma_scale` keeps the measurement and nudges it, which is almost always what
  you actually want.
- **Raising `strength` to fight leftover grain.** Grain that survives means the noise level read
  low, and extra strength scrubs detail before it removes grain. Fix the level (`sigma_scale`)
  instead.
- **`prefilter`** (the NLMeans pilot and bilateral modes) changes what patch matching sees, and
  under NLMeans-HQ's calibrated automatic handling both modes measured neutral at best on
  default settings. They exist for experimentation rather than as a default quality upgrade.
- **`search_radius` and `patch_radius`** reshape the whole matching problem, and every other
  default is tuned around them. Cost grows quadratically with the search radius, and the presets
  already adjust these parameters based on exhaustive tuning.
- **`device="cpu"`** selects a software device where the platform offers one, such as lavapipe
  under Vulkan. It is for testing the pipeline, not for real encodes.

Setting an NL4D-only parameter such as `lambda_ht_scale`, `spatial_radius` or `refine` raises
`vs.Error` here rather than being ignored.

## NLMeans

```python
import vapoursynth as vs
import vsavd as avd

core = vs.core
clip = core.lsmas.LWLibavSource("noisy.mkv")
clean = avd.Nlm(clip, strength=1.2)
clean.set_output()
```

`Nlm` has no noise estimator, so there is no measured level to correct and nothing for a scale
parameter to nudge. `sigma` and `sigma_scale` raise `vs.Error` here. That leaves `strength` as the
dial that carries the noise level *and* the aggression, which is why this variant needs the most
hand-tuning of the three and gives the least back.

`strength` matches FFmpeg's nlmeans scaling, so a value you already trust from that filter
transfers. The default is `1.2` at every preset, since nothing is measured to adapt it.

| Symptom                        | First thing to try                 | Second                   |
|--------------------------------|------------------------------------|--------------------------|
| Grain or noise still visible   | `strength` up in steps of 0.1      | one preset higher        |
| Fine texture getting scrubbed  | `strength` down in steps of 0.1    | one preset lower         |
| Smearing or ghosting on motion | `motion_compensation=True`         | one preset lower         |
| Colour speckle survives        | `chroma_strength` up               | -                        |
| Detail loss whatever you try   | switch to `NlmHQ`                  | switch to `Nl4d`         |

**`strength` is the only quality dial.** Move it in steps of about 0.1 and judge by eye. Luma and
chroma want different values on most sources, so once you have a level you like, `luma_strength`
and `chroma_strength` pin one plane each and override the shared `strength` for that plane.

```python
clean = avd.Nlm(clip, luma_strength=1.2, chroma_strength=1.5)
```

**`preset` still sets the temporal window**, from spatial-only at `veryfast` to a 17-frame window
at `veryslow`. A deeper window removes noise more effectively than any strength increase, but
without an estimator behind it the fast variant blurs sooner, so a preset bump on `Nlm` is more
likely to cost detail than the same bump on `NlmHQ`.

### What not to touch in NLMeans

- **Pushing `strength` to fight heavy grain.** This is the limit of the algorithm rather than of
  the dial. Plain NLMeans can only really do a light denoise before detail loss becomes visible.
  When you get there, move to `NlmHQ`, and from `NlmHQ` to `Nl4d`, rather than spending more time
  on a filter that has hit its ceiling.
- **`search_radius` and `patch_radius`** reshape the whole matching problem, and the presets
  already resolve them from exhaustive tuning. Cost grows quadratically with the search radius.
- **`prefilter`** changes what patch matching sees. It exists for experimentation, not as a default
  quality upgrade.
- **`device="cpu"`** is for testing the pipeline, not for real encodes.
