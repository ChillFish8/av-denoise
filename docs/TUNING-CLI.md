# Tuning guide for CLI

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

## Nl4d

```bash
av-denoise nl4d --input noisy.mkv | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

`--preset` sets how many neighbouring frames are searched, from a 3-frame window at `fast` to a
17-frame one at `veryslow`. `veryfast` keeps the 3-frame window and narrows the spatial search
instead, since NL4D has nothing to group without neighbours.

### The NL4D dials

| Symptom                               | First thing to try                | Second              |
|---------------------------------------|-----------------------------------|---------------------|
| Grain or noise still visible          | `--lambda-ht-scale` up a little   | one preset higher   |
| Fine texture getting scrubbed         | `--lambda-ht-scale` down a little | `--sigma-scale 0.9` |
| Result looks under-cleaned everywhere | `--sigma-scale 1.1`               | one preset higher   |
| Smearing or ghosting on motion        | one preset lower                  | `--refine` up       |
| Too slow                              | `--spatial-radius` down           | one preset lower    |

**`--lambda-ht-scale` is the main dial.** The threshold it scales is how many standard deviations
of estimated noise a transform coefficient has to clear to survive, so raising the scale removes
more noise and takes more fine detail with it. Move in steps of **0.05 to 0.1** and judge by eye,
before you consider going up a preset.
You should try this parameter before touching the absolute values, since luma and chroma start
from different defaults and the scale keeps that separation.

**`--lambda-ht` sets those thresholds outright.** The defaults are 4.24 for luma and 3.36 for
chroma. Luma's value was tuned and deliberately biased toward keeping detail. Chroma's carries
that same bias over rather than being tuned on its own. A single value here flattens both planes
onto the same number, so prefer the scale unless you have a figure you want.
`--luma-lambda-ht` and `--chroma-lambda-ht` pin one plane without touching the other, and `--lambda-ht-scale` still
applies on top of whatever is pinned.

**`--sigma-scale` is the other one**, and it does something different. The lambda dials decide how
aggressive to be at a given noise level. `--sigma-scale` corrects the noise level itself. That
estimate also feeds the motion confidence scoring, so when the whole result reads uniformly
under- or over-cleaned, correcting the level fixes the cause rather than the symptom. When you
are happy with the level and just want to adjust how much noise is removed vs detail, use `--lambda-ht-scale`.

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
- **`--mismatch-scale`** sets how much less a poorly matched patch is trusted, rather than whether it
  is, judged by the patch's own match residual rather than the motion block's score.
  The variance it controls grows with the square of the value, so `2` distrusts a bad match four times
  as much. The effect saturates. It saturates sooner the worse the patch matched, because the
  variance the mechanism derives is capped at 64 times the channel's own variance, and `0` is the
  same thing as `--no-confidence-variance`.
- **`--thsad-scale`, `--mc-blksize`, `--mc-overlap`, `--mc-search`, `--mc-pyramid-levels`** tune
  the motion machinery's internals, changing any of these will likely invalidate all other defaults.
- **`--field-lambda`** pulls each block's motion vector toward its neighbours. `0` is off. Raise it
  on noisy or flat content where vectors wander.

## NLMeans

```bash
av-denoise nlmeans --input noisy.mkv | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

`--preset` picks the variant as well as the window: `veryfast` runs the `fast` variant with no
temporal window at all, and everything above it runs NLMeans-HQ with a window from 3 frames at `fast`
to 17 at `veryslow`.

| Symptom                        | First thing to try                 | Second                      |
|--------------------------------|------------------------------------|-----------------------------|
| Grain or noise still visible   | `--hq-sigma-scale 1.1`             | one preset higher           |
| Fine texture getting scrubbed  | `--hq-sigma-scale 0.9`             | lower `--strength` slightly |
| Smearing or ghosting on motion | `--motion-compensation`            | one preset lower            |
| Colour speckle survives        | `--chroma-strength` up             | -                           |
| Too slow                       | check the GPU is actually selected | one preset lower            |


### Still too noisy

Work through these in order.

1. **Nudge `--hq-sigma-scale` up** (try 1.1, then 1.2). This tells the denoiser the noise is a
   little stronger than it measured, and everything downstream (strength, patch matching, motion
   confidence) adapts together. Increase in steps of 0.1, judging by eye each time. It costs
   nothing in throughput, which is why it comes first.
2. **Enable `--motion-compensation`** on footage with fast movement, so the temporal window keeps
   finding usable matches instead of falling back to the current frame.
   * For **Anime** sources, you may not want to enable this option. Anime sources typically cope with
    high motion much more effectively, and introducing motion compensation can work against you.
3. **Go up a preset.** Noise and grain are independent frame to frame, so a deeper temporal
   window removes them more effectively than any strength increase. This is the strongest lever in
   the tool, and the most expensive, so save it for once the scale has stopped helping.

### Losing detail

The same dial works the other way: `--hq-sigma-scale 0.9` tells NLMeans-HQ the source is cleaner than measured.

The rule of thumb for choosing between the two:

- `--hq-sigma-scale` says *how noisy the source really is*
- `--strength` / `--luma-strength` / `--chroma-strength` says *how aggressively to clean at that noise level*.
    * Remember that strength for luma and chroma planes are different. You should use this as a last resort.

**Prefer the sigma scale first.** The noise level also steers patch matching and motion confidence, so correcting it
fixes the cause rather than the symptom.

### Common situations

> [!IMPORTANT]
> These points assume that you have already tried with the base defaults.

- **Old live action, heavy film grain.** Real grain is spatially correlated and hides from naive
  estimators. The estimator here measures it from frame-to-frame residuals, so if the result reads
  slightly weak to your eye, `--hq-sigma-scale 1.1` is the intended fix. Add
  `--motion-compensation` next, and only then go up a preset.
- **Colour speckle.** Chroma already gets its own measured strength, but stubborn colour noise can
  take `--chroma-strength` above the default without touching luma.
- **Fast motion looks smeary.** Enable `--motion-compensation` first. If a scene still trails,
  drop one preset (a shallower window has less material to mis-blend).

### What not to touch in NLMeans

These exist for debugging, calibration work, and unusual sources. Reaching for these usually
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
- **`--prefilter`** (the NLMeans pilot and bilateral modes) changes what patch matching sees, and
  under NLMeans-HQ's calibrated automatic handling both modes measured neutral at best on
  default settings. They exist for experimentation and for library users supplying their own
  reference clip, not as a default quality upgrade.
- **`--search-radius` and `--patch-radius`** reshape the whole matching problem, and every other
  default is tuned around them. Cost grows quadratically with the search radius, and the presets
  already adjust these parameters based on exhaustive tuning.
- **`--device cpu`** selects a software device where the platform offers one, such as lavapipe
  under Vulkan. It is for testing the pipeline, not for real encodes.
