# Choosing an algorithm

av-denoise aims to cover effectively every use case you might have, but working out which algorithm is right for you
can be a bit daunting and difficult to compare.

In short, **NL4D should be your default** since has the greatest denoising and detail retention compared to the others.

This guide explains what each algorithm does at a high level and what its pros and cons are in the order we think
you should try them from the highest priority, to the lowest priority.

> [!IMPORTANT]
> You may _think_ you want a prefilter or no temporal window, but that is probably incorrect or operating on an
> assumption that does not apply to av-denoise. It is **highly** advised to ignore any advice recommending a prefilter
> for the NL4D or NLMeans-HQ algorithms in av-denoise.

## NL4D

A cousin of both NLMeans-HQ and V-BM3D, NL4D inherits the noise measurement and motion tracking from the first
and the collaborative filtering from the second.

<details>
<summary><b>How it works</b></summary>

Like every algorithm here, it starts by searching for patches of pixels that look like the patch it is currently
working on, pulling candidates from both elsewhere in the current frame and from future and past frames.
This is done while following the motion so that a moving object is compared against itself rather than against
whatever happens to be at the same coordinates a few frames later selecting the eight "best" candidates
forming a _group_.

This group is where it diverges from NLMeans. NLMeans averages the group together, 
weighted by how similar each member of the group is. Crude, but fast, and effective enough that it is still used often.
However, it comes at a cost of removing both the noise and low-contrast detail alike because it has no way of
knowing what is truly noise, or just fine detail. 

NL4D instead transforms the whole group into frequency coefficients, sorted so that content every member agrees 
on goes into one bucket and content that only one or two members carry lands in another. Noise is what 
the members do not agree on. Real detail, even faint and very fine detail, is what they do agree on.
Each coefficient is then kept or discarded depending on whether it stands far enough above the noise level _measured in 
that frame_, and what survives is transformed back into pixels.

One detail matters here though, not every member is trusted equally. A patch pulled from a neighbouring frame that 
only roughly matched is treated as a noisier observation of the same content, which raises the bar its disagreement 
with the other patches has to clear. A bad match therefore cannot pass itself off as detail the group found. This 
is one of the key differences from how the BM3D and BM4D families handles the same problem.

The practical effect is two-fold. Against NLMeans, NL4D can remove noise that averaging can only smooth over, because
it makes a decision about each piece of the signal separately rather than one blended compromise about
the pixel as a whole. Against BM3D (and V-BM3D), which makes those same per-coefficient decisions, the gain is 
they are better decisions.

</details>

#### Pros

- The strongest denoising and best detail retention of anything here (and against BM3D)
- Handles motion, occlusion, and scene changes without smearing.
- Automatic per-frame noise measurement, so it needs no tuning to a source.
- Faster than the V-BM3D & BM4D algorithms.

#### Cons

- The slowest of the algorithms provided, although it is worth noting it ties when Nlmeans-HQ enables motion compensation.
- Requires a temporal window and as such cannot be used on single images.
- Slightly GPU memory than the others.


## NLMeans-HQ (NLMeans High Quality)

The same front end as [NL4D](#nl4d) which includes the noise measurement, motion tracking and confidence weighting
stopping short of the collaborative filtering stage.

<details>
<summary><b>How it works</b></summary>

Like with NL4D, it searches for candidate patches that look like the patch it is currently working on, both in the 
current frame and elsewhere within the temporal window following the motion so that a moving object is compared against
itself.

Where NL4D collects the best candidates into a group and filters them collaboratively, NLMeans-HQ simply averages
them, weighted by how similar each one is. A patch that matched well counts for a lot, one that barely matched
counts for almost nothing. Logically the same as regular non-local means.

The interesting part is what "how similar" means once the noise is in the picture.
Two photographs of _identical_ content, both noisy, do not look identical to the distance kernel,
they differ by the noise. So every comparison comes back looking like a worse match than it really is, and the
classic answer is to loosen your definition of "similar" until matches start turning up again. Which works, and is
what the regular NLMeans does, but a loose definition admits patches that are not actually alike, and the 
average drifts toward "average everything nearby", which is just a blur.

NLMeans-HQ removes the need for that. It measures how much noise is in each frame and for each channel, works 
out how much of every comparison that noise is responsible for, and subtracts it before deciding how much a candidate
counts. A genuine match now scores as a genuine match no matter how grainy or noisy the source is, so the definition of
similar remains tight. The measurement also sets the multiplier for the filter strength for you, which is why the `strength`
you pass to the filter is a multiplier on the measured noise rather than a fixed amount of filtering, and why
the same value behaves consistently across sources of very different quality.

In practice, the effect is a weighted average made of the _right_ candidates. It is still averaging the candidates together
and averaging is still a compromise, which is why there is still a practical limit for how aggressively it can 
remove noise before perceptible detail loss begins to appear. Which is what NL4D solves. But it is cheaper to compute
on your GPU than NL4D is by default.

</details>

#### Pros

- Faster than NL4D whenever motion compensation is off, since it skips the grouping and transform stage entirely.
- Works on a single image, with no temporal window at all.
- Automatic per-frame noise measurement, so it needs no tuning to a source.
- Cannot produce blocking or ringing, because there is no block grid and no transform to produce them. 
  Its failure mode is softness, which is far easier to live with.
- Lowest memory use when the temporal window is small or absent.

#### Cons

- Significantly less detail retention than NL4D at matched strength. Averaging cannot separate faint detail from noise, 
  it can only choose what to average which is a hard limit on how effective this algorithm can be.
- Once motion compensation is enabled the speed advantage largely disappears, at which point NL4D is better in 
  every way.
- Still slower than regular NLMeans which does no noise measurement, confidence, etc...

## NLMeans

The classic, unchanged from the original paper, just implemented to run efficiently on the GPU.
It performs no measurement, collaborative filtering, etc... It is the simplest algorithm currently
implemented in av-denoise.

> [!CAUTION]
> This simplicity comes at a cost of quality and is often not a trade-off people are willing to make. Especially
> when compared to the improvement you get using NLMeans-HQ for a relatively low cost.

<details>
<summary><b>How it works</b></summary>

The searching process is the same as NLMeans-HQ and NL4D. For the patch it is working on, it looks for patches
that are similar to it, both within the current frame and within the temporal window. Each candidate is scored
on how closely it matches and the result is a weighted average, with close matches counting for a lot, and weak
matches counting for almost nothing. That is the algorithm, and that is why it is fast the fastest to execute.

What it does not do is ask how noisy the frame is. You must tell it _exactly_ how hard to filter with a single, fixed
`strength` value which is applied across every frame regardless of what is in front of it. This is what NLMeans-HQ
was originally designed and built to improve upon.

Since every patch carries noise, every comparison comes back looking worse than it really is, because it performs
no measurement, there is no way to account for this outside of increasing the strength which will produce more matches 
at the cost of also allowing in matches which are not really matches. The more these incorrect matches are admitted
the more the averaging slides towards averaging everything nearby which is the same as applying a blur. 
Turn the strength down and the frame remains noisy. Somewhere between these two extremes is likely where you want to
be, but this requires you to find this value manually and adjust for each source you want to process.

A [pre-filter](#pre-filters) exists to take of the sting out of this, smoothing the image that is used to find
similar candidates while still averaging the original pixels. A light bilateral blur is likely to improve the 
effectiveness, but is also a bandaid solution compared to NL4D and NLMeans-HQ. This is why we don't recommend
using pre-filters and instead reaching for the other algorithms.

</details>

#### Pros

- The fastest algorithm here by a wide margin.
- `strength` means the same thing as FFmpeg's `nlmeans` filter, so existing settings and recipes carry over directly.
- Works on a single image, with no temporal window at all.
- Lowest memory use of anything here.
- Predictable. The same input and the same settings do the same thing to every frame, which matters if you are matching
  the look of an existing pipeline. (If that matters more than improving quality.)

#### Cons

- `strength` has to be tuned to your source by hand, and one value has to serve a whole clip whose grain is almost 
  certainly not constant.
- Detail retention is the weakest of the three, and the way you lose it is by turning the strength up which often
  you cannot avoid.
- No noise measurement, so nothing adapts. A source that gets grainier or noisier halfway through will not be handled 
  any differently.
- Best paired with a prefilter to be considered, which reduces the initial speed advantage.

#### Pre-filters

A pre-filter gives the search a second, cleaner copy of the frame to *compare* against. It never contributes a 
single pixel to the output. The image being averaged is always your original input, only the weights change. 
That is what makes it worth doing at all, you get better matching without paying for the detail the blur destroyed.

> [!NOTE]
> A pre-filter only gets you so far, at a point the detail it destroys will weaken your matching enough itself to
> itself contribute to detail loss. It is not a magic bullet and is recommended to use NLMeans-HQ or NL4D instead
> of a pre-filter.

It costs one extra GPU pass per frame. Two are built in.

##### Bilateral - `--prefilter bilateral:<sigma_s>,<sigma_r>`

An edge-aware blur, run on the GPU at push time. It smooths flat areas while leaving strong edges intact, 
so the search gets a calmer image without losing the boundaries it uses to tell one patch from another.

- `sigma_s` is the spatial radius in pixels, above `0` and at most `11.0`.
- `sigma_r` is the colour-similarity threshold in normalised units, above `0`. Values up to `1` are the useful range, 
  and small values are the point. A large one stops the filter being edge-aware at all.
- `bilateral:3.0,0.02` is a good starting point and the one to reach for first.

This is the cheaper of the two and the one we recommend for the fast variant.

##### NLM - `--prefilter nlm` / `--prefilter nlm:<strength_scale>`**

A light NLM pass over the frame on its own, with no temporal window, whose output becomes the reference. 
In other words, NLMeans run twice, gently the first time to work out what matches what, and properly the second time.

- `strength_scale` multiplies the main pass strength to get the pilot pass strength. Bare `nlm` uses the calibrated 
  default of `0.4`.
- Stronger and more expensive than the bilateral option. Worth trying when bilateral is not cleaning up enough 
  for the search to find matches.
