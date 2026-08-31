# _Don't do this!_

Before you begin using this library, please take a moment to review this list of things to avoid, some of them go
against what is currently fairly common in pipelines which have some form of denoising and will produce a worse result
in both noise removal and detail retention.

> [!TIP]
> If you want a more in-depth explanation as to why these aren't recommended, click the "Why not?" sections
for each point.
> 

## Don't use pre-filters

This is almost certainly the wrong solution to whatever issue you are trying to solve, there are a few reasons why with
the biggest being it reduces detail retention and adds additional work.

Instead, use the **NLMeans-HQ** or **NL4D** algorithm families which handle the source noise natively within
their algorithms rather than using a prefilter as a crutch.

The only relevant usage of pre-filters is for the base **NLMeans** algorithm for which we supply a pre-built selection
of algorithms as part of the internal processing. But again, you should reach for the better algorithms available to you.

<details>
<summary><b>Why not?</b> - A "simplified" explanation of how it impacts the algorithms.</summary>

Non-local algorithms like NLMeans have a "search" phase where they look for similar patches/blocks in order
to work out what to do with the noise in particular section. They effectively work out how similarly two pixels
look alike. The problem this causes is that on a noisy image, you are comparing the noise as well as its contents.

That noise inflates the distance of every patch _randomly_ so the weight kernel effectively has to be widened (higher strength)
in order to find any matches as all. As a consequence the wider range admits genuinely dissimilar patches/blocks
and pushes the filter towards averaging all patches together equally rather than biasing towards the genuinely matched
patches.

Introducing a pre-filter like a light `nlm` pass or a `bilateral` blur smooths out that noisy signal making the signal
clearer and resulting in better matching patches being selected and averaged together, rather than all patches
equally. (NOTE: "equally" here is doing a lot of heavy lifting, but for simplicity it is fine.)

However, there are no free lunches. The prefilter's pixels never appear in the final output since it is only used for
the patch comparison stage. But, it's smoothing of the noise removes the fine, low-contrast structure that is used
to distinguish between two candidate patches.

Now, on the base NLMeans algorithm, this is a tradeoff you'd be willing to make because the impact the noise has
on your patch search results in stronger detail loss than having the pre-filter. But past a certain point you've traded
"distances polluted by noise" for "distances with nothing left to distinguish", which flattens the weights the same
way having your strength too high would. That's why it is a bandaid solution, it treats the symptom in the metric
rather than the metric being wrong.

The `NLMeans-HQ` fixes this within the algorithm itself by modeling the noise in the image and subtracting it
from the patches when comparing. More specifically, every frame we measure the sigma _σ_ of each frame and each channel
which is fed into the algorithm rather than relying on a fixed, assumed constant sigma.

Using our measured value, we can subtract it from the patch distance itself, before that distance is turned into
a weight. This results in a genuine match now gets a distance much closer to zero regardless of how noisy or grainy the
frame is, while a structurally different patch keeps its difference.

Because the floor is a flat offset applied across all patches, the strength no longer has to be inflated in order to
compensate for a noisy image. This allows us to achieve the same behaviour as a pre-filter would do, without
sacrificing fine detail to get there.

> [!NOTE]
> NM3D and NL4D differ slightly in how pre-filters impact them due to the algorithmic differences compared to NLMeans.
> But the issue presented is equally if not more relevant.
</details>

## Don't do two passes

"Two passes" refers to using another denoising algorithm ahead of the main denoiser to do some initial "cleanup" pass.
_This is a bad idea._ It carries with it the [same downsides as pre-filters](#dont-use-pre-filters) while _also_
having the added downside of it directly impacting the output pixels. 

None of the algorithms implemented in av-denoise are built with a 2-pass system in mind, and any success you get with 
them is more likely your parameters being suboptimal.

**See the ["Don't use pre-filters" Why not](#dont-use-pre-filters) section for why this is a bad thing.**

## Don't set `sigma` yourself (_NLMeans-HQ & NL4D only_)

Both NLMeans-HQ and NL4D are built around their automatic noise estimation kernels, choosing to manually set the sigma
effectively invalidates all defaults across all parameters on the algorithm and makes it far more prone to detail 
loss or producing artefacts. 

_You are effectively doing the equivalent of removing the control surfaces from an aircraft and attempting to fly it._

**Instead**, use the `*-scale` parameters each algorithm exposes, these allow you to adjust the behaviour relative to
its consistent baseline which will result in it still producing a constant level of quality and denoising strength
regardless of the source content.

<details>
<summary><b>Why not?</b> - How `sigma` is not as simple as you think.</summary>

In most filters sigma _σ_ is one dial to adjust among many, alongside strength, patch size, etc...
Both NLMeans-HQ and NL4D put sigma _above_ the other parameters. The measured σ is what the other parameters derive 
from, so getting this value wrong by even a relatively small margin causes a cascade of error at each stage. 

Put simply, what you can potentially gain by hand tuning the sigma value is grossly outweighed by the cost of
getting that value wrong even a little bit.

### One number cannot be right for a whole clip

The estimator measures σ **per frame and per channel**, then smooths it over time. This is by design.
Grain level changes with exposure, with scene, and with how hard the encoder that produced your source was pushed,
along with chroma almost never being as noisy as luma. A single `sigma` for the entire clip is right for the 
shot you sampled and wrong for everything else.

Just trust the estimator, I promise it will do a better job.

### What sigma σ impacts

For context here is a brief summary of all the ways sigma impacts the two algorithms:

- Strength is a multiplier on the observed noise level, the strength value has no meaning unless it is paired with
  an accurate sigma value.
- Patch distance noise floor. This is the amount subtracted from every patch distance before it becomes a weight and
  is calculated at `2 * σ² * taps`, small variance can shift this value dramatically and in turn, change how the 
  algorithms see similar patches.
- Motion confidence block matching threshold is directly driven by the luma plane's sigma value. This directly impacts
  what NL4D's grouping trusts which can cause heavy artefacting and detail loss.

...There are about 23 different places where sigma directly impacts upstream kernels and each small amount of
error compounds into the next kernel.
</details>

