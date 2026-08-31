# CLI tutorial

Welcome to the tutorial for using av-denoise in CLI form! This is the primary way we recommend using av-denoise
as it has full control over the input stream allowing us to make use of more optimisations and algorithms to
bring you a better denoising experience.
 
## Installation

By default, when compiling only the `vulkan` feature is enabled, since that is typically the default accelerators
you will want to use.

> [!IMPORTANT]
> If you are building on macOS, you will want to add `--no-default-features` and add `--features metal` to
> enable acceleration for Apple Silicon.

When compiling the binary, enable the `binary` feature. It pulls in both ingestion paths (FFMS2 for
file input, y4m for piped input), so there's nothing else to pick between.

The following (non-accelerator) features are available:

- `binary` - Enables the dependencies and code required to compile `av-denoise` as a binary.
    * This pulls in `ffms2` as hard dependencies. This means you must install `ffms2` before you can compile and link
      the binary.

### Cargo install

This builds the binary with the default accelerators enabled (`vulkan`)

```bash
cargo install --locked av-denoise --features binary
```

### From source

This builds the binary with the default accelerators enabled (`vulkan`)

```bash
git clone https://github.com/ChillFish8/av-denoise.git
cargo build --release --features binary
cp ./target/release/av-denoise ./av-denoise
```

### Container images

Images are published to GHCR for each accelerator backend, one image per backend:

```
ghcr.io/chillfish8/av-denoise:vulkan-<version>
ghcr.io/chillfish8/av-denoise:cuda-<version>
ghcr.io/chillfish8/av-denoise:rocm-<version>
```

Stable releases also get a `<backend>-<major>.<minor>` and a `<backend>-latest` tag. Pre-releases only get
the exact `<backend>-<version>` tag, so an alpha has to be pinned by its full version.

The binary is the entrypoint, so flags are passed straight to the container. The GPU comes from the host, which
means the devices have to be handed in and the input has to be mounted. Output is y4m on stdout, same as the
bare binary.

```bash
docker run --rm \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    -v "$PWD:/in:ro" \
    ghcr.io/chillfish8/av-denoise:vulkan-0.4.0-alpha3 \
    nl4d --input /in/noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

The `cuda` image needs the NVIDIA Container Toolkit and `--gpus all` instead of the `/dev/kfd` and `/dev/dri`
devices. The `rocm` image takes the same devices as `vulkan` shown above.

> [!TIP]
> The JIT recompiles its kernels on every fresh container. Mount a volume and point
> `AV_DENOISE_COMPILATION_CACHE` at it to keep that cost to the first run.


## Binary usage

Two algorithms share one set of global flags. NL4D is the one to reach
for first, and gives up a spatial-only mode to do it. NLMeans is there
for when throughput matters more than the result, or when you need
something NL4D does not carry.

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

### Listing devices

`av-denoise list-devices` prints what each backend can see on this machine.
Every row is a device in the format `--device` takes, next to the backends
that offer it, so it can be copied straight onto a denoising run.

```bash
av-denoise list-devices
```

```text
DEVICE        BACKENDS
default       rocm, vulkan
discrete:0    rocm, vulkan
discrete:1    rocm, vulkan
discrete:2    rocm
integrated:0  vulkan
```

`-A, --accelerators` narrows which backends are asked. Any backend the build
enables but the machine cannot start is named under the table.

> [!NOTE]
> Ordinals are counted per backend. `discrete:1` under ROCm and `discrete:1`
> under Vulkan are each that backend's second discrete GPU, which is not
> always the same card.

### `nl4d` options

Groups matching 8x8 patches from across the whole temporal window into
one stack and shrinks the stack's transform coefficients together, rather
than filtering with non-local means first. Patches are searched both
inside the centre frame and around where each neighbour frame's motion
predicts they moved to.

Motion tracking is always on, and every preset keeps a temporal window,
which this algorithm needs. There is no NLMeans weighting pass, so none of
the NLMeans knobs below exist here.

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
| `--temporal-radius <N>` | How many neighbouring frames to search on each side, between 1 and 8. More frames means more patches to group.                                                                                                       | from `--preset`      |
| `--lambda-ht <f>`       | How aggressively small transform coefficients are zeroed out. Higher removes more noise and more fine detail with it. `--luma-lambda-ht` and `--chroma-lambda-ht` override one plane.                                | 5.3 luma, 4.2 chroma |
| `--lambda-ht-scale <f>` | Multiplies the `--lambda-ht` in effect for each plane. The main quality dial, since luma and chroma start from different defaults and this moves both together.                                                      | `1.0`                |
| `--spatial-radius <N>`  | Half-width of the candidate search inside the centre frame, between 1 and 16. Most of the search work goes here, since the window covers `(2N+1)^2` positions.                                                       | from `--preset`      |
| `--sigma-scale <f>`     | Nudges the measured noise level, the same dial NLMeans spells `--hq-sigma-scale`.                                                                                                                                    | `1.0`                |

<details>
<summary><b>Expert flags</b> — calibration and debugging, not everyday tuning</summary>

| Flag                                                                 | What it does                                                                                                                                                                                                                                                      | Default             |
|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| `--refine <N>`                                                       | Half-width of the window searched around each neighbour frame's motion-predicted position, between 1 and 4. Raise it when motion tracking lands close but not exact.                                                                                              | `2`                 |
| `--sigma <f>`                                                        | Pins the noise level in 8-bit units, turning the per-scene measurement off entirely.                                                                                                                                                                              | measured            |
| `--thsad-scale <f>`                                                  | How badly a neighbour frame may match before its patches stop being trusted.                                                                                                                                                                                      | `1.0`               |
| `--c-min <f>`                                                        | Confidence floor below which a whole neighbour block is skipped rather than scored. Only changes how much compute a frame costs, never which patches are admitted once scored.                                                                                    | `0.05`              |
| `--no-confidence-variance`                                           | Gives every patch the same noise estimate, instead of trusting a poorly matched one less.                                                                                                                                                                         | off                 |
| `--mismatch-scale <f>`                                               | How much less a poorly matched patch is trusted. The variance grows with the square of it, and the effect saturates between roughly 3 and 13 depending on source noise. `0` matches `--no-confidence-variance`. Per-plane as `--luma-`/`--chroma-mismatch-scale`. | `1.0`               |
| `--mc-blksize`, `--mc-overlap`, `--mc-search`, `--mc-pyramid-levels` | Motion-search geometry. NL4D always tracks motion, so these are always live.                                                                                                                                                                                      | `16`, `8`, `4`, `2` |

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
| `--self-weight <f>`                                                  | How much weight the centre pixel gets in the average. `0` is pure NLMeans.                                                                                                         | `1.0`               |
| `--hq-sigma <f>`                                                     | Pins the noise level in 8-bit units, turning the per-scene measurement off entirely. `--hq-sigma-scale` keeps the measurement and nudges it, which is almost always what you want. | measured            |
| `--hq-thsad-scale <f>`                                               | How badly a neighbour may match before its contribution starts dropping.                                                                                                           | `1.0`               |
| `--hq-no-auto-strength`                                              | Reads `--strength` as an absolute value instead of a multiplier on the measured noise.                                                                                             | off                 |
| `--hq-no-noise-floor`                                                | Keeps the expected noise floor inside patch distances instead of subtracting it.                                                                                                   | off                 |
| `--hq-no-temporal-confidence`                                        | Weights every neighbour equally, however badly it matches.                                                                                                                         | off                 |
| `--mc-blksize`, `--mc-overlap`, `--mc-search`, `--mc-pyramid-levels` | Motion-search geometry. Only used with `--motion-compensation`.                                                                                                                    | `16`, `8`, `4`, `2` |

</details>
---

## Piped input and high bit depth

`--input` also takes `-` (or `pipe:0`) to read a y4m stream from standard input, and `pipe:N` for
an inherited file descriptor:

```bash
ffmpeg -i noisy.mkv -pix_fmt yuv420p -f yuv4mpegpipe - \
  | av-denoise nlmeans --input - \
  | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

Piped input has no scene detection, so the temporal window slides across the whole stream and
`--workers` does not apply.

ffmpeg will not write 10 or 12-bit y4m without `-strict -1`, so high-depth pipelines need it on
the *producing* side:

```bash
ffmpeg -i noisy.mkv -pix_fmt yuv420p10le -strict -1 -f yuv4mpegpipe - \
  | av-denoise nlmeans --input - \
  | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

File input via `--input noisy.mkv` needs no such flag.
---

## [Tuning guide](TUNING-CLI.md)

For more information about how to adjust the algorithms to best fit your needs.

## Example commands

Two dials do almost everything: a core scale parameter to nudge the result when your eye disagrees
with it, and `--preset` for how hard to work. Reach for the scale first, in steps of 0.1, and only
move up the preset ladder once the scale has stopped buying you anything. Each example below
changes one thing from the defaults.

**Clean up a noisy file.** NL4D measures the noise per scene and picks its own threshold.

```bash
av-denoise nl4d --input noisy.mkv | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

**Keep more detail, or take out more grain.** `--lambda-ht-scale` is NL4D's main dial. Lower keeps
more detail, higher removes more noise. It moves luma and chroma together, which matters because
they start from different defaults. Either plane can still be pinned outright with
`--luma-lambda-ht` / `--chroma-lambda-ht`.

```bash
av-denoise nl4d --lambda-ht-scale 0.85 --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

**A noisier source.** Take the scale up by 0.1 and look again. This costs nothing in throughput and
is usually enough on its own.

```bash
av-denoise nl4d --lambda-ht-scale 1.1 --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

**Still noisy once the scale has run out.** Go up the preset ladder. A deeper temporal window is
the strongest lever in the tool for either algorithm, and it is also the one that costs the most
time, so it is worth trying the scale first.

```bash
av-denoise nl4d --preset slow --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

**Trade quality for throughput.** NLMeans is the faster family. Its `fast` variant does no noise
measurement at all.

```bash
av-denoise nlmeans --variant fast --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

**Still grainy under NLMeans.** Tell it the noise is a little stronger than it measured. Move in
steps of 0.1 and judge by eye.

```bash
av-denoise nlmeans --hq-sigma-scale 1.1 --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

**Fine texture getting scrubbed.** The same dial works downward.

```bash
av-denoise nlmeans --hq-sigma-scale 0.9 --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

**Live action with real movement.** Motion compensation keeps the temporal window finding usable
matches instead of smearing. Anime is often better off without it.

```bash
av-denoise nlmeans --motion-compensation --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

**Brightness only, for speed.** Both planes are cleaned by default. Narrow to luma when the colour
is already clean, and you want the time back.

```bash
av-denoise nl4d --channel-mode luma --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

**Pick a specific GPU.** Both flags are global, so they work either side of the subcommand.

```bash
av-denoise --accelerators vulkan --device discrete:1 nl4d --input noisy.mkv \
  | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

<details>
<summary><b>Advanced examples</b> — fixed variant, manual per-plane strength, explicit backends</summary>

These pin `--preset veryfast` to select the `fast` variant, which makes `--strength` an absolute
value rather than the noise multiplier NLMeans-HQ applies. That trades away the per-scene
measurement, so treat them as calibration and debugging recipes rather than a starting point. If
your goal is a better-looking result, the dials above are the ones to reach for first — see
[What not to touch in NLMeans](TUNING-CLI.md#what-not-to-touch-in-nlmeans).

**Y/UV Denoise - ROCm/Vulkan - On GPU 1 - Light Denoise - Spatial - strength=luma:0.3,choma:0.6**
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

**YUV Fused Denoise - Vulkan - On Default GPU - Medium Denoise - Spatial - strength=yuv:0.8**
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
