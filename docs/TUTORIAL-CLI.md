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

When compiling the binary, enable the `binary` feature. It pulls in both ingestion paths (BestSource for
file input, y4m for piped input), so there's nothing else to pick between.

The following (non-accelerator) features are available:

- `binary` - Enables the dependencies and code required to compile `av-denoise` as a binary.
    * This pulls in BestSource as a hard dependency. You must install FFmpeg (libavcodec and libavformat 63.1 or
      newer, libavutil 61.1 or newer) and xxHash 0.8 or newer, and `pkg-config` must be able to find them.
    * BestSource saves an index beside each source file so later runs open faster. Use `--no-index-cache` to disable it.
- `avds` - Reads and writes `.avds` grain data files. `binary` turns it on.
- `static-bestsource` - Enables `binary` and links BestSource statically. It reads the library from
  `BESTSOURCE_LIB_DIR` and FFmpeg from `PKG_CONFIG_PATH`. `just static-deps` builds both with vcpkg and
  `just build-static` builds the binary against them.

### Release binaries

Each [GitHub release](https://github.com/ChillFish8/av-denoise/releases) carries a prebuilt CLI for
Linux x86, Windows x86 and macOS on Apple Silicon. BestSource and FFmpeg are linked statically, so
it is not required to have FFmpeg installed on your host system, however, the JIT will still
attempt to dynamically load the CUDA headers and compiler system from your system if you attempt to use
CUDA.

### Cargo install

This builds the binary with the default accelerators enabled (`vulkan`)

```bash
cargo install --locked --git https://github.com/ChillFish8/av-denoise.git av-denoise --features binary
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
    ghcr.io/chillfish8/av-denoise:vulkan-0.6.1 \
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
| `--preset <veryfast..veryslow>`    | Speed vs quality for `nlmeans`, see the ladder below. `nl4d` runs `base` and `slow`.                             | `base`        |
| `--channel-mode <luma,chroma>`     | Which planes to clean.                                                                                           | `luma,chroma` |
| `-A, --accelerators <list>`        | Backends to try, in order. Comma-separated, for example `cuda,vulkan`.                                           | `vulkan`      |
| `-d, --device <spec>`              | `default`, `discrete[:N]`, `integrated[:N]`, `virtual[:N]`, or `cpu`.                                            | `default`     |
| `--progress`                       | Draws a progress bar. Off by default, because anything else writing to the terminal scrambles it.                | off           |

Both subcommands also take:

| Flag                    | What it does                                                                                     | Default  |
|-------------------------|--------------------------------------------------------------------------------------------------|----------|
| `-i, --input <path\|->` | A path is opened with BestSource. `-` or `pipe:0` reads y4m from stdin. Both are split by scene. | required |
| `-W, --workers <N>`     | How many scenes to clean in parallel. Trades GPU memory for throughput.                          | `2`      |
| `--no-index-cache`      | Never read or write a BestSource index beside the input file.                                    | off      |

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

Motion tracking is always on, and the temporal window always reaches at
least one frame, which this algorithm needs. There is no NLMeans weighting pass, so none of
the NLMeans knobs below exist here.

NL4D runs the `base` and `slow` presets. `slow` widens the search for motion blocks whose match
fails, up to `--mc-max-search-escalation` (8), and runs slower. Otherwise, trade speed for quality
with `--temporal-radius` and `--spatial-radius`.

| Flag                               | What it does                                                                                                                                                                                                                                                                         | Default                  |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
| `--temporal-radius <N>`            | How many neighbouring frames to search on each side, between 1 and 8. More frames means more patches to group.                                                                                                                                                                       | `2`                      |
| `--lambda-ht <f>`                  | How aggressively small transform coefficients are zeroed out. Higher removes more noise and more fine detail with it. `--luma-lambda-ht` and `--chroma-lambda-ht` override one plane.                                                                                                | 4.158 luma, 3.234 chroma |
| `--lambda-ht-scale <f>`            | Multiplies the `--lambda-ht` in effect for each plane. The main quality dial, since luma and chroma start from different defaults and this moves both together.                                                                                                                      | `1.0`                    |
| `--spatial-radius <N>`             | Half-width of the candidate search inside the centre frame, between 1 and 16. Most of the search work goes here, since the window covers `(2N+1)^2` positions.                                                                                                                       | `9`                      |
| `--sigma-scale <f>`                | Nudges the measured noise level, the same dial NLMeans spells `--hq-sigma-scale`.                                                                                                                                                                                                    | `1.0`                    |
| `--export-avds <PATH>`             | Writes every grain measurement of the run to a grain data file, one row per output frame. See [Grain data and `av-denoise synthesize`](#grain-data-and-av-denoise-synthesize). Needs luma to be denoised.                                                                            | off                      |

<details>
<summary><b>Expert flags</b>. For calibration and debugging, not everyday tuning.</summary>

| Flag                                                                 | What it does                                                                                                                                                                                                                                                      | Default             |
|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| `--refine <N>`                                                       | Half-width of the window searched around each neighbour frame's motion-predicted position, between 1 and 4. Raise it when motion tracking lands close but not exact.                                                                                              | `2`                 |
| `--sigma <f>`                                                        | Pins the noise level in 8-bit units, turning the per-scene measurement off entirely.                                                                                                                                                                              | measured            |
| `--thsad-scale <f>`                                                  | How badly a neighbour frame may match before its patches stop being trusted.                                                                                                                                                                                      | `1.0`               |
| `--c-min <f>`                                                        | Confidence floor below which a whole neighbour block is skipped rather than scored. A volume left short of frames by the skip makes its group filter from the centre frame alone.                                                                                    | `0.05`              |
| `--no-noise-map`                                                     | Turns off the luma noise map, which scales the luma threshold by how noisy each brightness level and colour is. Also turns off the luma colour gains.                                                                                                                                                          | map on              |
| `--no-pooled-threshold`                                              | Turns off the pooled threshold, which judges each frequency together with its neighbours so faint texture survives.                                                                                                                                               | pooling on          |
| `--mc-blksize`, `--mc-overlap`, `--mc-search`, `--mc-pyramid-levels` | Motion-search geometry. NL4D always tracks motion, so these are always live.                                                                                                                                                                                      | `16`, `8`, `4`, `2` |
| `--mc-max-search-escalation <N>`                                     | Widest radius a motion block that matched badly searches again with, between `--mc-search` and 8. Equal to `--mc-search` turns the wider search off.                                                                                                              | from `--preset`     |

</details>

### `nlmeans` options

Compares small patches of pixels and averages the ones that look alike,
either inside a single frame or across a temporal window.

What `--preset` fills in:

| Preset     | Variant | Temporal radius | Search radius | Motion search ceiling |
|------------|---------|-----------------|---------------|-----------------------|
| `veryfast` | `fast`  | 0               | 2             | `--mc-search`         |
| `fast`     | `hq`    | 1               | 2             | `--mc-search`         |
| `base`     | `hq`    | 2               | 2             | `--mc-search`         |
| `slow`     | `hq`    | 4               | 4             | 8                     |
| `veryslow` | `hq`    | 8               | 4             | 8                     |

The motion search ceiling is `--mc-max-search-escalation`, and it only applies with
`--motion-compensation`. At 8, motion blocks whose match fails search again with a wider window,
which runs slower.

| Flag                    | What it does                                                                                                                                                                                   | Default         |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| `--variant <fast\|hq>`  | `fast` uses fixed weighting and is the cheapest option. `hq` measures the noise level per scene and matches its weighting to it.                                                               | from `--preset` |
| `--temporal-radius <N>` | How many neighbouring frames to use on each side. `0` cleans each frame on its own. This is the strongest lever in the tool.                                                                   | from `--preset` |
| `--hq-sigma-scale <f>`  | Nudges the measured noise level. Raise it when the result still looks noisy, lower it when texture is going. Move in steps of 0.1.                                                             | `1.0`           |
| `--strength <f>`        | How hard to filter. Under `hq` this multiplies the measured noise level, and its default is calibrated per plane and per radius. `--luma-strength` and `--chroma-strength` override one plane. | calibrated      |
| `--motion-compensation` | Tracks where blocks moved, so a deep window lines frames up instead of smearing them. Usually worth it on live action, often not on anime.                                                     | off             |
| `--prefilter <mode>`    | Compares patches against a cleaned reference instead of the noisy input. `nlm[:<scale>]` or `bilateral:<sigma_s>,<sigma_r>`. Costs one extra GPU pass per frame.                               | `none`          |

<details>
<summary><b>Expert flags</b>. For calibration and debugging, not everyday tuning.</summary>

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
| `--mc-max-search-escalation <N>`                                     | Widest radius a badly matched motion block searches again with, between `--mc-search` and 8. Equal to `--mc-search` turns it off. Only used with `--motion-compensation`.          | from `--preset`     |

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

Piped input is split by scene just like a file, and `--workers` applies to both.

ffmpeg will not write 10 or 12-bit y4m without `-strict -1`, so high-depth pipelines need it on
the *producing* side:

```bash
ffmpeg -i noisy.mkv -pix_fmt yuv420p10le -strict -1 -f yuv4mpegpipe - \
  | av-denoise nlmeans --input - \
  | ffmpeg -f yuv4mpegpipe -i - -c:v ffv1 clean.mkv
```

File input via `--input noisy.mkv` needs no such flag.

---

## Grain data and `av-denoise synthesize`

`--export-avds` on `nl4d` measures the source's grain while it denoises, and writes every
measurement to a grain data file with one row per output frame. The file is Parquet, so any
Parquet reader can open it.

```bash
av-denoise nl4d --input noisy.mkv --export-avds run.avds > /dev/null
```

`av-denoise synthesize` builds grain outputs from that file. It needs no GPU and no source, so you
can try different options without denoising again.

### `synthesize av1-fgs`

Writes an AV1 film grain table for all of the source's grain, as one frame shows it. Add it to a
finished AV1 encode with `regrain apply`, which applies each scene's entry. SVT-AV1 `--fgs-table`
and `grav1synth apply` apply only its first entry, so they suit a single-scene table only.

The table holds luma and chroma grain by default. Chroma grain needs 4:2:0 video from a run that
denoised chroma. Otherwise the command warns and writes luma grain only.

The table's timestamps count the run's output frames from 0 at the source frame rate. It fits
constant frame rate sources and encodes of the run's output.

| Flag                    | What it does                                                                                                                                  | Default       |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `--input <PATH>`        | The grain data file written by `--export-avds`.                                                                                               | required      |
| `--output <PATH>`       | Where to write the table.                                                                                                                     | required      |
| `--planes <PLANES>`     | Which planes get grain, `luma` or `luma,chroma`. Chroma alone is refused.                                                                     | `luma,chroma` |
| `--fine-noise <AMOUNT>` | The share of the fine, fresh scan noise moved out of the grain's texture, between 0 and 1. Strength stays as measured. `0` moves none.        | `0.5`         |
| `--strength <FACTOR>`   | A multiple of the source's grain, between 0 and 4. `1` matches the source. A lower value suits an encode that keeps some grain itself.        | `1`           |

```bash
av-denoise synthesize av1-fgs --input run.avds --output grain.tbl --fine-noise 0.25 --strength 0.8
```

### `synthesize report`

Writes the per-frame grain report. See [Reading the grain report](#reading-the-grain-report).

| Flag              | What it does                                    | Default  |
|-------------------|-------------------------------------------------|----------|
| `--input <PATH>`  | The grain data file written by `--export-avds`. | required |
| `--output <PATH>` | Where to write the report.                      | required |

```bash
av-denoise synthesize report --input run.avds --output grain.jsonl
```

---

## Reading the grain report

`synthesize report` writes one JSON object per output frame, one per line, in frame order.
It describes the grain in the source, not what the denoiser removed.

```json
{"frame":120,"scene":3,"cells":19804,"strength":0.93,"strength_cb":0.41,"strength_cr":0.38,"persistence":[0.62,0.46,0.34,0.27,0.23,0.21,0.2,0.2]}
```

| Key           | Meaning                                                                                                                           |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `frame`       | Output frame index, starting at 0.                                                                                                |
| `scene`       | Index of the scene av-denoise cut, in frame order. Encoders may cut their own scenes, so treat it as a hint.                      |
| `cells`       | How many 8x8 cells the reading rests on, counting the frame before and after.                                                     |
| `strength`    | The grain standard deviation in 8-bit code values, whatever the bit depth.                                                        |
| `strength_cb` | The same for the Cb plane. `null` when chroma is not denoised, such as with `--channel-mode luma`.                                |
| `strength_cr` | The same for the Cr plane. `null` when chroma is not denoised.                                                                    |
| `persistence` | Eight entries, for 1 to 8 frames away. Each is the share of grain unchanged that far before and after the frame, between 0 and 1. |

Grain is measured in flat areas of luma, at the same screen position with no motion compensation.
The chroma strengths come from flat areas of each chroma plane, between each frame and its neighbours.
Grain fixed to the screen reads as persistent, and texture carried by a pan reads as moving.

A reading is `null` when fewer than 200 flat cells back it at that distance. A side of the window
counts only when that frame is in the same scene. A scene shorter than 9 frames never fills the
last entries, and a single-frame scene is all `null`. Very busy, black or blown-out frames can also
be `null`.

The level and fall of the curve tell the grain apart:

- Moving grain is already near 0 at the first entry.
- Slow grain starts high and falls over a few frames.
- Static grain stays high to the last entry.
- Grain held for N frames then replaced reads about `(N - 1) / N` at the first entry and falls to near 0 by entry N.

A source that repeats a whole frame, grain included, reads as persistent for that pair.
Real readings never reach exactly 0 or 1. Anything fixed in a held shot, such as compression
residue, adds a little persistence to moving grain, and part of a flat area's grain is always fresh.

To summarise a scene, take the median of each `persistence` entry over its rows, skipping `null`.
As a starting point, a first entry under 0.4 suggests moving grain. Otherwise, a last entry under
0.34 suggests slow grain, and anything higher suggests static grain.

---

## [Tuning guide](TUNING-CLI.md)

For more information about how to adjust the algorithms to best fit your needs.

## Example commands

Two dials do almost everything: a core scale parameter to nudge the result when your eye disagrees
with it, and the temporal window for how hard to work. Reach for the scale first, in steps of 0.1, and only
widen the window once the scale has stopped buying you anything. Each example below
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

**Still noisy once the scale has run out.** Raise `--temporal-radius`. A deeper temporal window is
the strongest lever in the tool for either algorithm, and it is also the one that costs the most
time, so it is worth trying the scale first.

```bash
av-denoise nl4d --temporal-radius 4 --input noisy.mkv \
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
<summary><b>Advanced examples</b>. Fixed variant, manual per-plane strength and explicit backends.</summary>

These pin `--preset veryfast` to select the `fast` variant, which makes `--strength` an absolute
value rather than the noise multiplier NLMeans-HQ applies. That trades away the per-scene
measurement, so treat them as calibration and debugging recipes rather than a starting point. If
your goal is a better-looking result, the dials above are the ones to reach for first. See
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
