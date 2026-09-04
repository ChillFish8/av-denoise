"""
avd_denoise.vpy — av-denoise through its VapourSynth plugin, driven by vspipe.

This is the plugin arm of the end-to-end benchmark. The CLI arms measure
`target/release/av-denoise` directly, and this measures the same denoisers
reached through `vsavd`, so the two front ends can be compared against each
other and against the V-BM3D reference in `vbm3d_denoise.vpy`.

Run:
    uv run --directory scripts --group vs vspipe \
        --arg src=input.mkv --arg algo=nl4d vs/avd_denoise.vpy - -c y4m \
        | ffmpeg -i - -c:v libx264 -preset slow -crf 16 out.mkv

`algo` picks the denoiser. Every other dial is optional, and an unset dial is
not forwarded at all, so av-denoise's own default stands.

    --arg algo=nl4d                 nl4d | nlmhq | nlm
    --arg preset=base               veryfast | fast | base | slow | veryslow
    --arg channel_mode=lumachroma   luma | chroma | lumachroma | yuv, and
                                    lumachroma is already the default
    --arg device=discrete:1         compute device, `cpu` is for smoke tests only
    --arg accelerators=vulkan       comma separated, tried in order
    --arg bits=10                   working depth, 8 | 10 | 12
    --arg progress=1                show bestsource's indexing progress

NL4D only:

    --arg lambda_ht_scale=1.0       the main strength dial, steps of ~0.05
    --arg lambda_ht=...             pins luma and chroma to one absolute threshold
    --arg sigma_scale=1.0           nudges the measured noise level
    --arg spatial_radius=4          the speed dial, work grows as (2r+1)^2
    --arg refine=1                  search half-width around each predicted position

NLMeans-HQ and NLMeans:

    --arg strength=1.0              overall filter strength
    --arg luma_strength=1.0         strength override for luma
    --arg chroma_strength=1.0       strength override for chroma
    --arg motion_compensation=1     motion-compensate neighbours before matching
    --arg sigma_scale=1.0           NLMeans-HQ only, nudges the measured noise
                                    level, reach for this first when grain
                                    survives filtering
"""

from __future__ import annotations

import vapoursynth as vs
import vsavd
from vstools import core

# av-denoise works on integer YUV only, at one of these depths.
SUPPORTED_BITS = (8, 10, 12)

ALGOS = ("nl4d", "nlmhq", "nlm")


def opt(name: str, cast=str):
    """Reads an optional `--arg name=value`, or None when it was not passed."""
    if name not in globals():
        return None
    return cast(globals()[name])


def flag(value) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


def accelerator_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


SRC = str(globals().get("src", "input.mkv"))
CACHEMODE = int(globals().get("cachemode", 4))
# Off by default so a benchmark run's output stays readable.
SHOW_PROGRESS = flag(globals().get("progress", 0))

ALGO = str(globals().get("algo", "nl4d")).lower()
if ALGO not in ALGOS:
    raise ValueError(f"algo={ALGO} is not one of {ALGOS}")

src = core.bs.VideoSource(SRC, showprogress=SHOW_PROGRESS, cachemode=CACHEMODE)
assert src.format

# The depth is fixed once, here, on the whole YUV clip. av-denoise rejects
# float samples outright and only takes 8, 10 or 12 bit integer YUV, so a
# source outside that set is moved to 10 bit and moved back afterwards.
default_bits = src.format.bits_per_sample
if src.format.sample_type == vs.FLOAT or default_bits not in SUPPORTED_BITS:
    default_bits = 10

WORK_BITS = int(globals().get("bits", default_bits))
if WORK_BITS not in SUPPORTED_BITS:
    raise ValueError(f"bits={WORK_BITS} is not one of {SUPPORTED_BITS}")

work_fmt = src.format.replace(bits_per_sample=WORK_BITS, sample_type=vs.INTEGER)
work = (
    src
    if work_fmt.id == src.format.id
    else core.resize.Bicubic(src, format=work_fmt.id, dither_type="error_diffusion")
)

common = dict(
    preset=opt("preset"),
    channel_mode=opt("channel_mode"),
    device=opt("device"),
    accelerators=opt("accelerators", accelerator_list),
)

nlmeans = dict(
    strength=opt("strength", float),
    luma_strength=opt("luma_strength", float),
    chroma_strength=opt("chroma_strength", float),
    motion_compensation=opt("motion_compensation", flag),
)

if ALGO == "nl4d":
    denoised = vsavd.Nl4d(
        work,
        **common,
        lambda_ht_scale=opt("lambda_ht_scale", float),
        lambda_ht=opt("lambda_ht", float),
        sigma_scale=opt("sigma_scale", float),
        spatial_radius=opt("spatial_radius", int),
        refine=opt("refine", int),
    )
elif ALGO == "nlmhq":
    denoised = vsavd.NlmHQ(work, **common, **nlmeans, sigma_scale=opt("sigma_scale", float))
else:
    denoised = vsavd.Nlm(work, **common, **nlmeans)

out = (
    denoised
    if work_fmt.id == src.format.id
    else core.resize.Bicubic(denoised, format=src.format.id, dither_type="error_diffusion")
)
out.set_output()
