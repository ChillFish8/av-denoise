"""
vbm3d_denoise.vpy — V-BM3D denoising with vsdenoise on AMD (HIP), driven by vspipe.

This is the external quality and throughput reference the end-to-end
benchmark measures av-denoise against. It comes from the vs-scripting
repo and is kept here so `benchmark_e2e.py` has everything it needs.

Run:
    uv run --directory scripts --group vs vspipe \
        --arg src=input.mkv vs/vbm3d_denoise.vpy - -c y4m \
        | ffmpeg -i - -c:v libx264 -preset slow -crf 16 out.mkv

Optional args (all have defaults):
    --arg sigma=1.5        luma strength
    --arg sigma_c=1.0      chroma strength (0 leaves chroma alone)
    --arg tr=2             temporal radius, 0 is spatial only, 1-3 typical
    --arg profile=normal   fast | lc | normal | high
    --arg refine=1         extra final-estimate passes
    --arg device=0         GPU index, for multi-GPU boxes
    --arg matrix=1         override the colour matrix tag on an untagged source
    --arg progress=1       show bestsource's indexing progress
"""

from __future__ import annotations

import vapoursynth as vs
from vsdenoise import bm3d
from vstools import core


def arg(name: str, default):
    """Reads a `--arg name=value` off the script globals, cast to the default's type."""
    value = globals().get(name, default)
    return value if isinstance(value, type(default)) else type(default)(value)


SRC = str(globals().get("src", "input.mkv"))
SIGMA_Y = arg("sigma", 1.5)
SIGMA_C = arg("sigma_c", 1.0)
TR = arg("tr", 2)
REFINE = arg("refine", 1)
DEVICE = arg("device", 0)
PROFILE_NAME = arg("profile", "normal")
CACHEMODE = arg("cachemode", 4)
# Off by default so a benchmark run's output stays readable.
SHOW_PROGRESS = bool(arg("progress", 0))
MATRIX = int(globals()["matrix"]) if "matrix" in globals() else None

# VERY_NOISY is left out on purpose, vsdenoise only accepts it on the
# reference CPU backend.
PROFILES = {
    "fast": bm3d.Profile.FAST,
    "lc": bm3d.Profile.LOW_COMPLEXITY,
    "normal": bm3d.Profile.NORMAL,
    "high": bm3d.Profile.HIGH,
}
PROFILE = PROFILES[PROFILE_NAME.lower()]

src = core.bs.VideoSource(SRC, showprogress=SHOW_PROGRESS, cachemode=CACHEMODE)

if MATRIX is not None:
    src = core.std.SetFrameProps(src, _Matrix=MATRIX)

# Converted to float once, here, on the whole YUV clip. Pulling an integer
# chroma plane out on its own and converting that would centre it on 0.5 the
# way a luma plane is centred, instead of on 0, and the chroma would come
# back wrong by half a range. The HIP backend only takes 32-bit float, so
# this conversion is not optional.
assert src.format
float_fmt = src.format.replace(bits_per_sample=32, sample_type=vs.FLOAT)
float_src = core.resize.Bicubic(src, format=float_fmt.id)

# vsdenoise filters each plane of a subsampled clip at its own resolution,
# which on 4:2:0 is a quarter of the chroma work and matches what av-denoise
# itself does. `device_id` rides through to both BM3Dv2 calls.
denoised = bm3d(
    float_src,
    sigma=[SIGMA_Y, SIGMA_C, SIGMA_C],
    tr=TR,
    refine=REFINE,
    profile=PROFILE,
    backend=bm3d.Backend.HIP,
    device_id=DEVICE,
)

out = core.resize.Bicubic(denoised, format=src.format.id)
out.set_output()
