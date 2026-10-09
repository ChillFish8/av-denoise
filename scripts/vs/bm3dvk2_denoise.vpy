"""
V-BM3D denoising with bm3dvk2 on Vulkan, driven by vspipe.

The Vulkan counterpart of `vbm3d_denoise.vpy`, filtering with the same
strength, radius, profile and refine count so the two arms do the same work.
bm3dvk2 needs VapourSynth R81, which cannot load the HIP and CUDA plugins
the other script uses, so this one runs from the `vs-r81` project instead.

Run:
    uv run --directory scripts/vs-r81 vspipe \
        --arg src=input.mkv ../vs/bm3dvk2_denoise.vpy - -c y4m \
        | ffmpeg -i - -c:v libx264 -preset slow -crf 16 out.mkv

Optional args (all have defaults):
    --arg sigma=1.5        luma strength
    --arg sigma_c=1.0      chroma strength (0 leaves chroma alone)
    --arg tr=2             temporal radius, 0 is spatial only, 1-3 typical
    --arg profile=normal   fast | lc | normal | high
    --arg refine=1         extra final-estimate passes
    --arg device=0         Vulkan device index, as listed by core.vulkan_devices
    --arg matrix=1         override the colour matrix tag on an untagged source
    --arg progress=1       show bestsource's indexing progress
"""

from __future__ import annotations

import vapoursynth as vs

core = vs.core


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

# Named after vsdenoise's profiles, so an arm here and its `vbm3d_denoise.vpy`
# twin take the same `--arg profile=`.
PROFILES = {
    "fast": "fast",
    "lc": "lc",
    "normal": "np",
    "high": "high",
}
PROFILE = PROFILES[PROFILE_NAME.lower()]

# Selecting a device is only allowed before the first GPU filter is created.
core.set_vulkan_device(DEVICE)


def denoise_plane(plane: vs.VideoNode, sigma: float) -> vs.VideoNode:
    """Runs a basic estimate and `REFINE` final estimates over one gray plane.

    `matrix=0` turns off bm3dvk2's per-matrix sigma scaling, so `sigma` means
    the same as it does to bm3dhip and bm3dcuda, which have no such scaling.
    """
    if sigma == 0:
        return plane

    basic = core.bm3dvk2.BM3D(plane, profile=PROFILE, sigma=sigma, radius=TR, matrix=0)

    final = basic
    for _ in range(REFINE):
        final = core.bm3dvk2.BM3D(
            plane,
            ref=final,
            profile=PROFILE,
            sigma=sigma,
            radius=TR,
            matrix=0,
        )

    return final


src = core.bs.VideoSource(SRC, showprogress=SHOW_PROGRESS, cachemode=CACHEMODE)

if MATRIX is not None:
    src = core.std.SetFrameProps(src, _Matrix=MATRIX)

# Converted to float once, here, on the whole YUV clip. Pulling an integer
# chroma plane out on its own and converting that would centre it on 0.5 the
# way a luma plane is centred, instead of on 0, and the chroma would come
# back wrong by half a range.
assert src.format
float_fmt = src.format.replace(bits_per_sample=32, sample_type=vs.FLOAT)
float_src = core.resize.Bicubic(src, format=float_fmt.id)

# bm3dvk2 takes Gray or RGB only, so each plane is filtered as its own gray
# clip at its own resolution. On 4:2:0 that is a quarter of the chroma work,
# the same split vsdenoise hands bm3dhip in the other script.
planes = core.std.SplitPlanes(float_src)
sigmas = (SIGMA_Y, SIGMA_C, SIGMA_C)
denoised_planes = [denoise_plane(plane, sigma) for plane, sigma in zip(planes, sigmas)]
denoised = core.std.ShufflePlanes(denoised_planes, planes=[0, 0, 0], colorfamily=vs.YUV)

out = core.resize.Bicubic(denoised, format=src.format.id)
out.set_output()
