#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["vapoursynth-bm3dhip==2.16"]
# ///
"""Runs bm3dhip over a clip and writes the result to stdout as y4m.

Driven directly from Python rather than through `vspipe`, because the
system `vspipe` is bound to the system Python and cannot see a uv
environment. `clip.output` writes the same y4m `vspipe -c y4m` would.

`vapoursynth-bm3dhip` is pinned to 2.16 on purpose. The wheels bundle a
HIP runtime but no device bitcode, so the runtime JIT-links against
whatever ROCm the machine has. 2.17.x bundles HIP 7.14, which needs
symbols this machine's ROCm 7.2.4 bitcode does not carry, and every
call fails with an undefined `__amd_streamOpsDecrement` reported as
`hipStreamCreateWithFlags ... out of memory`. 2.16 bundles HIP 7.0 and
works. Re-check this pin if the system ROCm is upgraded.
"""

from __future__ import annotations

import argparse
import sys

import vapoursynth as vs

FFMS2 = "/usr/lib/vapoursynth/libffms2.so"


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument(
        "--sigma",
        default="3.0",
        help="per-plane sigma, comma separated, in the 8-bit scale bm3dhip expects",
    )
    p.add_argument(
        "--radius",
        type=int,
        default=0,
        help="0 runs spatial BM3D, above that runs V-BM3D over 2*radius+1 frames",
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--frames", type=int, default=0, help="0 means the whole clip")
    p.add_argument(
        "--basic-only",
        action="store_true",
        help="stop after the hard-threshold pass, skipping the Wiener pass",
    )
    return p.parse_args()


def main() -> None:
    args = parse_cli()
    core = vs.core
    core.std.LoadPlugin(FFMS2)

    src = core.ffms2.Source(args.input)
    if args.frames:
        src = core.std.Trim(src, first=0, last=args.frames - 1)

    sigma = [float(v) for v in args.sigma.split(",")]
    while len(sigma) < 3:
        sigma.append(sigma[-1])

    # Converted to float once, here, on the whole YUV clip. Pulling an
    # integer chroma plane out as GRAY first and converting that would
    # centre it on 0.5 the way a luma plane is centred, instead of on 0,
    # and the chroma would come back wrong by half a range.
    float_src = core.resize.Bicubic(src, format=vs.YUV420PS)

    def denoise(work: vs.VideoNode, sig: float) -> vs.VideoNode:
        """Runs both BM3D passes over one plane at its own resolution.

        `BM3Dv2` aggregates its own temporal output, unlike `BM3D`, which
        hands back a stack of `2 * radius + 1` frames for `VAggregate` to
        collapse. Calling `VAggregate` on top of `BM3Dv2` leaves the
        clip un-collapsed and the y4m header oversized.
        """
        basic = core.bm3dhip.BM3Dv2(
            work, sigma=[sig], radius=args.radius, device_id=args.device
        )
        if args.basic_only:
            return basic
        return core.bm3dhip.BM3Dv2(
            work, ref=basic, sigma=[sig], radius=args.radius, device_id=args.device
        )

    # Each plane is filtered at its own resolution rather than upsampling
    # chroma to 4:4:4 first. On 4:2:0 that is a quarter of the chroma work
    # and it matches what av-denoise itself does, which is what makes the
    # two arms comparable on both speed and quality.
    planes = [
        denoise(core.std.ShufflePlanes(float_src, i, vs.GRAY), sigma[i])
        for i in range(3)
    ]
    merged = core.std.ShufflePlanes(planes, [0, 0, 0], vs.YUV)
    # ShufflePlanes drops the frame props the format conversion needs, so
    # they are copied back off the float clip before converting down.
    merged = core.std.CopyFrameProps(merged, float_src)
    out = core.resize.Bicubic(merged, format=src.format.id)

    out.output(sys.stdout.buffer, y4m=True)


if __name__ == "__main__":
    main()
