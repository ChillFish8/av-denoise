#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["vapoursynth", "vapoursynth-bestsource"]
# ///
"""Renders one avd.NL4D or avd.NLMeans arm at its natural defaults, in
frame order, and writes y4m to stdout for ffmpeg to encode.

Ad hoc script for the brick VS-vs-CLI quality comparison
(2026-08-29). Not part of the test suite; `av-denoise-vs/tests/vs_harness.py`
is the real integration test and stays the source of truth for parity
mechanics.

CubeCL autotune has to be pinned off before any denoiser is created,
the same as in vs_harness.py, so this happens before core.std.LoadPlugin.

Reads with core.bs (BestSource) and calls VideoNode.output(..., prefetch=0)
so frames are requested strictly in order 0..N-1 -- never vspipe, and
never out-of-order random access, which would confound HQ's
order-dependent noise estimation.
"""

import argparse
import os
import pathlib
import sys

os.environ["AV_DENOISE_COMPILATION_CACHE"] = "off"

import vapoursynth as vs  # noqa: E402

core = vs.core


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="target/release/libav_denoise_vs.so")
    ap.add_argument("--input", required=True)
    ap.add_argument("--algo", choices=["nl4d", "nlmeans"], required=True)
    args = ap.parse_args()

    core.std.LoadPlugin(str(pathlib.Path(args.plugin).resolve()))

    src = core.bs.VideoSource(args.input)

    if args.algo == "nl4d":
        out = core.avd.NL4D(src)
    else:
        out = core.avd.NLMeans(src)

    out.output(sys.stdout.buffer, y4m=True, prefetch=0)


if __name__ == "__main__":
    main()
