# /// script
# requires-python = ">=3.11"
# dependencies = ["vapoursynth", "numpy"]
# ///
"""Integration tests for the av-denoise VapourSynth plugin.

Run with `just test-vs`. Needs a GPU and a VapourSynth install, which
is why these live outside `cargo nextest`.
"""

import argparse
import pathlib
import sys

import numpy as np
import vapoursynth as vs

core = vs.core

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def frame_to_array(frame, plane):
    return np.asarray(frame[plane]).copy()


def synthetic_clip(width=160, height=120, length=12, fmt=vs.YUV420P8, seed=7):
    """A deterministic noisy clip with real temporal structure.

    Each frame is a spatial ramp offset by the frame number plus
    deterministic pseudo-random dither, so both temporal motion and
    per-pixel noise are present for a temporal denoiser to exploit.
    The ramp and noise amplitude scale to the output format's bit
    depth, so this also works at 10-bit and 12-bit. Each frame seeds
    its own generator from (seed, n), so frame content depends only
    on the frame index and not on the order frames are requested in.
    """
    base = core.std.BlankClip(width=width, height=height, length=length, format=fmt)

    def draw(n, f):
        out = f.copy()
        rng = np.random.default_rng((seed, n))
        for plane in range(out.format.num_planes):
            arr = np.asarray(out[plane])
            h, w = arr.shape
            bits = out.format.bits_per_sample
            max_val = (1 << bits) - 1
            scale = max_val / 255
            ramp = (np.add.outer(np.arange(h), np.arange(w)) + n * 3) % 200
            ramp = ramp * scale
            noise = rng.integers(-12, 13, size=(h, w)) * scale
            arr[:] = np.clip(ramp + noise, 0, max_val).astype(arr.dtype)
        return out

    return core.std.ModifyFrame(base, base, draw)


@test
def plugin_loads():
    assert "avd" in [p.namespace for p in core.plugins()], "avd namespace not registered"


@test
def passthrough_returns_source_frames():
    src = synthetic_clip()
    out = core.avd.Passthrough(src)
    assert out.num_frames == src.num_frames
    assert out.format.id == src.format.id
    for n in (0, 5, src.num_frames - 1):
        a = frame_to_array(src.get_frame(n), 0)
        b = frame_to_array(out.get_frame(n), 0)
        assert np.array_equal(a, b), f"frame {n} differs"


@test
def synthetic_clip_content_is_independent_of_request_order():
    forward = synthetic_clip()
    shuffled = synthetic_clip()
    for n in reversed(range(shuffled.num_frames)):
        shuffled.get_frame(n)
    for n in range(forward.num_frames):
        a_frame = forward.get_frame(n)
        b_frame = shuffled.get_frame(n)
        for plane in range(a_frame.format.num_planes):
            a = frame_to_array(a_frame, plane)
            b = frame_to_array(b_frame, plane)
            assert np.array_equal(a, b), f"frame {n} plane {plane} differs by request order"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="target/release/libav_denoise_vs.so")
    ap.add_argument("--filter", default="", help="only run tests whose name contains this")
    args = ap.parse_args()

    core.std.LoadPlugin(str(pathlib.Path(args.plugin).resolve()))

    failed = 0
    for fn in TESTS:
        if args.filter and args.filter not in fn.__name__:
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}", file=sys.stderr)
        else:
            print(f"ok   {fn.__name__}")

    if failed:
        print(f"\n{failed} failed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
