# /// script
# requires-python = ">=3.11"
# dependencies = ["vapoursynth", "vapoursynth-bestsource", "numpy"]
# ///
"""Integration tests for the av-denoise VapourSynth plugin.

Run with `just test-vs`. Needs a GPU and a VapourSynth install, which
is why these live outside `cargo nextest`.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np
import vapoursynth as vs

# CubeCL autotune picks a kernel variant based on whether a compiled
# kernel cache exists, and a different variant can shift output by plus
# or minus one. `the_plugin_matches_the_cli_on_the_same_clip` compares
# bytes exactly, so this has to be off before anything denoises, in
# both this process (which runs `avd.NL4D` in-process) and the `cargo
# run` subprocess it launches for the CLI side. Set unconditionally,
# before `core.std.LoadPlugin` runs in `main`, since CubeCL locks its
# global config the moment the first denoiser is created.
os.environ["AV_DENOISE_COMPILATION_CACHE"] = "off"

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
def nl4d_renders_and_preserves_clip_properties():
    src = synthetic_clip()
    out = core.avd.NL4D(src)
    assert out.num_frames == src.num_frames
    assert out.format.id == src.format.id
    assert out.fps == src.fps
    frame = frame_to_array(out.get_frame(4), 0)
    assert frame.shape == frame_to_array(src.get_frame(4), 0).shape


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


def _assert_frame_matches(node, n, linear, label):
    frame = node.get_frame(n)
    for plane in range(frame.format.num_planes):
        got = frame_to_array(frame, plane)
        want = linear[(n, plane)]
        assert np.array_equal(got, want), f"{label}: frame {n} plane {plane} differs"


@test
def random_access_matches_sequential_access_nlmeans():
    """The seek fallback must produce the same frames as a linear render.

    Two separate filter instances so the shuffled run cannot benefit
    from the sequential run's pipeline state, which would make this
    test compare VapourSynth's frame cache instead of the plugin.
    """
    src = synthetic_clip(length=14)

    in_order = core.avd.NLMeans(src)
    start = time.perf_counter()
    linear = {
        (n, plane): frame_to_array(in_order.get_frame(n), plane)
        for n in range(src.num_frames)
        for plane in range(3)
    }
    linear_seconds = time.perf_counter() - start

    shuffled = core.avd.NLMeans(src)
    order = [9, 0, 13, 4, 5, 6, 1, 12, 2, 11, 3, 10, 7, 8]
    start = time.perf_counter()
    for n in order:
        _assert_frame_matches(shuffled, n, linear, "nlmeans")
    shuffled_seconds = time.perf_counter() - start
    print(
        f"    nlmeans: linear {linear_seconds:.3f}s, shuffled {shuffled_seconds:.3f}s, "
        f"ratio {shuffled_seconds / linear_seconds:.2f}x",
        file=sys.stderr,
    )


@test
def random_access_matches_sequential_access_nl4d():
    """Same guarantee as the NLMeans version, for the wider NL4D window.

    NL4D needs 2r frames of context on each side instead of NLMeans'
    r, so its reseed path is exercised harder by the same shuffle.
    """
    src = synthetic_clip(length=14)

    in_order = core.avd.NL4D(src)
    start = time.perf_counter()
    linear = {
        (n, plane): frame_to_array(in_order.get_frame(n), plane)
        for n in range(src.num_frames)
        for plane in range(3)
    }
    linear_seconds = time.perf_counter() - start

    shuffled = core.avd.NL4D(src)
    order = [9, 0, 13, 4, 5, 6, 1, 12, 2, 11, 3, 10, 7, 8]
    start = time.perf_counter()
    for n in order:
        _assert_frame_matches(shuffled, n, linear, "nl4d")
    shuffled_seconds = time.perf_counter() - start
    print(
        f"    nl4d: linear {linear_seconds:.3f}s, shuffled {shuffled_seconds:.3f}s, "
        f"ratio {shuffled_seconds / linear_seconds:.2f}x",
        file=sys.stderr,
    )


@test
def a_sequential_run_after_a_seek_stays_correct_nlmeans():
    """Exercises the fast path resuming right after a reseed."""
    src = synthetic_clip(length=14)

    linear_node = core.avd.NLMeans(src)
    linear = {
        (n, plane): frame_to_array(linear_node.get_frame(n), plane)
        for n in range(src.num_frames)
        for plane in range(3)
    }

    seeked = core.avd.NLMeans(src)
    seeked.get_frame(11)
    for n in [12, 13]:
        _assert_frame_matches(seeked, n, linear, "nlmeans")


@test
def a_sequential_run_after_a_seek_stays_correct_nl4d():
    """Exercises the fast path resuming right after a reseed, on NL4D."""
    src = synthetic_clip(length=14)

    linear_node = core.avd.NL4D(src)
    linear = {
        (n, plane): frame_to_array(linear_node.get_frame(n), plane)
        for n in range(src.num_frames)
        for plane in range(3)
    }

    seeked = core.avd.NL4D(src)
    seeked.get_frame(11)
    for n in [12, 13]:
        _assert_frame_matches(seeked, n, linear, "nl4d")


FORMATS = [
    ("YUV420P8", vs.YUV420P8),
    ("YUV422P8", vs.YUV422P8),
    ("YUV444P8", vs.YUV444P8),
    ("YUV420P10", vs.YUV420P10),
    ("YUV422P10", vs.YUV422P10),
    ("YUV444P10", vs.YUV444P10),
    ("YUV420P12", vs.YUV420P12),
    ("YUV422P12", vs.YUV422P12),
    ("YUV444P12", vs.YUV444P12),
]


@test
def every_supported_format_renders():
    for name, fmt in FORMATS:
        src = synthetic_clip(fmt=fmt, length=8)
        out = core.avd.NL4D(src)
        assert out.format.id == src.format.id, f"{name} changed format"
        assert out.num_frames == src.num_frames, f"{name} changed length"
        assert out.fps == src.fps, f"{name} changed framerate"
        frame = out.get_frame(4)
        for plane in range(frame.format.num_planes):
            frame_to_array(frame, plane)


def expect_error(fn, needle):
    try:
        fn()
    except vs.Error as exc:
        assert needle.lower() in str(exc).lower(), f"expected {needle!r} in {exc}"
    else:
        raise AssertionError(f"expected an error mentioning {needle!r}")


@test
def rgb_input_is_rejected():
    src = core.std.BlankClip(width=160, height=120, length=8, format=vs.RGB24)
    expect_error(lambda: core.avd.NL4D(src), "rgb")


@test
def float_input_is_rejected():
    src = core.std.BlankClip(width=160, height=120, length=8, format=vs.YUV420PS)
    expect_error(lambda: core.avd.NL4D(src), "float")


@test
def gray_input_is_rejected():
    """GRAY is out of scope, core cannot represent a chroma-free source."""
    src = core.std.BlankClip(width=160, height=120, length=8, format=vs.GRAY8)
    expect_error(lambda: core.avd.NL4D(src), "gray")


@test
def sixteen_bit_input_is_rejected():
    """av-denoise's Depth covers 8, 10 and 12 bit only."""
    src = core.std.BlankClip(width=160, height=120, length=8, format=vs.YUV420P16)
    expect_error(lambda: core.avd.NL4D(src), "depth")


@test
def a_large_search_radius_is_guarded():
    """Never allowed to abort the process with no message."""
    src = synthetic_clip(length=8)
    try:
        out = core.avd.NLMeans(src, search_radius=6)
        frame_to_array(out.get_frame(4), 0)
    except vs.Error:
        pass  # A clean rejection is the acceptable outcome.


# Pinned once, on an 8-bit av-denoise --sigma scale, and reused for both
# front ends below so the same number never has to be retyped in two
# unit systems. `--sigma` on the CLI is documented in 8-bit pixel
# units; `avd.NL4D`'s `sigma=` takes the normalised [0, 1] units
# `Nl4dOptions.sigma` itself uses, so the plugin side divides by 255.
PARITY_SIGMA_8BIT = 6.0
PARITY_SIGMA_NORMALIZED = PARITY_SIGMA_8BIT / 255.0


def _parity_source_filter():
    if hasattr(core, "bs"):
        return core.bs.VideoSource
    if hasattr(core, "lsmas"):
        return core.lsmas.LWLibavSource
    raise AssertionError("neither core.bs (BestSource) nor core.lsmas (LSMASHSource) is installed")


# Both front ends resolve an unset temporal_radius to this same value
# (the CLI's `base` preset and `DEFAULT_NL4D_TEMPORAL_RADIUS` in
# av-denoise-vs/src/params.rs), and the test relies on that agreement
# rather than pinning it explicitly, so a future default drift between
# the two shows up here too.
PARITY_TEMPORAL_RADIUS = 2

# nl4d's WindowSpan is `{behind: 2 * radius, ahead: 2 * radius}`
# (av-denoise-core/src/denoiser.rs, `PlanarDenoiser::window_span`), so a
# window reaches `2 * radius` frames behind its centre. Those are the
# only output frames close enough to the clip's start for the two front
# ends' leading-edge padding to differ: `reseed` (what the plugin's
# window rebuild always uses) fills the whole `behind` span by
# repeating the clip's first frame, while the CLI's streaming path
# primes only `radius` duplicates of it before real frames start
# arriving. nl4d's cross-frame accumulator folds a different amount of
# duplicated history in each case, so outputs in this region land close
# but not bit-exact. This is accepted, documented behaviour, not a bug
# — see `av-denoise-vs/README.md`.
PARITY_LEADING_EDGE_FRAMES = 2 * PARITY_TEMPORAL_RADIUS

# The bound the leading edge frames are allowed to drift within, the
# same value and reasoning as `BEHIND_EDGE_TOLERANCE` in
# av-denoise-core/src/frame/tests.rs: full-range luma codes span 255,
# and the extra duplicated history moves the result by at most a
# handful of 8-bit codes.
PARITY_LEADING_EDGE_TOLERANCE = 8


def _max_abs_diff(a, b):
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max())


@test
def the_plugin_matches_the_cli_on_the_same_clip():
    """Both front ends build the same PlaneOptions, so both must agree.

    Renders a short clip through the CLI's y4m output and through the
    plugin, with the same sigma pinned on both sides, then compares
    every plane of every frame. Sigma has to be pinned because the
    CLI's automatic estimate is a temporal EMA over stream history, and
    the plugin's is a fresh window-local measurement, so the two
    legitimately disagree whenever sigma is left automatic. With sigma
    pinned, neither estimator runs, and both front ends should resolve
    to the exact same `PlaneOptions`.

    Frames at `PARITY_LEADING_EDGE_FRAMES` or beyond are asserted
    bit-exact: this is the drift check the CLI/plugin crate split
    exists to provide, and it must not be weakened. The clip's first
    `PARITY_LEADING_EDGE_FRAMES` frames are asserted within
    `PARITY_LEADING_EDGE_TOLERANCE` instead of skipped, so a genuine
    regression there still fails this test, but the accepted
    leading-edge padding difference does not.
    """
    src_path = pathlib.Path("data/parity-clip.y4m")
    if not src_path.exists():
        raise AssertionError(
            f"{src_path} is missing. Create it with: "
            "ffmpeg -i data/bench-sample.mkv -frames:v 12 -pix_fmt yuv420p -y data/parity-clip.y4m"
        )

    source_filter = _parity_source_filter()
    env = dict(os.environ, AV_DENOISE_COMPILATION_CACHE="off")

    with tempfile.NamedTemporaryFile(suffix=".y4m") as out, open(src_path, "rb") as src_stream:
        # The gutter lines the CLI's arguments up with the plugin call they mirror,
        # so the two sides of the parity check can be read against each other. Ruff
        # would put every argument on its own line and lose that.
        # fmt: off
        subprocess.run(  # CLI                                    ||  Plugin, same clip and sigma below
            [                                                     # |
                "cargo", "run", "--release",                      # |
                "-p", "av-denoise", "--features", "binary", "--",  # |
                "nl4d",                                            # |
                "--input", "-",                                    # |  core.avd.NL4D(
                "--sigma", str(PARITY_SIGMA_8BIT),                 # |      src,
            ],                                                     # |      sigma=PARITY_SIGMA_NORMALIZED,
            stdin=src_stream,                                      # |  )
            stdout=out,
            env=env,
            check=True,
        )
        # fmt: on
        out.flush()

        cli = source_filter(out.name)
        src = source_filter(str(src_path))
        plugin = core.avd.NL4D(src, sigma=PARITY_SIGMA_NORMALIZED)

        assert cli.num_frames == plugin.num_frames, (
            f"frame count differs: cli={cli.num_frames} plugin={plugin.num_frames}"
        )
        for n in range(cli.num_frames):
            cli_frame = cli.get_frame(n)
            plugin_frame = plugin.get_frame(n)
            for plane in range(cli_frame.format.num_planes):
                a = frame_to_array(cli_frame, plane)
                b = frame_to_array(plugin_frame, plane)
                if n < PARITY_LEADING_EDGE_FRAMES:
                    diff = _max_abs_diff(a, b)
                    assert diff <= PARITY_LEADING_EDGE_TOLERANCE, (
                        f"frame {n} plane {plane} is a leading-edge frame "
                        f"(< {PARITY_LEADING_EDGE_FRAMES}) and should stay within "
                        f"{PARITY_LEADING_EDGE_TOLERANCE}, got max abs diff {diff}"
                    )
                else:
                    assert np.array_equal(a, b), f"frame {n} plane {plane} differs"


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
