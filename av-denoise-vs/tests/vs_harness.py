# /// script
# requires-python = ">=3.11"
# dependencies = ["vapoursynth", "vapoursynth-bestsource", "numpy"]
# ///
"""Integration tests for the av-denoise VapourSynth plugin.

Run with `just test-vs`. Needs a GPU and a VapourSynth install, which
is why these live outside `cargo nextest`.
"""

import argparse
import collections
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


def _max_abs_diff(a, b):
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max())


# Pinned once, on an 8-bit av-denoise --sigma scale, and reused for both
# front ends below so the same number never has to be retyped in two
# unit systems. `--sigma` on the CLI is documented in 8-bit pixel
# units. `avd.NL4D`'s `sigma=` takes the normalised units between 0 and 1
# that `Nl4dOptions.sigma` itself uses, so the plugin side divides by 255.
PARITY_SIGMA_8BIT = 6.0
PARITY_SIGMA_NORMALIZED = PARITY_SIGMA_8BIT / 255.0


# Bounds how far a frame rendered after a seek may sit from a sequential render. The
# noise level is an EMA over stream history, and a seek changes the history it has seen.
EMA_DRIFT_TOLERANCE = 16

PINNED_SIGMA = {"sigma": PARITY_SIGMA_NORMALIZED}


def _assert_frame_matches(node, n, linear, label):
    frame = node.get_frame(n)
    for plane in range(frame.format.num_planes):
        got = frame_to_array(frame, plane)
        want = linear[(n, plane)]
        assert np.array_equal(got, want), f"{label}: frame {n} plane {plane} differs"


def _assert_frame_close(node, n, linear, label):
    frame = node.get_frame(n)
    for plane in range(frame.format.num_planes):
        got = frame_to_array(frame, plane)
        want = linear[(n, plane)]
        diff = _max_abs_diff(got, want)
        assert diff <= EMA_DRIFT_TOLERANCE, (
            f"{label}: frame {n} plane {plane} drifts by {diff}, over {EMA_DRIFT_TOLERANCE}"
        )


def _render_linear(node):
    return {
        (n, plane): frame_to_array(node.get_frame(n), plane)
        for n in range(node.num_frames)
        for plane in range(3)
    }


SHUFFLED_ORDER = [9, 0, 13, 4, 5, 6, 1, 12, 2, 11, 3, 10, 7, 8]


def _check_random_access(make_filter, label, assert_frame):
    """
    Two separate filter instances so the shuffled run cannot benefit
    from the sequential run's pipeline state, which would make this
    test compare VapourSynth's frame cache instead of the plugin.
    """
    src = synthetic_clip(length=14)

    start = time.perf_counter()
    linear = _render_linear(make_filter(src))
    linear_seconds = time.perf_counter() - start

    shuffled = make_filter(src)
    start = time.perf_counter()
    for n in SHUFFLED_ORDER:
        assert_frame(shuffled, n, linear, label)
    shuffled_seconds = time.perf_counter() - start
    print(
        f"    {label}: linear {linear_seconds:.3f}s, shuffled {shuffled_seconds:.3f}s, "
        f"ratio {shuffled_seconds / linear_seconds:.2f}x",
        file=sys.stderr,
    )


def _check_sequential_run_after_seek(make_filter, label, assert_frame):
    """Exercises the fast path resuming right after a reseed."""
    src = synthetic_clip(length=14)
    linear = _render_linear(make_filter(src))

    seeked = make_filter(src)
    seeked.get_frame(11)
    for n in [12, 13]:
        assert_frame(seeked, n, linear, label)


def _nlmeans_pinned(src):
    return core.avd.NLMeans(src, **PINNED_SIGMA)


def _nl4d_pinned(src):
    return core.avd.NL4D(src, **PINNED_SIGMA)


@test
def random_access_stays_close_to_sequential_access_nlmeans():
    _check_random_access(core.avd.NLMeans, "nlmeans", _assert_frame_close)


@test
def random_access_stays_close_to_sequential_access_nl4d():
    _check_random_access(core.avd.NL4D, "nl4d", _assert_frame_close)


@test
def random_access_matches_sequential_access_with_pinned_sigma_nlmeans():
    _check_random_access(_nlmeans_pinned, "nlmeans pinned", _assert_frame_matches)


@test
def random_access_matches_sequential_access_with_pinned_sigma_nl4d():
    _check_random_access(_nl4d_pinned, "nl4d pinned", _assert_frame_matches)


@test
def a_sequential_run_after_a_seek_stays_close_nlmeans():
    _check_sequential_run_after_seek(core.avd.NLMeans, "nlmeans", _assert_frame_close)


@test
def a_sequential_run_after_a_seek_stays_close_nl4d():
    _check_sequential_run_after_seek(core.avd.NL4D, "nl4d", _assert_frame_close)


@test
def a_sequential_run_after_a_seek_matches_with_pinned_sigma_nlmeans():
    _check_sequential_run_after_seek(_nlmeans_pinned, "nlmeans pinned", _assert_frame_matches)


@test
def a_sequential_run_after_a_seek_matches_with_pinned_sigma_nl4d():
    _check_sequential_run_after_seek(_nl4d_pinned, "nl4d pinned", _assert_frame_matches)


PARALLEL_THREADS = 8


def _render_in_parallel_bursts(node, first_frame=0):
    """
    Renders frames from `first_frame` on through `get_frame_async`, one burst at a time.

    Each burst is submitted highest frame first, so the filter is likely to see its requests out of order.
    """
    rendered = {}
    for first in range(first_frame, node.num_frames, PARALLEL_THREADS):
        indices = range(first, min(first + PARALLEL_THREADS, node.num_frames))
        futures = {n: node.get_frame_async(n) for n in reversed(indices)}
        for n, future in futures.items():
            _store_frame(rendered, n, future.result())
    return rendered


def _render_with_sliding_window(node):
    """
    Renders every frame keeping `PARALLEL_THREADS` requests in flight, like vspipe does.

    Frames are requested in ascending order and the window is topped up as the oldest one completes.
    """
    rendered = {}
    pending = collections.deque()
    next_frame = 0
    while next_frame < node.num_frames or pending:
        while next_frame < node.num_frames and len(pending) < PARALLEL_THREADS:
            pending.append((next_frame, node.get_frame_async(next_frame)))
            next_frame += 1

        n, future = pending.popleft()
        _store_frame(rendered, n, future.result())
    return rendered


def _store_frame(rendered, n, frame):
    for plane in range(frame.format.num_planes):
        rendered[(n, plane)] = frame_to_array(frame, plane)


def _assert_parallel_render_matches(make_filter, label, render, max_drift=0):
    """
    A multi-threaded render must match a sequential one frame for frame, within `max_drift`.

    The parallel filter is built after the thread count changes, so it is created under that count.
    """
    src = synthetic_clip(length=40)
    linear = _render_linear(make_filter(src))

    previous_threads = core.num_threads
    core.num_threads = PARALLEL_THREADS
    try:
        rendered = render(make_filter(src))
    finally:
        core.num_threads = previous_threads

    assert rendered, f"{label}: nothing was rendered"
    for (n, plane), got in rendered.items():
        want = linear[(n, plane)]
        diff = _max_abs_diff(got, want)
        assert diff <= max_drift, f"{label}: frame {n} plane {plane} differs by {diff}"


@test
def parallel_bursts_match_sequential_access_nl4d():
    _assert_parallel_render_matches(core.avd.NL4D, "nl4d", _render_in_parallel_bursts)


@test
def parallel_bursts_match_sequential_access_nlmeans():
    _assert_parallel_render_matches(core.avd.NLMeans, "nlmeans", _render_in_parallel_bursts)


def _render_bursts_after_seek(node):
    return _render_in_parallel_bursts(node, first_frame=17)


@test
def parallel_bursts_after_a_seek_stay_close_to_sequential_access_nl4d():
    _assert_parallel_render_matches(
        core.avd.NL4D, "nl4d seek", _render_bursts_after_seek, max_drift=EMA_DRIFT_TOLERANCE
    )


@test
def parallel_bursts_after_a_seek_match_sequential_access_with_pinned_sigma_nl4d():
    _assert_parallel_render_matches(_nl4d_pinned, "nl4d seek pinned", _render_bursts_after_seek)


def _render_from_an_early_frame_first(node):
    """
    Renders frame 7 on its own before any other request, then the rest in order.

    Pins the race where a burst's highest frame reaches the filter before the frames below it.
    """
    rendered = {}
    _store_frame(rendered, 7, node.get_frame(7))
    for n in range(node.num_frames):
        if n != 7:
            _store_frame(rendered, n, node.get_frame(n))
    return rendered


@test
def an_early_first_request_matches_sequential_access_nl4d():
    _assert_parallel_render_matches(core.avd.NL4D, "nl4d early first", _render_from_an_early_frame_first)


@test
def sliding_window_requests_match_sequential_access_nl4d():
    _assert_parallel_render_matches(core.avd.NL4D, "nl4d window", _render_with_sliding_window)


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


def _parity_source_filter():
    if hasattr(core, "bs"):
        return core.bs.VideoSource
    if hasattr(core, "lsmas"):
        return core.lsmas.LWLibavSource
    raise AssertionError("neither core.bs (BestSource) nor core.lsmas (LSMASHSource) is installed")


PARITY_CLIP = pathlib.Path("data/parity-clip.y4m")


def _render_cli_to(out_file, cli_args):
    """Runs the CLI over the parity clip, writing its y4m output to `out_file`."""
    if not PARITY_CLIP.exists():
        raise AssertionError(
            f"{PARITY_CLIP} is missing. Create it with: "
            "ffmpeg -i data/bench-sample.mkv -frames:v 12 -pix_fmt yuv420p -y data/parity-clip.y4m"
        )

    env = dict(os.environ, AV_DENOISE_COMPILATION_CACHE="off")
    command = [
        "cargo", "run", "--release",
        "-p", "av-denoise", "--features", "binary", "--",
        "nl4d", "--input", "-",
        *cli_args,
    ]  # fmt: skip
    with open(PARITY_CLIP, "rb") as src_stream:
        subprocess.run(command, stdin=src_stream, stdout=out_file, env=env, check=True)
    out_file.flush()


def _parity_frame_diffs(cli, plugin):
    """Yields `(frame, plane, cli_plane, plugin_plane)` for every plane of every frame."""
    assert cli.num_frames == plugin.num_frames, (
        f"frame count differs: cli={cli.num_frames} plugin={plugin.num_frames}"
    )
    for n in range(cli.num_frames):
        cli_frame = cli.get_frame(n)
        plugin_frame = plugin.get_frame(n)
        for plane in range(cli_frame.format.num_planes):
            yield n, plane, frame_to_array(cli_frame, plane), frame_to_array(plugin_frame, plane)


@test
def the_plugin_matches_the_cli_on_the_same_clip():
    """Both front ends build the same PlaneOptions, so both must agree.

    Renders a short clip through the CLI's y4m output and through the
    plugin, with the same sigma pinned on both sides, then compares
    every plane of every frame byte for byte. With sigma pinned, neither
    estimator runs, and both front ends resolve to the exact same
    `PlaneOptions`.
    """
    source_filter = _parity_source_filter()

    with tempfile.NamedTemporaryFile(suffix=".y4m") as out:
        _render_cli_to(out, ["--sigma", str(PARITY_SIGMA_8BIT)])

        cli = source_filter(out.name)
        src = source_filter(str(PARITY_CLIP))
        plugin = core.avd.NL4D(src, sigma=PARITY_SIGMA_NORMALIZED)

        for n, plane, cli_plane, plugin_plane in _parity_frame_diffs(cli, plugin):
            diff = _max_abs_diff(cli_plane, plugin_plane)
            assert diff == 0, f"frame {n} plane {plane} differs by {diff}"


@test
def the_plugin_matches_the_cli_with_automatic_sigma():
    """Both front ends smooth the noise level over the stream, so they must agree."""
    source_filter = _parity_source_filter()

    with tempfile.NamedTemporaryFile(suffix=".y4m") as out:
        _render_cli_to(out, [])

        cli = source_filter(out.name)
        src = source_filter(str(PARITY_CLIP))
        plugin = core.avd.NL4D(src)

        for n, plane, a, b in _parity_frame_diffs(cli, plugin):
            diff = _max_abs_diff(a, b)
            assert diff == 0, f"frame {n} plane {plane} differs by {diff}"


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
