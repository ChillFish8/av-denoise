#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""XPSNR and SSIM quality benchmark for av-denoise variants.

Each run denoises a synthetically-noised copy of a clean source clip
and is scored against the clean original with ffmpeg's `xpsnr` and
`ssim` filters in a single pass. Noisy clips are generated once per
noise level with ffmpeg's `noise` filter (temporal noise, fixed seed)
and cached in the work directory. Denoised frames are piped straight
into the scoring ffmpeg, so no intermediate denoised files are
written. A `noisy` run kind scores the corrupted clip itself, making
each variant's recovery visible. SSIM scores each plane independently,
so its chroma values do not carry XPSNR's luma-masking cross-talk,
making it the cleaner chroma-comparison signal.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path


XPSNR_RE = re.compile(
    r"XPSNR\s+y:\s*(inf|[\d.]+)\s+u:\s*(inf|[\d.]+)\s+v:\s*(inf|[\d.]+)"
    r"\s+\(minimum:\s*(inf|[\d.]+)\)"
)

SSIM_RE = re.compile(
    r"SSIM\s+Y:([\d.]+)\s+\([^)]*\)\s+U:([\d.]+)\s+\([^)]*\)"
    r"\s+V:([\d.]+)\s+\([^)]*\)\s+All:([\d.]+)\s+\([^)]*\)"
)

# ffmpeg's framesync (the machinery behind dual-input filters like xpsnr
# and ssim) warns rather than fails when the two inputs disagree on
# timebase or drop/duplicate frames while pairing them up. A passing exit
# code and a normal-looking number are not proof the two streams were
# paired frame-for-frame, so these patterns must be swept for explicitly.
TIMEBASE_MISMATCH_RE = re.compile(
    r"not matching timebases found|timebase \(\d+/\d+\) (?:do|does) not match",
    re.IGNORECASE,
)
FRAME_SYNC_DROP_RE = re.compile(r"(\d+) frames? dropped, (\d+) frames? duplicated")

DEFAULT_CONFIG = Path(__file__).parent / "quality_runs.toml"

# Reference-implementation group sizes for the `ffmpeg-bm3d` row's two
# stages. ffmpeg defaults `group` to 1, which disables block matching
# entirely and is not BM3D's collaborative filtering. Used both to build
# the filter graph and to label the results table, so the two cannot
# drift apart.
BM3D_GROUP_BASIC = 16
BM3D_GROUP_FINAL = 32


@dataclass
class Run:
    name: str
    kind: str
    workers: int = 2
    args: list[str] = field(default_factory=list)
    strength: float | str = 1.2
    patch: int = 9
    search: int = 5
    sigma: float | str = 15.0


@dataclass
class Config:
    input: Path
    workdir: Path
    frames: int
    seed: int
    noise: list[int]
    strengths: dict[int, float]
    bm3d_sigmas: dict[int, float]
    runs: list[Run]


@dataclass
class Result:
    name: str
    alls: int
    y: float | None = None
    u: float | None = None
    v: float | None = None
    minimum: float | None = None
    ssim_y: float | None = None
    ssim_all: float | None = None
    ok: bool = False


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--only",
        default="",
        help="comma-separated run names to include (default: all)",
    )
    p.add_argument(
        "--device",
        default="",
        help="av-denoise --device value appended to every av-denoise run",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="regenerate the cached reference/noisy clips",
    )
    return p.parse_args()


def load_config(config_path: Path) -> Config:
    data = tomllib.loads(config_path.read_text())
    for key in ("input", "noise"):
        if key not in data:
            sys.exit(f"{config_path}: top-level `{key}` missing")

    runs: list[Run] = []
    for raw in data.get("runs", []):
        kind = raw.get("kind")
        if kind not in ("noisy", "av-denoise", "ffmpeg-nlmeans", "ffmpeg-bm3d"):
            sys.exit(f"unknown kind in run {raw.get('name')!r}: {kind!r}")
        runs.append(
            Run(
                name=raw["name"],
                kind=kind,
                workers=raw.get("workers", 2),
                args=list(raw.get("args", [])),
                strength=raw.get("strength", 1.2),
                patch=raw.get("patch", 9),
                search=raw.get("search", 5),
                sigma=raw.get("sigma", 15.0),
            )
        )
    return Config(
        input=Path(data["input"]),
        workdir=Path(data.get("workdir", "./data/quality_runs")),
        frames=data.get("frames", 240),
        seed=data.get("seed", 4242),
        noise=list(data["noise"]),
        strengths={int(k): float(v) for k, v in data.get("strengths", {}).items()},
        bm3d_sigmas={int(k): float(v) for k, v in data.get("bm3d_sigmas", {}).items()},
        runs=runs,
    )


def filter_runs(runs: list[Run], only: str) -> list[Run]:
    if not only:
        return runs
    wanted = {n.strip() for n in only.split(",") if n.strip()}
    unknown = wanted - {r.name for r in runs}
    if unknown:
        sys.exit(f"--only references unknown run(s): {sorted(unknown)}")
    return [r for r in runs if r.name in wanted]


def check_xpsnr() -> None:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=False,
    )
    if " xpsnr " not in proc.stdout:
        sys.exit("this ffmpeg build lacks the `xpsnr` filter (FFmpeg 7.1+ required)")


def generate_clip(cmd: list[str], target: Path, force: bool) -> Path:
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[generate] $ {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return target


def prepare_reference(cfg: Config, force: bool) -> Path:
    target = cfg.workdir / f"ref_f{cfg.frames}.mkv"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(cfg.input),
        "-frames:v", str(cfg.frames),
        "-c:v", "ffv1",
        str(target),
    ]
    return generate_clip(cmd, target, force)


def prepare_noisy(cfg: Config, alls: int, force: bool) -> Path:
    target = cfg.workdir / f"noisy_a{alls}_f{cfg.frames}_s{cfg.seed}.mkv"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(cfg.input),
        "-frames:v", str(cfg.frames),
        "-vf", f"noise=alls={alls}:allf=t:all_seed={cfg.seed}",
        "-c:v", "ffv1",
        str(target),
    ]
    return generate_clip(cmd, target, force)


def parse_xpsnr(stderr: str) -> tuple[float, float, float, float] | None:
    match = None
    for match in XPSNR_RE.finditer(stderr):
        pass
    if match is None:
        return None
    y, u, v, minimum = (float(g) for g in match.groups())
    return y, u, v, minimum


def parse_ssim(stderr: str) -> tuple[float, float, float, float] | None:
    match = None
    for match in SSIM_RE.finditer(stderr):
        pass
    if match is None:
        return None
    y, u, v, all_planes = (float(g) for g in match.groups())
    return y, u, v, all_planes


def find_scoring_corruption(stderr: str) -> str | None:
    """Look for signs that framesync silently mispaired frames.

    A mismatched timebase or a nonzero dropped/duplicated frame count
    means the XPSNR/SSIM numbers ffmpeg printed do not describe the two
    inputs compared frame-for-frame. ffmpeg treats this as a warning, not
    an error, so it must be caught here instead of trusted by exit code.
    Returns a human-readable reason, or None if the stderr looks clean.
    """
    if TIMEBASE_MISMATCH_RE.search(stderr):
        return "ffmpeg reported mismatched timebases between the scoring inputs"
    drop_match = FRAME_SYNC_DROP_RE.search(stderr)
    if drop_match and (int(drop_match.group(1)) or int(drop_match.group(2))):
        return "ffmpeg reported dropped or duplicated frames while pairing the scoring inputs"
    return None


def score_noisy(noisy: Path, ref: Path) -> tuple[bool, str]:
    graph = (
        "[0:v]split=2[n1][n2];[1:v]split=2[r1][r2];"
        "[n1][r1]xpsnr;[n2][r2]ssim"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(noisy),
        "-i", str(ref),
        "-lavfi", graph,
        "-f", "null", "-",
    ]
    print(f"$ {shlex.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode == 0, proc.stderr


def score_av_denoise(run: Run, noisy: Path, ref: Path, device: str) -> tuple[bool, str]:
    device_args = ["--device", device] if device else []
    p1_cmd = [
        "cargo", "run", "--release",
        "--bin", "av-denoise",
        "--features", "binary",
        "--",
        *run.args,
        *device_args,
        "file",
        "--workers", str(run.workers),
        "--input", str(noisy),
    ]
    # Input 0 is the av-denoise y4m pipe, exact timebase 1001/24000. Input
    # 1 is the reference file, whose Matroska container stores PTS
    # quantized to 1/1000 (millisecond) ticks. That quantization drifts against
    # the pipe's exact PTS as frame count grows, so xpsnr/ssim's internal
    # framesync (which pairs frames by nearest real time) walks out of
    # sync and silently scores the wrong frame pairs. ffmpeg only warns
    # about this, it does not fail the run. A plain `settb=AVTB` on both
    # branches does not fix it, since it only relabels the units and does
    # not undo drift already baked into the reference's stored PTS values.
    # `settb=1,setpts=N` does fix it. It puts both branches on the same
    # unit-second timebase and rewrites each frame's PTS to its own
    # sequential frame index, so frame k of one branch always lands at the
    # same real time as frame k of the other, which is what actually
    # matters here since both streams are the same fixed frame count by
    # construction.
    graph = (
        "[0:v]settb=1,setpts=N,split=2[d1][d2];"
        "[1:v]settb=1,setpts=N,split=2[r1][r2];"
        "[d1][r1]xpsnr;[d2][r2]ssim"
    )
    p2_cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-f", "yuv4mpegpipe", "-i", "-",
        "-i", str(ref),
        "-lavfi", graph,
        "-f", "null", "-",
    ]
    print(f"$ {shlex.join(p1_cmd)} | {shlex.join(p2_cmd)}", flush=True)

    p1 = subprocess.Popen(p1_cmd, stdout=subprocess.PIPE)
    assert p1.stdout is not None
    p2 = subprocess.Popen(p2_cmd, stdin=p1.stdout, stderr=subprocess.PIPE, text=True)
    p1.stdout.close()  # let p1 receive SIGPIPE if p2 exits

    _, stderr = p2.communicate()
    rc1 = p1.wait()
    return rc1 == 0 and p2.returncode == 0, stderr


def score_ffmpeg_nlmeans(run: Run, noisy: Path, ref: Path) -> tuple[bool, str]:
    graph = (
        f"[0:v]hwupload,nlmeans_opencl="
        f"s={run.strength}:p={run.patch}:pc={run.patch}:r={run.search}:rc={run.search},"
        f"hwdownload,format=yuv420p[den];"
        f"[den]split=2[d1][d2];[1:v]split=2[r1][r2];[d1][r1]xpsnr;[d2][r2]ssim"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-init_hw_device", "opencl=ocl:0.0",
        "-filter_hw_device", "ocl",
        "-i", str(noisy),
        "-i", str(ref),
        "-lavfi", graph,
        "-f", "null", "-",
    ]
    print(f"$ {shlex.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode == 0, proc.stderr


def score_ffmpeg_bm3d(run: Run, noisy: Path, ref: Path) -> tuple[bool, str]:
    # ffmpeg's bm3d defaults `group` (the max number of blocks collected
    # into one 3D group) to 1, which skips block matching entirely. That
    # is not collaborative filtering, the step that makes BM3D BM3D, so
    # every `ffmpeg_bm3d` row from before this file set `group` explicitly
    # is a weaker filter than BM3D and is not comparable to later runs.
    # The reference implementation groups up to 16 blocks in the
    # hard-thresholding stage and up to 32 in the Wiener stage, so both
    # stages run here with matching group sizes and the basic estimate
    # feeds the final estimate as its reference, same as bm3d_parallel.py.
    graph = (
        f"[0:v]split[a][b];"
        f"[b]bm3d=sigma={run.sigma}:group={BM3D_GROUP_BASIC}:estim=basic[basic];"
        f"[a][basic]bm3d=sigma={run.sigma}:group={BM3D_GROUP_FINAL}:estim=final:ref=1[den];"
        f"[den]split=2[d1][d2];[1:v]split=2[r1][r2];[d1][r1]xpsnr;[d2][r2]ssim"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(noisy),
        "-i", str(ref),
        "-lavfi", graph,
        "-f", "null", "-",
    ]
    print(f"$ {shlex.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode == 0, proc.stderr


def resolve_run(
    run: Run, alls: int, strengths: dict[int, float], bm3d_sigmas: dict[int, float]
) -> Run:
    """Substitute `$strength` placeholders with the `[strengths]` entry
    and `$bm3d_sigma` placeholders with the `[bm3d_sigmas]` entry for
    this noise level."""

    def lookup_strength() -> float:
        if alls not in strengths:
            sys.exit(
                f"run {run.name!r} uses $strength but [strengths] has no entry for noise {alls}"
            )
        return strengths[alls]

    def lookup_sigma() -> float:
        if alls not in bm3d_sigmas:
            sys.exit(
                f"run {run.name!r} uses $bm3d_sigma but [bm3d_sigmas] has no entry for noise {alls}"
            )
        return bm3d_sigmas[alls]

    def substitute(a: str) -> str:
        if a == "$strength":
            return str(lookup_strength())
        if a == "$bm3d_sigma":
            return str(lookup_sigma())
        return a

    args = [substitute(a) for a in run.args]
    strength = lookup_strength() if run.strength == "$strength" else run.strength
    sigma = lookup_sigma() if run.sigma == "$bm3d_sigma" else run.sigma
    return replace(run, args=args, strength=strength, sigma=sigma)


def execute(
    run: Run,
    noisy: Path,
    ref: Path,
    alls: int,
    strengths: dict[int, float],
    bm3d_sigmas: dict[int, float],
    device: str,
) -> Result:
    run = resolve_run(run, alls, strengths, bm3d_sigmas)
    if run.kind == "noisy":
        ok, stderr = score_noisy(noisy, ref)
    elif run.kind == "av-denoise":
        ok, stderr = score_av_denoise(run, noisy, ref, device)
    elif run.kind == "ffmpeg-nlmeans":
        ok, stderr = score_ffmpeg_nlmeans(run, noisy, ref)
    else:
        ok, stderr = score_ffmpeg_bm3d(run, noisy, ref)

    corruption = find_scoring_corruption(stderr)
    if corruption:
        sys.stderr.write(stderr)
        print(f"[{run.name} @ noise {alls}] FAILED: {corruption}", file=sys.stderr, flush=True)
        return Result(run.name, alls)

    xpsnr = parse_xpsnr(stderr)
    ssim = parse_ssim(stderr)
    if not ok or xpsnr is None or ssim is None:
        sys.stderr.write(stderr)
        return Result(run.name, alls)
    y, u, v, minimum = xpsnr
    ssim_y, _ssim_u, _ssim_v, ssim_all = ssim
    return Result(run.name, alls, y, u, v, minimum, ssim_y, ssim_all, ok=True)


def print_table(results: list[Result], bm3d_included: bool) -> None:
    if not results:
        return

    if bm3d_included:
        print(
            f"# ffmpeg_bm3d rows below: two-stage bm3d, "
            f"group={BM3D_GROUP_BASIC} (estim=basic) / group={BM3D_GROUP_FINAL} "
            f"(estim=final:ref=1). Rows produced before this line existed used "
            f"group=1, single-stage estim=basic, and are not comparable.",
            flush=True,
        )

    def fmt(value: float | None, precision: int = 2) -> str:
        if value is None:
            return "—"
        if value == float("inf"):
            return "inf"
        return f"{value:.{precision}f}"

    name_w = max(len("name"), max(len(r.name) for r in results))
    noise_w = max(len("noise"), 5)
    col_w = 8

    header = (
        f"{'name':<{name_w}}  {'noise':>{noise_w}}  "
        f"{'xpsnr_y':>{col_w}}  {'xpsnr_u':>{col_w}}  {'xpsnr_v':>{col_w}}  {'min':>{col_w}}  "
        f"{'ssim_y':>{col_w}}  {'ssim_all':>{col_w}}"
    )
    sep = (
        f"{'─' * name_w}  {'─' * noise_w}  "
        f"{'─' * col_w}  {'─' * col_w}  {'─' * col_w}  {'─' * col_w}  "
        f"{'─' * col_w}  {'─' * col_w}"
    )
    print()
    print(header)
    print(sep)
    for r in results:
        if r.ok:
            y, u, v, mn = fmt(r.y), fmt(r.u), fmt(r.v), fmt(r.minimum)
            sy, sa = fmt(r.ssim_y, 4), fmt(r.ssim_all, 4)
        else:
            y, u, v, mn = "FAILED", "—", "—", "—"
            sy, sa = "—", "—"
        print(
            f"{r.name:<{name_w}}  {r.alls:>{noise_w}}  "
            f"{y:>{col_w}}  {u:>{col_w}}  {v:>{col_w}}  {mn:>{col_w}}  "
            f"{sy:>{col_w}}  {sa:>{col_w}}"
        )


def main() -> None:
    args = parse_cli()
    cfg = load_config(args.config)
    runs = filter_runs(cfg.runs, args.only)
    if not runs:
        sys.exit("no runs selected")
    check_xpsnr()

    ref = prepare_reference(cfg, args.force)
    noisy_clips = {alls: prepare_noisy(cfg, alls, args.force) for alls in cfg.noise}

    results: list[Result] = []
    for alls in cfg.noise:
        for run in runs:
            print(f"[{run.name} @ noise {alls}]", flush=True)
            try:
                results.append(
                    execute(
                        run,
                        noisy_clips[alls],
                        ref,
                        alls,
                        cfg.strengths,
                        cfg.bm3d_sigmas,
                        args.device,
                    )
                )
            except Exception as e:  # noqa: BLE001
                print(f"[{run.name}] error: {e}", file=sys.stderr, flush=True)
                results.append(Result(run.name, alls))

    print_table(results, any(r.kind == "ffmpeg-bm3d" for r in runs))


if __name__ == "__main__":
    main()
