#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""XPSNR quality benchmark for av-denoise variants.

Each run denoises a synthetically-noised copy of a clean source clip
and is scored against the clean original with ffmpeg's `xpsnr` filter.
Noisy clips are generated once per noise level with ffmpeg's `noise`
filter (temporal noise, fixed seed) and cached in the work directory.
Denoised frames are piped straight into the scoring ffmpeg, so no
intermediate denoised files are written. A `noisy` run kind scores the
corrupted clip itself, making each variant's recovery visible.
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

DEFAULT_CONFIG = Path(__file__).parent / "quality_runs.toml"


@dataclass
class Run:
    name: str
    kind: str
    workers: int = 2
    args: list[str] = field(default_factory=list)
    strength: float | str = 1.2
    patch: int = 9
    search: int = 5


@dataclass
class Config:
    input: Path
    workdir: Path
    frames: int
    seed: int
    noise: list[int]
    strengths: dict[int, float]
    runs: list[Run]


@dataclass
class Result:
    name: str
    alls: int
    y: float | None = None
    u: float | None = None
    v: float | None = None
    minimum: float | None = None
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
        if kind not in ("noisy", "av-denoise", "ffmpeg-nlmeans"):
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
            )
        )
    return Config(
        input=Path(data["input"]),
        workdir=Path(data.get("workdir", "./data/quality_runs")),
        frames=data.get("frames", 240),
        seed=data.get("seed", 4242),
        noise=list(data["noise"]),
        strengths={int(k): float(v) for k, v in data.get("strengths", {}).items()},
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


def score_noisy(noisy: Path, ref: Path) -> tuple[bool, str]:
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(noisy),
        "-i", str(ref),
        "-lavfi", "xpsnr",
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
    p2_cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-f", "yuv4mpegpipe", "-i", "-",
        "-i", str(ref),
        "-lavfi", "[0:v][1:v]xpsnr",
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
        f"hwdownload,format=yuv420p[den];[den][1:v]xpsnr"
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


def resolve_run(run: Run, alls: int, strengths: dict[int, float]) -> Run:
    """Substitute `$strength` placeholders with the `[strengths]` entry
    for this noise level."""

    def lookup() -> float:
        if alls not in strengths:
            sys.exit(
                f"run {run.name!r} uses $strength but [strengths] has no entry for noise {alls}"
            )
        return strengths[alls]

    args = [str(lookup()) if a == "$strength" else a for a in run.args]
    strength = lookup() if run.strength == "$strength" else run.strength
    return replace(run, args=args, strength=strength)


def execute(
    run: Run, noisy: Path, ref: Path, alls: int, strengths: dict[int, float], device: str
) -> Result:
    run = resolve_run(run, alls, strengths)
    if run.kind == "noisy":
        ok, stderr = score_noisy(noisy, ref)
    elif run.kind == "av-denoise":
        ok, stderr = score_av_denoise(run, noisy, ref, device)
    else:
        ok, stderr = score_ffmpeg_nlmeans(run, noisy, ref)

    parsed = parse_xpsnr(stderr)
    if not ok or parsed is None:
        sys.stderr.write(stderr)
        return Result(run.name, alls)
    y, u, v, minimum = parsed
    return Result(run.name, alls, y, u, v, minimum, ok=True)


def print_table(results: list[Result]) -> None:
    if not results:
        return

    def fmt(value: float | None) -> str:
        if value is None:
            return "—"
        if value == float("inf"):
            return "inf"
        return f"{value:.2f}"

    name_w = max(len("name"), max(len(r.name) for r in results))
    noise_w = max(len("noise"), 5)
    col_w = 8

    header = (
        f"{'name':<{name_w}}  {'noise':>{noise_w}}  "
        f"{'xpsnr_y':>{col_w}}  {'xpsnr_u':>{col_w}}  {'xpsnr_v':>{col_w}}  {'min':>{col_w}}"
    )
    sep = (
        f"{'─' * name_w}  {'─' * noise_w}  "
        f"{'─' * col_w}  {'─' * col_w}  {'─' * col_w}  {'─' * col_w}"
    )
    print()
    print(header)
    print(sep)
    for r in results:
        if r.ok:
            y, u, v, mn = fmt(r.y), fmt(r.u), fmt(r.v), fmt(r.minimum)
        else:
            y, u, v, mn = "FAILED", "—", "—", "—"
        print(
            f"{r.name:<{name_w}}  {r.alls:>{noise_w}}  "
            f"{y:>{col_w}}  {u:>{col_w}}  {v:>{col_w}}  {mn:>{col_w}}"
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
                    execute(run, noisy_clips[alls], ref, alls, cfg.strengths, args.device)
                )
            except Exception as e:  # noqa: BLE001
                print(f"[{run.name}] error: {e}", file=sys.stderr, flush=True)
                results.append(Result(run.name, alls))

    print_table(results)


if __name__ == "__main__":
    main()
