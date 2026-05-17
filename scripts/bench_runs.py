#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Throughput benchmark for av-denoise vs ffmpeg's OpenCL nlmeans.

Each run in the TOML config is executed end-to-end with output discarded
via `-f null -` (so the encoder isn't part of the measurement). Frame
count and ffmpeg-reported fps come from scraping ffmpeg's `-stats` lines;
elapsed time is wall-clock around the subprocess. Reported `fps` is
frames / wall-clock — what the user actually pays.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


STATS_RE = re.compile(r"frame=\s*(\d+)\s+fps=\s*([\d.]+)")

DEFAULT_CONFIG = Path(__file__).parent / "bench_runs.toml"


@dataclass
class Run:
    name: str
    kind: str
    input: Path
    workers: int = 2
    args: list[str] = field(default_factory=list)
    strength: float = 1.2
    patch: int = 9
    search: int = 5


@dataclass
class Result:
    name: str
    kind: str
    frames: int | None
    elapsed: float
    ok: bool


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--only",
        default="",
        help="comma-separated run names to include (default: all)",
    )
    p.add_argument(
        "--warmup",
        action="store_true",
        help="run the first selected run once untimed before the measured pass",
    )
    return p.parse_args()


def load_runs(config_path: Path) -> list[Run]:
    data = tomllib.loads(config_path.read_text())
    default_input = data.get("input")
    if default_input is None and not all("input" in r for r in data.get("runs", [])):
        sys.exit(f"{config_path}: top-level `input` missing and not all runs override it")

    runs: list[Run] = []
    for raw in data.get("runs", []):
        kind = raw.get("kind")
        if kind not in ("av-denoise", "ffmpeg-nlmeans"):
            sys.exit(f"unknown kind in run {raw.get('name')!r}: {kind!r}")
        runs.append(
            Run(
                name=raw["name"],
                kind=kind,
                input=Path(raw.get("input", default_input)),
                workers=raw.get("workers", 2),
                args=list(raw.get("args", [])),
                strength=raw.get("strength", 1.2),
                patch=raw.get("patch", 9),
                search=raw.get("search", 5),
            )
        )
    return runs


def filter_runs(runs: list[Run], only: str) -> list[Run]:
    if not only:
        return runs
    wanted = {n.strip() for n in only.split(",") if n.strip()}
    unknown = wanted - {r.name for r in runs}
    if unknown:
        sys.exit(f"--only references unknown run(s): {sorted(unknown)}")
    return [r for r in runs if r.name in wanted]


def build_av_denoise(run: Run) -> tuple[list[str], list[str]]:
    p1 = [
        "cargo", "run", "--release",
        "--bin", "av-denoise",
        "--features", "binary",
        "--",
        *run.args,
        "file",
        "--workers", str(run.workers),
        "--input", str(run.input),
    ]
    p2 = [
        "ffmpeg",
        "-hide_banner",
        "-stats", "-stats_period", "0.5",
        "-loglevel", "info",
        "-y",
        "-f", "yuv4mpegpipe",
        "-i", "-",
        "-f", "null", "-",
    ]
    return p1, p2


def build_ffmpeg_nlmeans(run: Run) -> list[str]:
    vf = (
        f"hwupload,nlmeans_opencl="
        f"s={run.strength}:p={run.patch}:pc={run.patch}:r={run.search}:rc={run.search},"
        f"hwdownload,format=yuv420p"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-stats", "-stats_period", "0.5",
        "-loglevel", "info",
        "-init_hw_device", "opencl=ocl:0.0",
        "-filter_hw_device", "ocl",
        "-y",
        "-i", str(run.input),
        "-vf", vf,
        "-f", "null", "-",
    ]


class StatsTail:
    """Read ffmpeg stderr in a background thread; tee to our stderr and
    remember the last `frame= … fps= …` match."""

    def __init__(self, stream: Any, label: str) -> None:
        self.stream = stream
        self.label = label
        self.last_frame: int | None = None
        self.last_fps: float | None = None
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._t.start()

    def join(self) -> None:
        self._t.join()

    def _run(self) -> None:
        buf = b""
        while True:
            chunk = self.stream.read(1024)
            if not chunk:
                break
            buf += chunk
            # ffmpeg overwrites stats lines with \r; split on both.
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace")
                sys.stderr.write(f"[{self.label}] {line}\n")
                sys.stderr.flush()
                m = STATS_RE.search(line)
                if m:
                    self.last_frame = int(m.group(1))
                    self.last_fps = float(m.group(2))
        if buf:
            line = buf.decode("utf-8", errors="replace")
            sys.stderr.write(f"[{self.label}] {line}\n")
            m = STATS_RE.search(line)
            if m:
                self.last_frame = int(m.group(1))
                self.last_fps = float(m.group(2))


def run_av_denoise(run: Run) -> Result:
    p1_cmd, p2_cmd = build_av_denoise(run)
    print(f"[{run.name}] $ {shlex.join(p1_cmd)} | {shlex.join(p2_cmd)}", flush=True)

    start = time.monotonic()
    p1 = subprocess.Popen(p1_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p1.stdout is not None
    p2 = subprocess.Popen(p2_cmd, stdin=p1.stdout, stderr=subprocess.PIPE)
    p1.stdout.close()  # let p1 receive SIGPIPE if p2 exits

    p1_tail = StatsTail(p1.stderr, f"{run.name}/av-denoise")
    p2_tail = StatsTail(p2.stderr, f"{run.name}/ffmpeg-sink")
    p1_tail.start()
    p2_tail.start()

    rc2 = p2.wait()
    rc1 = p1.wait()
    elapsed = time.monotonic() - start
    p1_tail.join()
    p2_tail.join()

    ok = rc1 == 0 and rc2 == 0 and p2_tail.last_frame is not None
    return Result(run.name, run.kind, p2_tail.last_frame, elapsed, ok)


def run_ffmpeg_nlmeans(run: Run) -> Result:
    cmd = build_ffmpeg_nlmeans(run)
    print(f"[{run.name}] $ {shlex.join(cmd)}", flush=True)

    start = time.monotonic()
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    tail = StatsTail(proc.stderr, f"{run.name}/ffmpeg")
    tail.start()
    rc = proc.wait()
    elapsed = time.monotonic() - start
    tail.join()

    ok = rc == 0 and tail.last_frame is not None
    return Result(run.name, run.kind, tail.last_frame, elapsed, ok)


def execute(run: Run) -> Result:
    if run.kind == "av-denoise":
        return run_av_denoise(run)
    return run_ffmpeg_nlmeans(run)


def print_table(results: list[Result]) -> None:
    if not results:
        return

    name_w = max(len("name"), max(len(r.name) for r in results))
    kind_w = max(len("kind"), max(len(r.kind) for r in results))
    frames_w = max(len("frames"), 8)
    elapsed_w = max(len("elapsed_s"), 10)
    fps_w = max(len("fps"), 7)

    header = (
        f"{'name':<{name_w}}  {'kind':<{kind_w}}  "
        f"{'frames':>{frames_w}}  {'elapsed_s':>{elapsed_w}}  {'fps':>{fps_w}}"
    )
    sep = (
        f"{'─' * name_w}  {'─' * kind_w}  "
        f"{'─' * frames_w}  {'─' * elapsed_w}  {'─' * fps_w}"
    )
    print()
    print(header)
    print(sep)
    for r in results:
        if r.ok and r.frames is not None:
            frames = f"{r.frames}"
            fps = f"{r.frames / r.elapsed:.2f}" if r.elapsed > 0 else "—"
        else:
            frames = "FAILED"
            fps = "—"
        print(
            f"{r.name:<{name_w}}  {r.kind:<{kind_w}}  "
            f"{frames:>{frames_w}}  {r.elapsed:>{elapsed_w}.2f}  {fps:>{fps_w}}"
        )


def main() -> None:
    args = parse_cli()
    runs = filter_runs(load_runs(args.config), args.only)
    if not runs:
        sys.exit("no runs selected")

    if args.warmup:
        print(f"[warmup] {runs[0].name} (untimed)", flush=True)
        execute(runs[0])

    results: list[Result] = []
    for run in runs:
        try:
            results.append(execute(run))
        except Exception as e:  # noqa: BLE001
            print(f"[{run.name}] error: {e}", file=sys.stderr, flush=True)
            results.append(Result(run.name, run.kind, None, 0.0, False))

    print_table(results)


if __name__ == "__main__":
    main()
