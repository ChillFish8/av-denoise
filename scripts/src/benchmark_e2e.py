from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from pathlib import Path
from typing import IO

from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "scripts" / "configs" / "benchmark_e2e.toml"

PLACEHOLDER_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")
FRAME_RE = re.compile(r"frame=\s*(\d+)")
KNOWN_PLACEHOLDERS = frozenset({"input", "output"})


def from_repo_root(path: Path) -> Path:
    """Resolves a config path the way the commands see it.

    Variants run with the repo root as their working directory, so a
    relative path in the config means a path under the repo root, not one
    under wherever the script was invoked from.
    """

    return path if path.is_absolute() else REPO_ROOT / path


class ConfigError(Exception):
    """A config the runner cannot execute."""


class Variant(BaseModel):
    """One command to measure."""

    model_config = ConfigDict(extra="forbid")

    name: str
    command: str
    input: Path | None = None
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_placeholders(self) -> Variant:
        unknown = set(PLACEHOLDER_RE.findall(self.command)) - KNOWN_PLACEHOLDERS
        if unknown:
            raise ValueError(
                f"variant {self.name!r} uses unknown placeholder(s) "
                f"{sorted(unknown)}, only {sorted(KNOWN_PLACEHOLDERS)} are substituted"
            )
        return self


class Group(BaseModel):
    """A set of variants measured against the same input."""

    model_config = ConfigDict(extra="forbid")

    name: str
    input: Path | None = None
    repeats: int | None = Field(default=None, ge=1)
    variants: list[Variant] = Field(min_length=1)

    @model_validator(mode="after")
    def check_variant_names(self) -> Group:
        seen = set()
        for variant in self.variants:
            if variant.name in seen:
                raise ValueError(f"group {self.name!r} has two variants named {variant.name!r}")
            seen.add(variant.name)
        return self


class Config(BaseModel):
    """The whole benchmark."""

    model_config = ConfigDict(extra="forbid")

    input: Path | None = None
    output_dir: Path | None = None
    repeats: int = Field(default=1, ge=1)
    warmup: bool = False
    groups: list[Group] = Field(min_length=1)

    @model_validator(mode="after")
    def check_groups(self) -> Config:
        seen = set()
        for group in self.groups:
            if group.name in seen:
                raise ValueError(f"two groups are named {group.name!r}")
            seen.add(group.name)
            if group.input is None and self.input is None:
                raise ValueError(f"group {group.name!r} has no `input` and there is no top-level `input`")
            for variant in group.variants:
                if variant.input is None and group.input is None and self.input is None:
                    raise ValueError(f"variant {variant.name!r} has no input to run against")
        return self

    def repeats_for(self, group: Group) -> int:
        return group.repeats if group.repeats is not None else self.repeats

    def input_for(self, group: Group, variant: Variant) -> Path:
        path = variant.input or group.input or self.input
        assert path is not None, "validated in check_groups"
        return from_repo_root(path)


class Measurement(BaseModel):
    """What one group of repeats produced for one variant."""

    model_config = ConfigDict(extra="forbid")

    group: str
    name: str
    frames: int | None
    elapsed: list[float]
    ok: bool

    @property
    def median_elapsed(self) -> float:
        return statistics.median(self.elapsed) if self.elapsed else 0.0

    def fps(self, elapsed: float) -> float | None:
        if self.frames is None or elapsed <= 0.0:
            return None
        return self.frames / elapsed

    @property
    def median_fps(self) -> float | None:
        return self.fps(self.median_elapsed)

    @property
    def fps_range(self) -> tuple[float, float] | None:
        rates = [f for f in (self.fps(e) for e in self.elapsed) if f is not None]
        if not rates:
            return None
        return min(rates), max(rates)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML config to run")
    parser.add_argument(
        "--group",
        default="",
        help="comma-separated group names to include (default: all)",
    )
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated variant names to include (default: all)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="override the timed pass count for every group",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override where `{{output}}` files are written",
    )
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument(
        "--warmup",
        dest="warmup",
        action="store_true",
        default=None,
        help="run one untimed pass per variant first",
    )
    warmup.add_argument("--no-warmup", dest="warmup", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the substituted commands and exit without running them",
    )
    return parser.parse_args()


def load_config(path: Path) -> Config:
    try:
        raw = tomllib.loads(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"{path}: no such config") from None
    except tomllib.TOMLDecodeError as err:
        raise ConfigError(f"{path}: {err}") from None
    return Config.model_validate(raw)


def select(config: Config, groups: str, variants: str) -> Config:
    """Narrows the config to the `--group` and `--only` selections."""

    wanted_groups = {n.strip() for n in groups.split(",") if n.strip()}
    wanted_variants = {n.strip() for n in variants.split(",") if n.strip()}

    unknown = wanted_groups - {g.name for g in config.groups}
    if unknown:
        raise ConfigError(f"--group names no such group(s): {sorted(unknown)}")
    all_variants = {v.name for g in config.groups for v in g.variants}
    unknown = wanted_variants - all_variants
    if unknown:
        raise ConfigError(f"--only names no such variant(s): {sorted(unknown)}")

    kept = []
    for group in config.groups:
        if wanted_groups and group.name not in wanted_groups:
            continue
        members = [v for v in group.variants if not wanted_variants or v.name in wanted_variants]
        if members:
            kept.append(group.model_copy(update={"variants": members}))
    if not kept:
        raise ConfigError("the --group and --only selections leave nothing to run")
    return config.model_copy(update={"groups": kept})


def count_frames(path: Path, cache: dict[Path, int | None]) -> int | None:
    """Frame count of a clip, from the container header where it has one
    and from a packet count where it does not.

    This is the fallback for a variant whose command reports no progress
    of its own. Counting packets avoids decoding the clip, which matters
    for the 4K samples.
    """

    if path in cache:
        return cache[path]

    frames = _ffprobe(path, "nb_frames") or _ffprobe(path, "nb_read_packets", count_packets=True)
    if frames is None:
        print(f"[warn] cannot count frames in {path}, fps will be blank", file=sys.stderr)
    cache[path] = frames
    return frames


def _ffprobe(path: Path, field: str, count_packets: bool = False) -> int | None:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count_packets:
        cmd.append("-count_packets")
    cmd += ["-show_entries", f"stream={field}", "-of", "default=nokey=1:noprint_wrappers=1", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return int(out) if out.isdigit() and int(out) > 0 else None


def render(command: str, input_path: Path, output_path: Path) -> str:
    """Substitutes the placeholders a variant's command uses."""

    values = {"input": str(input_path), "output": str(output_path)}
    return PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], command)


class StderrTail:
    """Reads a child's stderr on a thread and tees it to ours with a
    label. ffmpeg redraws its stats line with `\\r`, so both line
    terminators break a line."""

    def __init__(self, stream: IO[bytes], label: str) -> None:
        self.stream = stream
        self.label = label
        self.last_frames: int | None = None  # highest `frame=` seen
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join()

    def _run(self) -> None:
        buf = b""
        while chunk := self.stream.read(1024):
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                if raw:
                    self._emit(raw)
        if buf:
            self._emit(buf)

    def _emit(self, raw: bytes) -> None:
        line = raw.decode("utf-8", errors="replace")
        if match := FRAME_RE.search(line):
            frames = int(match.group(1))
            if self.last_frames is None or frames > self.last_frames:
                self.last_frames = frames
        sys.stderr.write(f"[{self.label}] {line}\n")
        sys.stderr.flush()


def run_once(command: str, label: str, env: dict[str, str]) -> tuple[bool, float, int | None]:
    """Runs one pass and reports whether it succeeded, how long the wall
    clock says it took, and the last frame number it printed."""

    child_env = {**os.environ, **env}
    start = time.monotonic()
    proc = subprocess.Popen(
        ["bash", "-o", "pipefail", "-c", command],
        cwd=REPO_ROOT,
        env=child_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stderr is not None
    tail = StderrTail(proc.stderr, label)
    tail.start()
    code = proc.wait()
    elapsed = time.monotonic() - start
    tail.join()
    if code != 0:
        print(f"[{label}] exited with {code}", file=sys.stderr, flush=True)
    return code == 0, elapsed, tail.last_frames


def measure(config: Config, output_root: Path, dry_run: bool) -> list[Measurement]:
    frame_cache: dict[Path, int | None] = {}
    results: list[Measurement] = []

    for group in config.groups:
        repeats = config.repeats_for(group)
        group_dir = output_root / group.name
        for variant in group.variants:
            input_path = config.input_for(group, variant)
            command = render(variant.command, input_path, group_dir / f"{variant.name}.mkv")
            label = f"{group.name}/{variant.name}"

            if dry_run:
                print(f"[{label}] repeats={repeats}\n{command}\n")
                continue

            if not input_path.exists():
                print(f"[{label}] input {input_path} does not exist, skipping", file=sys.stderr)
                results.append(
                    Measurement(group=group.name, name=variant.name, frames=None, elapsed=[], ok=False)
                )
                continue

            group_dir.mkdir(parents=True, exist_ok=True)
            frames: int | None = None

            if config.warmup:
                print(f"[{label}] warmup pass (untimed)", flush=True)
                run_once(command, label, variant.env)

            elapsed: list[float] = []
            ok = True
            for pass_index in range(1, repeats + 1):
                print(f"[{label}] pass {pass_index}/{repeats}: {command}", flush=True)
                passed, seconds, reported = run_once(command, label, variant.env)
                if not passed:
                    ok = False
                    break
                elapsed.append(seconds)
                frames = reported if reported is not None else frames
            if ok and frames is None:
                frames = count_frames(input_path, frame_cache)
            results.append(
                Measurement(group=group.name, name=variant.name, frames=frames, elapsed=elapsed, ok=ok)
            )

    return results


def print_report(results: list[Measurement]) -> None:
    """Prints one table per group, each variant's fps relative to the
    first variant in its group that finished."""

    headers = ("variant", "frames", "median_s", "fps", "fps_range", "rel")

    for group in dict.fromkeys(r.group for r in results):
        rows = [r for r in results if r.group == group]
        baseline = next((r.median_fps for r in rows if r.ok and r.median_fps), None)

        cells = []
        for row in rows:
            if not row.ok:
                cells.append((row.name, "FAILED", "—", "—", "—", "—"))
                continue
            fps, span = row.median_fps, row.fps_range
            cells.append(
                (
                    row.name,
                    str(row.frames) if row.frames is not None else "—",
                    f"{row.median_elapsed:.2f}",
                    f"{fps:.3f}" if fps else "—",
                    f"{span[0]:.3f}-{span[1]:.3f}" if span else "—",
                    f"{fps / baseline:.2f}x" if fps and baseline else "—",
                )
            )

        widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(headers)]
        header = "  ".join(
            h.ljust(w) if i == 0 else h.rjust(w) for i, (h, w) in enumerate(zip(headers, widths))
        )
        print(f"\n{group}")
        print(header)
        print("─" * len(header))
        for cell in cells:
            print(
                "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cell, widths)))
            )


def main() -> None:
    args = parse_cli()
    try:
        config = select(load_config(args.config), args.group, args.only)
    except ConfigError as err:
        sys.exit(str(err))
    except ValueError as err:
        sys.exit(f"{args.config}: {err}")

    if args.repeats is not None:
        config = config.model_copy(update={"repeats": args.repeats})
        config.groups = [g.model_copy(update={"repeats": None}) for g in config.groups]
    if args.warmup is not None:
        config = config.model_copy(update={"warmup": args.warmup})

    output_dir = args.output_dir or config.output_dir
    scratch = Path(tempfile.mkdtemp(prefix="benchmark-e2e-")) if output_dir is None else None
    output_root = scratch if scratch is not None else from_repo_root(Path(output_dir))

    try:
        results = measure(config, output_root, args.dry_run)
        if not args.dry_run:
            print_report(results)
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)

    if any(not r.ok for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
