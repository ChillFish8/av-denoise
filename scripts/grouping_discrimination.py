#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Measures how well the collaborative filter's spatial grouping
discriminates real matches from noise-fooled ones.

Extracts a clean frame from `data/clean-1080p.mkv` and a synthetically
noised copy of the same frame, using ffmpeg's `noise` filter the same
way `scripts/quality_runs.py` does (`allf=t`, a fixed seed), then hands
both raw frames to `grouping_diag`, a small Rust binary that runs the
real `collab_group_spatial` / `collab_filter_ht` / `collab_aggregate`
kernels and scores what each stage's grouping admitted against ground
truth taken from the clean frame. See `src/bin/grouping_diag.rs`'s own
doc comment for the method and what it does and does not reproduce from
the real nl3d cascade.

Run:

    uv run scripts/grouping_discrimination.py

Add `--force` to regenerate the cached extracted clips and raw frames.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "data" / "clean-1080p.mkv"
WORKDIR = REPO_ROOT / "data" / "grouping_diag"
BINARY = REPO_ROOT / "target" / "release" / "grouping_diag"

WIDTH, HEIGHT = 1920, 1080
START_SECONDS = 4.166
FRAMES = 2
ALLS = 16  # matches scripts/quality_runs.toml's middle noise level
SEED = 4242  # matches scripts/quality_runs.toml's seed


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def build_binary() -> None:
    run(["cargo", "build", "--release", "--bin", "grouping_diag", "--features", "vulkan"])


def extract_clean_clip(target: Path, force: bool) -> None:
    if target.exists() and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(START_SECONDS), "-i", str(SOURCE),
            "-vframes", str(FRAMES),
            "-c:v", "ffv1",
            str(target),
        ]
    )


def add_noise(clean: Path, target: Path, force: bool) -> None:
    if target.exists() and not force:
        return
    run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(clean),
            "-vf", f"noise=alls={ALLS}:allf=t:all_seed={SEED}",
            "-c:v", "ffv1",
            str(target),
        ]
    )


def extract_raw_gray(clip: Path, target: Path, force: bool) -> None:
    if target.exists() and not force:
        return
    run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(clip),
            "-pix_fmt", "gray",
            "-f", "rawvideo",
            str(target),
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="regenerate cached clips and raw frames")
    p.add_argument("--frame", type=int, default=0, help="frame index (within the extracted run) to score")
    p.add_argument("--flat-region", default="860,500,1140,690")
    p.add_argument("--texture-region", default="1560,300,1680,500")
    p.add_argument("--sample-target", type=int, default=60)
    args = p.parse_args()

    if not SOURCE.exists():
        sys.exit(f"{SOURCE} not found")

    WORKDIR.mkdir(parents=True, exist_ok=True)
    build_binary()

    clean_clip = WORKDIR / f"clean_f{FRAMES}.mkv"
    noisy_clip = WORKDIR / f"noisy_a{ALLS}_s{SEED}_f{FRAMES}.mkv"
    clean_raw = WORKDIR / "clean.raw"
    noisy_raw = WORKDIR / "noisy.raw"

    extract_clean_clip(clean_clip, args.force)
    add_noise(clean_clip, noisy_clip, args.force)
    extract_raw_gray(clean_clip, clean_raw, args.force)
    extract_raw_gray(noisy_clip, noisy_raw, args.force)

    if not BINARY.exists():
        sys.exit(f"{BINARY} not found even after building it")

    run(
        [
            str(BINARY),
            "--clean", str(clean_raw),
            "--noisy", str(noisy_raw),
            "--width", str(WIDTH),
            "--height", str(HEIGHT),
            "--frame", str(args.frame),
            "--flat-region", args.flat_region,
            "--texture-region", args.texture_region,
            "--sample-target", str(args.sample_target),
        ]
    )


if __name__ == "__main__":
    main()
