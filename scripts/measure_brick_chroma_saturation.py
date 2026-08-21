#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Measures chroma magnitude (distance from neutral grey) for a directory of brick
comparison clips, and compares it against the source.

Chroma magnitude at a pixel is sqrt((U-128)^2 + (V-128)^2) on the raw 8-bit U/V
planes. A pixel at (128, 128) is neutral grey; magnitude rises with how saturated
that pixel is, regardless of hue. This is a different measure from the Laplacian
high-frequency ratio in `measure_brick_hf_energy_yuv.py`: that one tracks fine
structure, this one tracks how colourful the picture is overall. A filter that
flattens saturated regions towards grey (a desaturation failure) lowers this
number even in places with no fine structure to lose, so the two measures can
disagree and both are worth checking.

Reports whole-clip mean and standard deviation of chroma magnitude, per clip,
alongside the same figures for the source.

Usage:
  uv run scripts/measure_brick_chroma_saturation.py --dir data/nl3d_chroma_residual_sigma_scale_high_sweep
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np


def probe_dims(clip: Path) -> tuple[int, int]:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(clip),
        ],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
    )
    w, h = proc.stdout.strip().split("x")
    return int(w), int(h)


def extract_plane_frames(clip: Path, plane_filter: str, w: int, h: int) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(clip),
            "-vf", plane_filter,
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    data = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = data.size // (w * h)
    assert data.size == n * w * h, f"unexpected size for {clip} ({plane_filter}): {data.size}"
    return data.reshape(n, h, w).astype(np.float64)


def chroma_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt((u - 128.0) ** 2 + (v - 128.0) ** 2)


def default_label(filename: str) -> str:
    stem = Path(filename).stem
    prefix, sep, rest = stem.partition("_")
    if sep and prefix.isdigit():
        return rest
    return stem


def discover_clips(directory: Path, source_name: str) -> dict[str, Path]:
    return {
        default_label(path.name): path
        for path in sorted(directory.glob("*.mkv"))
        if path.name != source_name
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dir", type=Path, required=True, help="directory holding the clips")
    ap.add_argument("--source", default="00_source.mkv", help="source clip filename within --dir")
    args = ap.parse_args()

    directory: Path = args.dir
    clips = discover_clips(directory, args.source)
    if not clips:
        raise SystemExit(f"no clips found in {directory} (besides {args.source})")

    source_path = directory / args.source
    w, h = probe_dims(source_path)
    cw, ch = w // 2, h // 2
    print(f"{directory}: chroma {cw}x{ch} (assumes 4:2:0)")

    def magnitude_for(path: Path) -> np.ndarray:
        u = extract_plane_frames(path, "extractplanes=u", cw, ch)
        v = extract_plane_frames(path, "extractplanes=v", cw, ch)
        return chroma_magnitude(u, v)

    src_mag = magnitude_for(source_path)
    print(f"{src_mag.shape[0]} frames in source")
    print()

    header = f"{'clip':<20}{'mean':>10}{'std':>10}{'mean vs src':>14}"
    print(header)
    src_mean, src_std = float(src_mag.mean()), float(src_mag.std())
    print(f"{'source':<20}{src_mean:>10.3f}{src_std:>10.3f}{'--':>14}")

    for label, path in clips.items():
        mag = magnitude_for(path)
        mean, std = float(mag.mean()), float(mag.std())
        delta = mean - src_mean
        print(f"{label:<20}{mean:>10.3f}{std:>10.3f}{delta:>+14.3f}")


if __name__ == "__main__":
    main()
