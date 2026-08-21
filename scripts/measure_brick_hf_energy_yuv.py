#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Measures Laplacian variance (a high-frequency-energy proxy) per
plane, luma and both chroma planes separately, for a directory of
brick comparison clips, relative to the source clip.

This is the chroma-aware sibling of `measure_brick_hf_energy.py`, which
only ever looked at luma. Luma and chroma are extracted and reported
independently here on purpose. A combined number would hide exactly
the luma/chroma gap this script exists to show.

Same caveat as the luma-only script: Laplacian variance says how much
high-frequency content survives, relative to the source. It cannot
tell retained real texture apart from retained grain, so a higher
ratio is not automatically better. Treat this as context next to the
residual images, not as a quality ranking.

Chroma planes are read at their native (typically half) resolution and
never upsampled, so the luma and chroma ratios are not directly
comparable in absolute terms, only each against its own source plane.

Usage:
  uv run scripts/measure_brick_hf_energy_yuv.py --dir data/nl3d_chroma_residual_sigma_scale_sweep
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

LAPLACIAN = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


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


def laplacian_variance(field: np.ndarray) -> float:
    h, w = field.shape
    resp = np.zeros((h - 2, w - 2))
    for dy in range(3):
        for dx in range(3):
            k = LAPLACIAN[dy, dx]
            if k == 0.0:
                continue
            resp += k * field[dy: dy + h - 2, dx: dx + w - 2]
    return float(np.var(resp))


def per_frame_ratio(arr: np.ndarray, src: np.ndarray) -> np.ndarray:
    n = min(arr.shape[0], src.shape[0])
    vars_ = np.array([laplacian_variance(arr[i]) for i in range(n)])
    src_vars = np.array([laplacian_variance(src[i]) for i in range(n)])
    return vars_ / src_vars


def whole_clip_ratio(arr: np.ndarray, src: np.ndarray) -> float:
    return float(per_frame_ratio(arr, src).mean())


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


def parse_clips_arg(directory: Path, spec: str) -> dict[str, Path]:
    clips: dict[str, Path] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        filename, _, label = item.partition("=")
        clips[label or default_label(filename)] = directory / filename
    return clips


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dir", type=Path, required=True, help="directory holding the clips")
    ap.add_argument("--source", default="00_source.mkv", help="source clip filename within --dir")
    ap.add_argument(
        "--clips", default="",
        help="comma list of filename=label pairs; default is every other *.mkv in --dir",
    )
    ap.add_argument(
        "--still-frames", default="22,33,10",
        help="comma list of frame indices reported individually per plane",
    )
    args = ap.parse_args()

    directory: Path = args.dir
    clips = parse_clips_arg(directory, args.clips) if args.clips else discover_clips(directory, args.source)
    if not clips:
        raise SystemExit(f"no clips found in {directory} (besides {args.source})")

    missing = [name for name, path in clips.items() if not path.exists()]
    if missing:
        print(f"skipping missing clips: {missing}")
        clips = {name: path for name, path in clips.items() if path.exists()}

    source_path = directory / args.source
    w, h = probe_dims(source_path)
    cw, ch = w // 2, h // 2
    print(f"{directory}: luma {w}x{h}, chroma {cw}x{ch} (assumes 4:2:0)")

    planes = {
        "Y": ("format=gray", w, h),
        "U": ("extractplanes=u", cw, ch),
        "V": ("extractplanes=v", cw, ch),
    }

    src_planes = {
        name: extract_plane_frames(source_path, filt, pw, ph)
        for name, (filt, pw, ph) in planes.items()
    }
    n_frames = src_planes["Y"].shape[0]
    print(f"{n_frames} frames in source")
    print()

    still_frames = tuple(int(v) for v in args.still_frames.split(","))
    clip_planes: dict[str, dict[str, np.ndarray]] = {}

    print("whole-clip mean ratio vs source:")
    header = f"{'clip':<20}" + "".join(f"{name:>10}" for name in planes)
    print(header)
    for label, path in clips.items():
        clip_planes[label] = {
            name: extract_plane_frames(path, filt, pw, ph) for name, (filt, pw, ph) in planes.items()
        }
        row = f"{label:<20}"
        for name in planes:
            ratio = whole_clip_ratio(clip_planes[label][name], src_planes[name])
            row += f"{ratio:>10.3f}"
        print(row)

    print()
    print(f"per-frame ratio at frames {still_frames}:")
    for plane_name in planes:
        print(f"  plane {plane_name}:")
        for f in still_frames:
            row = "    ".join(
                f"{label}={per_frame_ratio(arr[plane_name], src_planes[plane_name])[f]:.3f}"
                for label, arr in clip_planes.items()
                if f < arr[plane_name].shape[0]
            )
            print(f"    frame {f:>2}: {row}")


if __name__ == "__main__":
    main()
