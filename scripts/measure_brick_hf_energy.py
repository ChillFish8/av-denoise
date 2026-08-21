#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Measures Laplacian variance (a high-frequency-energy proxy) for a
directory of brick comparison clips, relative to the source clip.

This clip has no clean reference, so XPSNR/SSIM are not possible here.
Laplacian variance is a stand-in that says how much high-frequency
content survives in each output, relative to the source. It does not
say whether removing that content was correct, since it cannot tell
retained real texture apart from retained grain. An arm can score
higher simply by leaving more noise in. Treat this as supporting
context next to the residual images, not as the primary evidence.

One directory under `data/` holds one comparison: a `00_source.mkv`
plus a handful of denoised arms, all covering the same segment. This
script works against any such directory, so the same method applies to
every round of the brick comparison without a bespoke copy per round.

Reports, for every clip in the directory except the source:
  - whole-frame mean/min/max ratio over the full clip
  - per-frame ratio at the chosen still frames (22, 33, 10 by default)
  - the frame-22 brick-tower crop ratio (texture-only region, no sky)
  - the whole-clip tower crop mean ratio

Usage:
  uv run scripts/measure_brick_hf_energy.py --dir data/brick_visual_comparison_current
  uv run scripts/measure_brick_hf_energy.py --dir data/nl3d_temporal_sigma \\
      --clips 02_nl3d_r2_before.mkv=nl3d_before,03_nl3d_r2_after.mkv=nl3d_after,04_bm3d_reference.mkv=bm3d

With no `--clips`, every other `*.mkv` file in `--dir` is measured,
labeled from its filename by dropping the leading `NN_` ordering
prefix.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

W, H = 1920, 1080
LAPLACIAN = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])

# Brick-tower crop on frame 22, no sky, texture-only region. Same crop
# every round of this comparison has used.
TOWER_CROP = (700, 90, 1150, 950)
STILL_FRAMES = (22, 33, 10)


def extract_gray_frames(clip: Path) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(clip),
            "-vf", "format=gray",
            "-f", "rawvideo",
            "-",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    data = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = data.size // (W * H)
    assert data.size == n * W * H, f"unexpected size for {clip}: {data.size}"
    return data.reshape(n, H, W).astype(np.float64)


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


def default_label(filename: str) -> str:
    """Derives a short label from a `NN_name.mkv` filename by dropping
    the leading ordering prefix and the extension."""
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


def parse_region(s: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in s.split(","))
    return x0, y0, x1, y1


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
    ap.add_argument("--tower-crop", type=parse_region, default=TOWER_CROP, help="x0,y0,x1,y1")
    ap.add_argument(
        "--still-frames", default=",".join(str(v) for v in STILL_FRAMES),
        help="comma list of frame indices reported individually",
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

    still_frames = tuple(int(v) for v in args.still_frames.split(","))
    x0, y0, x1, y1 = args.tower_crop

    src_frames = extract_gray_frames(directory / args.source)
    n_frames = src_frames.shape[0]
    print(f"{n_frames} frames in source, {directory}")

    src_vars = np.array([laplacian_variance(src_frames[i]) for i in range(n_frames)])
    src_crop_vars = np.array(
        [laplacian_variance(src_frames[i, y0:y1, x0:x1]) for i in range(n_frames)]
    )

    clip_frames = {name: extract_gray_frames(path) for name, path in clips.items()}

    print()
    print("whole-frame mean ratio vs source, over all frames:")
    results: dict[str, np.ndarray] = {}
    for name, arr in clip_frames.items():
        n = min(n_frames, arr.shape[0])
        vars_ = np.array([laplacian_variance(arr[i]) for i in range(n)])
        ratio = vars_ / src_vars[:n]
        results[name] = ratio
        print(f"  {name:<28}: mean={ratio.mean():.3f}  min={ratio.min():.3f}  max={ratio.max():.3f}")

    print()
    print(f"per-frame ratio at frames {still_frames}:")
    for f in still_frames:
        row = "  ".join(f"{name}={ratio[f]:.3f}" for name, ratio in results.items() if f < len(ratio))
        print(f"  frame {f:>2}: {row}")

    print()
    print(f"frame-22 tower-body crop {args.tower_crop} (texture-only region, no sky):")
    for name, arr in clip_frames.items():
        if 22 >= arr.shape[0]:
            continue
        crop_var = laplacian_variance(arr[22, y0:y1, x0:x1])
        print(f"  {name:<28}: ratio={crop_var / src_crop_vars[22]:.3f}")

    print()
    print("whole-clip tower-body crop mean ratio (all frames, same crop coords):")
    for name, arr in clip_frames.items():
        n = min(n_frames, arr.shape[0])
        crop_vars = np.array([laplacian_variance(arr[i, y0:y1, x0:x1]) for i in range(n)])
        ratio = crop_vars / src_crop_vars[:n]
        print(f"  {name:<28}: mean={ratio.mean():.3f}  min={ratio.min():.3f}  max={ratio.max():.3f}")


if __name__ == "__main__":
    main()
