#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pillow"]
# ///
"""Writes amplified removed-residual images for a directory of comparison arms.

The residual is `source - arm` on one frame, centred on mid-grey and
amplified, so what a denoiser took out becomes visible. Earlier rounds
built these with an ffmpeg `blend=all_mode=grainextract,eq=contrast=16`
chain. That chain is kept in the older READMEs, but it fails silently on
low-residual content: on the near-clean asterisk clip it produced four
byte-identical PNGs for four different denoisers, dominated by 0 and 255,
which is not a residual image at all. Nothing in the recipe reports that
it happened.

This script computes the residual arithmetic itself, so the numbers behind
each image are printed next to it, and it refuses to stay quiet when two
arms come out identical.

Amplification is chosen from the data by default. Grain-heavy content
carries residuals of tens of code levels and clean content carries one or
two, so a single fixed gain cannot serve both. `--gain auto` picks the
largest integer gain that keeps the 99.9th percentile of the residual
inside range, which keeps the bulk of the detail unclipped.

Usage:
  uv run scripts/make_residual_images.py --dir data/full_comparison_asterisk \\
      --crop 220:220:1220:200 --frame 22 --out residual22_windowB
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MID = 128


def parse_crop(text: str) -> tuple[int, int, int, int]:
    parts = [int(v) for v in text.split(":")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop takes w:h:x:y, the same order ffmpeg uses")
    return tuple(parts)  # type: ignore[return-value]


def extract(clip: Path, frame: int, crop: tuple[int, int, int, int]) -> np.ndarray:
    w, h, x, y = crop
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(clip),
            "-vf", f"select='eq(n\\,{frame})',crop={w}:{h}:{x}:{y},format=gray",
            "-frames:v", "1", "-f", "rawvideo", "-",
        ],
        capture_output=True,
        check=True,
    )
    if len(proc.stdout) != w * h:
        raise SystemExit(
            f"{clip.name}: extracted {len(proc.stdout)} bytes, expected {w * h}. "
            f"Frame {frame} may be past the end of the clip."
        )
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(h, w).astype(np.int16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--source", default="00_source.mkv")
    ap.add_argument("--arms", default="", help="comma list, default every other mkv in --dir")
    ap.add_argument("--frame", type=int, default=22)
    ap.add_argument("--crop", type=parse_crop, required=True, help="w:h:x:y, ffmpeg order")
    ap.add_argument("--gain", default="auto", help="integer amplification, or 'auto'")
    ap.add_argument("--zoom", type=int, default=2, help="nearest-neighbour scale factor")
    ap.add_argument("--out", default="residual", help="subdirectory of --dir to write into")
    args = ap.parse_args()

    src_path = args.dir / args.source
    if not src_path.exists():
        raise SystemExit(f"no source at {src_path}")

    if args.arms:
        arms = [args.dir / a for a in args.arms.split(",")]
    else:
        arms = sorted(p for p in args.dir.glob("*.mkv") if p.name != args.source)
    if not arms:
        raise SystemExit(f"no arms found in {args.dir}")

    src = extract(src_path, args.frame, args.crop)
    residuals = {arm.stem: src - extract(arm, args.frame, args.crop) for arm in arms}

    if args.gain == "auto":
        # Keep the 99.9th percentile inside range rather than the maximum, so
        # a handful of outlier pixels do not flatten everything else.
        reach = max(float(np.percentile(np.abs(d), 99.9)) for d in residuals.values())
        gain = max(1, int((MID - 8) // max(reach, 1e-6)))
    else:
        gain = int(args.gain)

    out_dir = args.dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"frame {args.frame}, crop w:h:x:y = {':'.join(str(v) for v in args.crop)}, gain {gain}x")
    print(f"{'arm':34} {'max|d|':>7} {'p99.9':>7} {'std':>7} {'clipped':>9}")
    digests: dict[str, str] = {}
    for name, d in residuals.items():
        scaled = np.clip(d.astype(np.int32) * gain + MID, 0, 255).astype(np.uint8)
        clipped = 100.0 * np.count_nonzero((scaled == 0) | (scaled == 255)) / scaled.size
        img = Image.fromarray(scaled, mode="L")
        if args.zoom > 1:
            img = img.resize((img.width * args.zoom, img.height * args.zoom), Image.NEAREST)
        path = out_dir / f"{name}_removed_x{gain}.png"
        img.save(path)
        digests[name] = str(np.asarray(img).tobytes().__hash__())
        print(
            f"{name:34} {int(np.abs(d).max()):7d} {np.percentile(np.abs(d), 99.9):7.1f} "
            f"{d.std():7.3f} {clipped:8.2f}%"
        )

    # The failure this script exists to catch. Two denoisers producing a
    # byte-identical residual image means the image is not showing the
    # residual, whatever it looks like.
    duplicates = [n for n, h in digests.items() if list(digests.values()).count(h) > 1]
    if duplicates:
        print(
            f"\nERROR: identical residual images for {', '.join(sorted(duplicates))}. "
            "These are not showing the residual. Check the crop region and the gain.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"\n{len(residuals)} images written to {out_dir}, all distinct")


if __name__ == "__main__":
    main()
