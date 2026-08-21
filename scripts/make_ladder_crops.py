#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Builds stills, zoom crops and amplified residuals for a ladder of arms.

The crop window for each frame is found by search, as the window where a
probe arm departs furthest from the reference arm, rather than chosen by
eye. Every arm is then cropped at that same window, so the comparison is
not steered toward whichever arm the window was found on.

Outputs, under `--dir`:
  stills/f<frame>_<arm>.png     full 1920x1080 frame, untouched
  zoom/f<frame>_<arm>.png       crop scaled by --zoom, nearest neighbour
  residual/f<frame>_<arm>.png   (arm - reference) amplified, crop window

Usage:
  uv run scripts/make_ladder_crops.py --dir data/nl4d_mismatch_ladder \\
      --ref 10_off.mkv --probe 14_scale_08.mkv --frames 242,243,250,100
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def probe_dims(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def gray_frame(path: Path, frame: int, w: int, h: int) -> np.ndarray:
    """One frame's luma plane, selected by decoded frame index."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
         "-vf", f"select=eq(n\\,{frame})", "-fps_mode", "passthrough", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    if len(raw) < w * h:
        sys.exit(f"short read for frame {frame} of {path}")
    return np.frombuffer(raw[: w * h], dtype=np.uint8).reshape(h, w)


def best_window(a: np.ndarray, b: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left of the size x size window where a and b differ most.

    Uses a summed-area table so every candidate window is considered
    exactly, rather than sampling a coarse grid and hoping.
    """
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    integral = np.zeros((diff.shape[0] + 1, diff.shape[1] + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(diff, axis=0), axis=1)

    # Window sums for every valid top-left, via inclusion-exclusion.
    s = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    y, x = np.unravel_index(int(np.argmax(s)), s.shape)
    return int(x), int(y)


def run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def write_png(arr: np.ndarray, path: Path) -> None:
    """Writes a grayscale array as a PNG through ffmpeg."""
    h, w = arr.shape
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{w}x{h}", "-i", "-", str(path)],
        input=arr.astype(np.uint8).tobytes(), check=True, capture_output=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--ref", required=True, help="reference arm filename within --dir")
    ap.add_argument("--probe", required=True, help="arm used to locate the crop window")
    ap.add_argument("--frames", required=True, help="comma list of decoded frame indices")
    ap.add_argument("--size", type=int, default=220, help="crop window edge in pixels")
    ap.add_argument("--zoom", type=int, default=3, help="nearest-neighbour scale factor")
    ap.add_argument("--amplify", type=int, default=24, help="residual amplification")
    args = ap.parse_args()

    directory: Path = args.dir
    arms = sorted(p for p in directory.glob("*.mkv") if not p.is_symlink())
    if not arms:
        sys.exit(f"no arms found in {directory}")

    ref_path = directory / args.ref
    probe_path = directory / args.probe
    for p in (ref_path, probe_path):
        if not p.exists():
            sys.exit(f"missing: {p}")

    for sub in ("stills", "zoom", "residual"):
        (directory / sub).mkdir(exist_ok=True)

    w, h = probe_dims(ref_path)
    frames = [int(v) for v in args.frames.split(",")]

    print(f"arms: {', '.join(p.name for p in arms)}")
    print(f"window located by {args.probe} against {args.ref}, {args.size}px\n")

    for frame in frames:
        ref_g = gray_frame(ref_path, frame, w, h)
        probe_g = gray_frame(probe_path, frame, w, h)
        x0, y0 = best_window(ref_g, probe_g, args.size)
        peak = int(np.max(np.abs(ref_g.astype(np.int32) - probe_g.astype(np.int32))))
        print(f"frame {frame}: window x={x0} y={y0}  peak|d| {peak}")

        for arm in arms:
            label = arm.stem
            run_ffmpeg([
                "ffmpeg", "-v", "error", "-y", "-nostdin", "-i", str(arm),
                "-vf", f"select=eq(n\\,{frame})", "-fps_mode", "passthrough", "-frames:v", "1",
                str(directory / "stills" / f"f{frame}_{label}.png"),
            ])
            run_ffmpeg([
                "ffmpeg", "-v", "error", "-y", "-nostdin", "-i", str(arm),
                "-vf", (f"select=eq(n\\,{frame}),crop={args.size}:{args.size}:{x0}:{y0},"
                        f"scale=iw*{args.zoom}:ih*{args.zoom}:flags=neighbor"),
                "-fps_mode", "passthrough", "-frames:v", "1",
                str(directory / "zoom" / f"f{frame}_{label}.png"),
            ])

            if arm == ref_path:
                continue
            arm_g = gray_frame(arm, frame, w, h)
            resid = np.abs(arm_g.astype(np.int32) - ref_g.astype(np.int32))
            resid = np.clip(resid * args.amplify, 0, 255)
            crop = resid[y0:y0 + args.size, x0:x0 + args.size]
            crop = np.repeat(np.repeat(crop, args.zoom, axis=0), args.zoom, axis=1)
            write_png(crop, directory / "residual" / f"f{frame}_{label}.png")

    print(f"\nwrote stills/, zoom/ and residual/ under {directory}")


if __name__ == "__main__":
    main()
