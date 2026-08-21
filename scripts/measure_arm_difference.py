#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Pixel-difference statistics between a reference arm and one or more test arms.

Reports the same table shape the confidence ablation recorded, so numbers
from different rounds are directly comparable: overall RMS in 8-bit code
levels, largest absolute difference, share of pixels differing by more than
one level, and the per-frame RMS spread.

Luma-only and all-plane figures are both reported. A sweep that moves only
one plane shows up far more clearly in the luma-only column, and the
all-plane column is what a whole-frame measure of an both-plane toggle
produces, so quoting both keeps a cross-round comparison honest about which
measure it is reading.

Frames are streamed as raw yuv420p, so nothing depends on container
timestamps or frame reordering.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def probe_dims(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def open_stream(path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-nostdin",
            "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-",
        ],
        stdout=subprocess.PIPE,
    )


@dataclass
class Accum:
    """Running difference statistics, kept in float64 to avoid drift."""

    sq_sum: float = 0.0
    count: int = 0
    max_abs: int = 0
    over_one: int = 0
    per_frame_rms: list[float] = field(default_factory=list)

    def add(self, diff: np.ndarray) -> None:
        sq = float(np.sum(diff.astype(np.float64) ** 2))
        self.sq_sum += sq
        self.count += diff.size
        self.max_abs = max(self.max_abs, int(np.max(np.abs(diff))) if diff.size else 0)
        self.over_one += int(np.count_nonzero(np.abs(diff) > 1))
        self.per_frame_rms.append((sq / diff.size) ** 0.5 if diff.size else 0.0)

    @property
    def rms(self) -> float:
        return (self.sq_sum / self.count) ** 0.5 if self.count else 0.0

    @property
    def over_one_pct(self) -> float:
        return 100.0 * self.over_one / self.count if self.count else 0.0


def compare(ref: Path, test: Path) -> tuple[Accum, Accum, int]:
    """Streams both arms once, returning (luma, all-plane, frames)."""
    w, h = probe_dims(ref)
    tw, th = probe_dims(test)
    if (w, h) != (tw, th):
        sys.exit(f"dimension mismatch: {ref} is {w}x{h}, {test} is {tw}x{th}")

    luma_bytes = w * h
    frame_bytes = luma_bytes * 3 // 2

    a, b = open_stream(ref), open_stream(test)
    luma, allp = Accum(), Accum()
    frames = 0

    try:
        while True:
            fa = a.stdout.read(frame_bytes)
            fb = b.stdout.read(frame_bytes)
            if len(fa) < frame_bytes or len(fb) < frame_bytes:
                break

            # int16 so the subtraction cannot wrap.
            va = np.frombuffer(fa, dtype=np.uint8).astype(np.int16)
            vb = np.frombuffer(fb, dtype=np.uint8).astype(np.int16)
            diff = vb - va

            luma.add(diff[:luma_bytes])
            allp.add(diff)
            frames += 1
    finally:
        for p in (a, b):
            if p.stdout:
                p.stdout.close()
            p.wait()

    return luma, allp, frames


def report(name: str, acc: Accum, top: int) -> None:
    rms = np.array(acc.per_frame_rms)
    print(f"  {name:10s} RMS {acc.rms:7.4f}  max|d| {acc.max_abs:6d}  "
          f">1 level {acc.over_one_pct:6.3f}%  "
          f"per-frame min/mean/max {rms.min():.3f}/{rms.mean():.3f}/{rms.max():.3f}")
    if top:
        order = np.argsort(rms)[::-1][:top]
        listed = ", ".join(f"{int(i)} ({rms[i]:.3f})" for i in order)
        print(f"  {'':10s} top {top} frames by RMS: {listed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=Path, required=True, help="reference arm")
    ap.add_argument("test", type=Path, nargs="+", help="arms to compare against the reference")
    ap.add_argument("--top", type=int, default=8, help="list this many worst frames (0 to skip)")
    args = ap.parse_args()

    for path in [args.ref, *args.test]:
        if not path.exists():
            sys.exit(f"missing: {path}")

    print(f"reference: {args.ref}\n")
    for test in args.test:
        luma, allp, frames = compare(args.ref, test)
        print(f"{test.name}  ({frames} frames)")
        report("luma", luma, args.top)
        report("all-plane", allp, 0)
        print()


if __name__ == "__main__":
    main()
