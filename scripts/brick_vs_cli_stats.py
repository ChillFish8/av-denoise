#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Residual-std matching table and pairwise CLI/plugin diffs for the
brick VS-vs-CLI quality comparison (2026-08-29, ad hoc, not committed
as a permanent script).

Residual std (source - arm), per plane, matches this project's own
rule against Laplacian-based matching (see
project_laplacian_ratio_misleads_on_haar). Pairwise diffs are computed
per frame and reduced to max/mean overall and per leading-edge-vs-rest
split, so a leading-edge-confined difference is visible directly
instead of being averaged away.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

DIR = Path("data/brick_vs_cli")
W, H = 1920, 1080
NFRAMES = 53
PLANES = {
    "y": ("format=gray", W, H),
    "u": ("extractplanes=u", W // 2, H // 2),
    "v": ("extractplanes=v", W // 2, H // 2),
}
LEADING_EDGE = 4  # 2 * temporal_radius(2), per vs_harness.py PARITY_LEADING_EDGE_FRAMES


def read_planes(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for name, (vf, w, h) in PLANES.items():
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
             "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, check=True,
        ).stdout
        arr = np.frombuffer(raw, dtype=np.uint8)
        n = arr.size // (w * h)
        out[name] = arr[: n * w * h].reshape(n, h, w).astype(np.int32)
    return out


def main() -> None:
    arms = ["source", "cli_nl4d", "cli_hq", "vs_nl4d", "vs_hq"]
    data = {arm: read_planes(DIR / f"{arm}.mkv") for arm in arms}
    for arm in arms:
        for p in PLANES:
            n = data[arm][p].shape[0]
            assert n == NFRAMES, f"{arm} plane {p}: {n} frames, expected {NFRAMES}"

    src = data["source"]

    print("=== residual std (source - arm), per plane, mean over clip ===")
    print(f"{'arm':10s} {'Y':>8s} {'U':>8s} {'V':>8s}")
    for arm in ["cli_nl4d", "cli_hq", "vs_nl4d", "vs_hq"]:
        row = []
        for p in PLANES:
            resid = (src[p] - data[arm][p]).astype(np.float64)
            per_frame_std = resid.reshape(resid.shape[0], -1).std(axis=1)
            row.append(per_frame_std.mean())
        print(f"{arm:10s} " + " ".join(f"{v:8.4f}" for v in row))

    print()
    print("=== pairwise CLI vs plugin, per plane ===")

    def pair_stats(a_arm, b_arm):
        a = data[a_arm]
        b = data[b_arm]
        for p in PLANES:
            d = np.abs(a[p] - b[p])
            n = d.shape[0]
            all_max = int(d.max())
            all_mean = float(d.mean())
            lead_max = int(d[:LEADING_EDGE].max())
            lead_mean = float(d[:LEADING_EDGE].mean())
            rest_max = int(d[LEADING_EDGE:].max())
            rest_mean = float(d[LEADING_EDGE:].mean())
            identical = bool(np.array_equal(a[p], b[p]))
            print(
                f"  plane {p}: max={all_max:3d} mean={all_mean:.5f}  "
                f"| leading(0:{LEADING_EDGE}) max={lead_max:3d} mean={lead_mean:.5f}  "
                f"| rest({LEADING_EDGE}:{n}) max={rest_max:3d} mean={rest_mean:.5f}  "
                f"| identical={identical}"
            )

    print("nl4d: cli_nl4d vs vs_nl4d")
    pair_stats("cli_nl4d", "vs_nl4d")
    print("hq:   cli_hq vs vs_hq")
    pair_stats("cli_hq", "vs_hq")


if __name__ == "__main__":
    main()
