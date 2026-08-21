#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""High-frequency energy ratio against a clean reference, for the
architecture-split configurations on clean-1080p.mkv plus synthetic
noise.

Unlike scripts/measure_brick_hf_energy.py (which has no clean
reference and can only compare a denoised clip's Laplacian variance to
the noisy source's), this script compares each denoised clip's
Laplacian variance to the CLEAN reference's own Laplacian variance. A ratio near 1.0 means the denoiser's output carries about as
much high-frequency energy as the true, noise-free image, at every
scale the Laplacian is sensitive to. Below 1.0 means detail was
smoothed away along with the noise; above 1.0 usually means residual
noise or ringing is still contributing high-frequency energy the clean
image does not have.

Reuses the cached reference and noisy clips quality_runs.py already
built in data/quality_runs_light/ (same input, frame count, and seed),
and pipes av-denoise's own release binary directly rather than through
`cargo run`, since the binary is already built.

Also reports the ffmpeg BM3D reference clips scripts/bm3d_parallel.py
writes to the same directory (data/quality_runs_light/bm3d_a{N}_f240_s4242.mkv),
one per noise level, against the same clean reference. BM3D runs on its
own schedule (a separate, much slower CPU pass), so those clips are read
if present rather than generated here; pass --skip-bm3d to omit them
entirely.

Usage:
  uv run scripts/measure_architecture_hf_energy_groundtruth.py --device discrete:1
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

W, H = 1920, 1080
FRAMES = 240
WORKDIR = Path("data/quality_runs_light")
BIN = Path("target/release/av-denoise")

LAPLACIAN = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])

CONFIGS: dict[str, list[str]] = {
    "av_hq_temporal_r2": [
        "nlmeans", "--variant", "hq", "--temporal-radius", "2",
        "--channel-mode", "luma,chroma",
    ],
    "av_nl3d_temporal_r2": [
        "nl3d", "--variant", "hq", "--temporal-radius", "2",
        "--channel-mode", "luma,chroma",
    ],
    "av_nl3d_collab_alone_r0s0": [
        "nl3d", "--variant", "hq", "--temporal-radius", "0",
        "--search-radius", "0", "--channel-mode", "luma,chroma",
    ],
    "av_nl3d_temporal_only_r2s0": [
        "nl3d", "--variant", "hq", "--temporal-radius", "2",
        "--search-radius", "0", "--channel-mode", "luma,chroma",
    ],
    "av_nl3d_temporal_only_r2s0_fss03": [
        "nl3d", "--variant", "hq", "--temporal-radius", "2",
        "--search-radius", "0", "--front-strength-scale", "0.3",
        "--channel-mode", "luma,chroma",
    ],
    "av_nl3d_temporal_only_r2s0_fss015": [
        "nl3d", "--variant", "hq", "--temporal-radius", "2",
        "--search-radius", "0", "--front-strength-scale", "0.15",
        "--channel-mode", "luma,chroma",
    ],
}


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="discrete:1")
    p.add_argument("--noise", default="4,6,8")
    p.add_argument("--only", default="", help="comma-separated config names")
    p.add_argument("--skip-bm3d", action="store_true", help="omit the ffmpeg BM3D reference rows")
    return p.parse_args()


def extract_gray_frames_from_file(clip: Path) -> np.ndarray:
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


def extract_gray_frames_denoised(noisy: Path, args: list[str], device: str) -> np.ndarray:
    p1 = subprocess.Popen(
        [
            str(BIN), "-A", "vulkan", "--device", device,
            *args, "--workers", "2", "--input", str(noisy),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert p1.stdout is not None
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "yuv4mpegpipe", "-i", "-",
            "-vf", "format=gray",
            "-f", "rawvideo",
            "-",
        ],
        stdin=p1.stdout,
        stdout=subprocess.PIPE,
        check=True,
    )
    p1.stdout.close()
    rc1 = p1.wait()
    if rc1 != 0:
        raise RuntimeError(f"av-denoise exited {rc1} for args {args}")
    data = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = data.size // (W * H)
    assert data.size == n * W * H, f"unexpected denoised frame count for args {args}"
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


def main() -> None:
    args = parse_cli()
    noise_levels = [int(x) for x in args.noise.split(",") if x.strip()]
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    configs = {k: v for k, v in CONFIGS.items() if not only or k in only}

    ref_path = WORKDIR / f"ref_f{FRAMES}.mkv"
    ref_frames = extract_gray_frames_from_file(ref_path)
    ref_vars = np.array([laplacian_variance(ref_frames[i]) for i in range(ref_frames.shape[0])])
    print(f"clean reference: {ref_frames.shape[0]} frames, mean laplacian var={ref_vars.mean():.2f}")

    print()
    print(f"{'config':<34}  {'noise':>5}  {'hf_ratio_mean':>14}  {'min':>8}  {'max':>8}")
    for alls in noise_levels:
        noisy_path = WORKDIR / f"noisy_a{alls}_f{FRAMES}_s4242.mkv"
        for name, cli_args in configs.items():
            den_frames = extract_gray_frames_denoised(noisy_path, cli_args, args.device)
            n = min(den_frames.shape[0], ref_frames.shape[0])
            den_vars = np.array([laplacian_variance(den_frames[i]) for i in range(n)])
            ratio = den_vars / ref_vars[:n]
            print(
                f"{name:<34}  {alls:>5}  {ratio.mean():>14.4f}  {ratio.min():>8.4f}  {ratio.max():>8.4f}"
            )

        if not args.skip_bm3d:
            bm3d_path = WORKDIR / f"bm3d_a{alls}_f{FRAMES}_s4242.mkv"
            if not bm3d_path.exists():
                print(f"{'ffmpeg_bm3d':<34}  {alls:>5}  (not found, run scripts/bm3d_parallel.py first)")
                continue
            bm3d_frames = extract_gray_frames_from_file(bm3d_path)
            n = min(bm3d_frames.shape[0], ref_frames.shape[0])
            bm3d_vars = np.array([laplacian_variance(bm3d_frames[i]) for i in range(n)])
            ratio = bm3d_vars / ref_vars[:n]
            print(
                f"{'ffmpeg_bm3d':<34}  {alls:>5}  {ratio.mean():>14.4f}  {ratio.min():>8.4f}  {ratio.max():>8.4f}"
            )


if __name__ == "__main__":
    main()
