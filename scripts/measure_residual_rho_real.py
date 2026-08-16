#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Measures NLM's residual noise correlation on real footage.

`src/nlmeans/tests/residual_correlation.rs` measures this on synthetic
flat and sine-wave content, generated inside the crate. This script
repeats the same measurement on `data/clean-1080p.mkv`, a genuine
lossless clean source, so the synthetic table can be checked against
real fine texture instead of only a smooth periodic stand-in.

Method, matched to the Rust test file's `measure_diff`:

1. Extract a short run of clean frames from the source (lossless, so
   they are exact ground truth).
2. Add two independently-seeded synthetic noise realisations to the
   *same* clean frames with ffmpeg's `noise` filter, the same filter
   and `allf=t` (temporal, no persistent spatial pattern) mode
   `scripts/quality_runs.py` uses.
3. Denoise both noisy copies with the real `av-denoise nlmeans` CLI,
   `--variant hq`, pinning `--hq-sigma` to the true injected sigma
   measured directly from the two clips (not trusted from ffmpeg's
   `alls` value, which is an amplitude knob, not a calibrated sigma).
4. Difference the two denoised outputs. For two i.i.d. copies of the
   same underlying process, `Corr(A - B) == Corr(A)` exactly, because
   the shared deterministic structural response to the clean image
   (which is not noise, and would corrupt a correlation measurement
   built on `output - clean` alone on textured content) cancels out of
   the difference. This is exactly what `measure_diff` in
   `residual_correlation.rs` does, generalised from a synthetic frame
   to a real one.

Regions: `--flat-region` and `--texture-region` (each `x0,y0,x1,y1`)
let the correlation be measured separately over a flat patch and a
genuinely fine-detail patch of the same real frame, which the sine
generator in the Rust tests cannot exercise (it is smooth and
periodic relative to the ~9x9 window a `search_radius=4` run looks
at; real footage is not).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY = REPO_ROOT / "target" / "release" / "av-denoise"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def extract_clean_clip(source: Path, start: float, frames: int, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start), "-i", str(source),
            "-vframes", str(frames),
            "-c:v", "ffv1",
            str(target),
        ]
    )


def add_noise(clean: Path, alls: int, seed: int, target: Path) -> None:
    if target.exists():
        return
    run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(clean),
            "-vf", f"noise=alls={alls}:allf=t:all_seed={seed}",
            "-c:v", "ffv1",
            str(target),
        ]
    )


def extract_raw_gray(clip: Path, target: Path) -> None:
    if target.exists():
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


def load_raw_gray_frames(path: Path, w: int, h: int) -> np.ndarray:
    """Loads a rawvideo `gray` dump as `(n_frames, h, w)` float64."""
    data = np.fromfile(path, dtype=np.uint8)
    n = data.size // (w * h)
    return data[: n * w * h].reshape(n, h, w).astype(np.float64)


def run_denoise(
    input_clip: Path,
    search_radius: int,
    patch_radius: int,
    strength: float,
    hq_sigma_8bit: float,
    device_args: list[str],
    out_y4m: Path,
) -> None:
    if out_y4m.exists():
        return
    cmd = [
        str(BINARY),
        *device_args,
        "nlmeans",
        "--variant", "hq",
        "--temporal-radius", "0",
        "--search-radius", str(search_radius),
        "--patch-radius", str(patch_radius),
        "--channel-mode", "luma",
        "--strength", str(strength),
        "--self-weight", "1.0",
        "--hq-sigma", f"{hq_sigma_8bit:.6f}",
        "--input", str(input_clip),
        "--workers", "1",
    ]
    with out_y4m.open("wb") as out:
        print(f"$ {' '.join(cmd)} > {out_y4m}", flush=True)
        subprocess.run(cmd, check=True, stdout=out, stderr=subprocess.PIPE)


Y4M_HEADER_RE = re.compile(rb"YUV4MPEG2\s+W(\d+)\s+H(\d+)")


def load_y4m_luma_frames(path: Path) -> np.ndarray:
    """Reads every frame's Y plane out of a y4m stream, `(n, h, w)` uint8.

    Minimal reader: parses W/H from the stream header, then alternates
    reading a `FRAME...\\n` marker and a raw Y-plane (chroma is skipped
    by seeking, since only luma was denoised and only luma is compared
    here).
    """
    with path.open("rb") as f:
        header = f.readline()
        m = Y4M_HEADER_RE.search(header)
        if not m:
            raise ValueError(f"{path}: not a recognised y4m header: {header!r}")
        w, h = int(m.group(1)), int(m.group(2))
        # yuv420p 8-bit: Y is w*h, each chroma plane is (w/2)*(h/2).
        y_bytes = w * h
        chroma_bytes = (w // 2) * (h // 2) * 2
        frames = []
        while True:
            marker = f.readline()
            if not marker:
                break
            if not marker.startswith(b"FRAME"):
                raise ValueError(f"{path}: expected FRAME marker, got {marker!r}")
            y_plane = f.read(y_bytes)
            if len(y_plane) < y_bytes:
                break
            frames.append(np.frombuffer(y_plane, dtype=np.uint8).reshape(h, w).copy())
            f.seek(chroma_bytes, 1)
    return np.stack(frames).astype(np.float64)


def lag1_h(field: np.ndarray) -> float:
    """Pearson correlation at lag 1 along the last (x) axis, pooled over
    every leading axis (frames, or a stack of them)."""
    a = field[..., :-1].reshape(-1)
    b = field[..., 1:].reshape(-1)
    return float(np.corrcoef(a, b)[0, 1])


def lag1_v(field: np.ndarray) -> float:
    """Same as `lag1_h` but along the second-to-last (y) axis."""
    a = field[..., :-1, :].reshape(-1)
    b = field[..., 1:, :].reshape(-1)
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class Sample:
    rho_h: float
    rho_v: float
    sigma_ratio: float


def measure_diff_sample(
    output_a: np.ndarray,
    output_b: np.ndarray,
    input_noise: np.ndarray,
) -> Sample:
    diff = output_a - output_b
    sqrt2 = np.sqrt(2.0)
    return Sample(
        rho_h=lag1_h(diff),
        rho_v=lag1_v(diff),
        sigma_ratio=(float(np.std(diff)) / sqrt2) / float(np.std(input_noise)),
    )


def crop(field: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = region
    return field[..., y0:y1, x0:x1]


def parse_region(s: str) -> tuple[int, int, int, int]:
    parts = [int(p) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("region must be x0,y0,x1,y1")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=REPO_ROOT / "data" / "clean-1080p.mkv")
    p.add_argument("--workdir", type=Path, default=REPO_ROOT / "data" / "rho_real")
    p.add_argument("--start", type=float, default=4.166, help="seek time (s) for the extracted run")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--alls", type=int, default=16, help="ffmpeg noise filter `alls`, matching quality_runs.toml")
    p.add_argument("--seed-a", type=int, default=1001)
    p.add_argument("--seed-b", type=int, default=2002)
    p.add_argument("--search-radii", type=int, nargs="+", default=[2, 4])
    p.add_argument("--patch-radius", type=int, default=4, help="shipped library default")
    p.add_argument("--strength", type=float, default=1.2, help="matches NlmParams.strength in the Rust sweep")
    p.add_argument(
        "--flat-region", type=parse_region, default="860,500,1140,690",
        help="x0,y0,x1,y1 of a flat sub-rectangle to measure separately",
    )
    p.add_argument(
        "--texture-region", type=parse_region, default="1560,300,1680,500",
        help="x0,y0,x1,y1 of a fine-texture sub-rectangle to measure separately",
    )
    p.add_argument("--device", default="discrete:1", help="av-denoise --device value")
    p.add_argument("--accelerators", default="vulkan", help="av-denoise -A value")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.force and args.workdir.exists():
        shutil.rmtree(args.workdir)
    args.workdir.mkdir(parents=True, exist_ok=True)

    if not BINARY.exists():
        sys.exit(f"{BINARY} not found, build it first: cargo build --release --bin av-denoise --features binary,vulkan")

    device_args = ["-A", args.accelerators, "--device", args.device]

    clean_clip = args.workdir / f"clean_f{args.frames}_t{args.start}.mkv"
    noisy_a_clip = args.workdir / f"noisy_a{args.alls}_sA{args.seed_a}_f{args.frames}.mkv"
    noisy_b_clip = args.workdir / f"noisy_a{args.alls}_sB{args.seed_b}_f{args.frames}.mkv"

    extract_clean_clip(args.source, args.start, args.frames, clean_clip)
    add_noise(clean_clip, args.alls, args.seed_a, noisy_a_clip)
    add_noise(clean_clip, args.alls, args.seed_b, noisy_b_clip)

    clean_raw = args.workdir / "clean.raw"
    noisy_a_raw = args.workdir / "noisy_a.raw"
    extract_raw_gray(clean_clip, clean_raw)
    extract_raw_gray(noisy_a_clip, noisy_a_raw)

    # Resolution is fixed for this source; probe once via ffprobe-free
    # arithmetic using the known file size instead of shelling out again.
    w, h = 1920, 1080
    clean = load_raw_gray_frames(clean_raw, w, h)
    noisy_a = load_raw_gray_frames(noisy_a_raw, w, h)
    n = min(clean.shape[0], noisy_a.shape[0])
    clean = clean[:n]
    noisy_a = noisy_a[:n]

    input_noise = noisy_a - clean
    measured_sigma = float(np.std(input_noise))
    print(
        f"measured true injected sigma (8-bit units, {n} frames, whole frame): "
        f"{measured_sigma:.4f} (ffmpeg noise alls={args.alls} is an amplitude knob, "
        f"not a calibrated sigma, hence measuring directly)"
    )

    flat_noise_sigma = float(np.std(crop(input_noise, args.flat_region)))
    tex_noise_sigma = float(np.std(crop(input_noise, args.texture_region)))
    print(f"input noise sigma in flat region:    {flat_noise_sigma:.4f}")
    print(f"input noise sigma in texture region: {tex_noise_sigma:.4f}")

    rows = []
    for search_radius in args.search_radii:
        out_a = args.workdir / f"out_a_r{search_radius}_p{args.patch_radius}.y4m"
        out_b = args.workdir / f"out_b_r{search_radius}_p{args.patch_radius}.y4m"
        run_denoise(
            noisy_a_clip, search_radius, args.patch_radius, args.strength,
            measured_sigma, device_args, out_a,
        )
        run_denoise(
            noisy_b_clip, search_radius, args.patch_radius, args.strength,
            measured_sigma, device_args, out_b,
        )
        output_a = load_y4m_luma_frames(out_a)
        output_b = load_y4m_luma_frames(out_b)
        n2 = min(output_a.shape[0], output_b.shape[0], n)

        whole = measure_diff_sample(output_a[:n2], output_b[:n2], input_noise[:n2])
        flat = measure_diff_sample(
            crop(output_a[:n2], args.flat_region),
            crop(output_b[:n2], args.flat_region),
            crop(input_noise[:n2], args.flat_region),
        )
        tex = measure_diff_sample(
            crop(output_a[:n2], args.texture_region),
            crop(output_b[:n2], args.texture_region),
            crop(input_noise[:n2], args.texture_region),
        )
        rows.append((search_radius, whole, flat, tex))

    print()
    print(
        f"{'R_s':>4} | {'whole_h':>8} {'whole_v':>8} {'whole_sig':>9} | "
        f"{'flat_h':>8} {'flat_v':>8} {'flat_sig':>8} | "
        f"{'tex_h':>8} {'tex_v':>8} {'tex_sig':>8}"
    )
    for search_radius, whole, flat, tex in rows:
        print(
            f"{search_radius:>4} | {whole.rho_h:>8.4f} {whole.rho_v:>8.4f} {whole.sigma_ratio:>9.4f} | "
            f"{flat.rho_h:>8.4f} {flat.rho_v:>8.4f} {flat.sigma_ratio:>8.4f} | "
            f"{tex.rho_h:>8.4f} {tex.rho_v:>8.4f} {tex.sigma_ratio:>8.4f}"
        )

    print()
    for search_radius, whole, flat, tex in rows:
        valid = whole.sigma_ratio < 0.9 and flat.sigma_ratio < 0.9 and tex.sigma_ratio < 0.9
        print(
            f"search_radius={search_radius}: "
            f"{'VALID' if valid else 'INVALID (sigma_ratio too close to 1.0)'}"
        )


if __name__ == "__main__":
    main()
