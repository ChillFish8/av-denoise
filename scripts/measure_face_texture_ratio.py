#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Re-measures the face-crop high-frequency-energy ratio task-C4's report
used, against a freshly built av-denoise binary, to check whether
correlation-aware noise shaping changes how much fine texture nl3d
preserves relative to hq's own residual-noise-inflated baseline.

Method, matched to task-C4-shift-report.md's "Step 5" exactly: Laplacian
variance (a texture/high-frequency-energy proxy) over two face crops
(`face1` = the left character's head, `face2` = the right character's),
each split left/right of its own midline, at frame 15 of the 140-frame
segment already checked into `data/nl3d_visual_comparison/` (absolute
frame 45 of the source clip). `01_noisy_n{4,6,8}.mkv` and
`00_clean_reference.mkv` are reused directly; `03_nl3d_r2_n{4,6,8}.mkv`
(generated before the noise-shaping change) supplies the "before" ratio
for a direct old-vs-new comparison, and a fresh nl3d run against the
noisy input, through whatever `target/release/av-denoise` currently is,
supplies "after".
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np

W, H = 1920, 1080
FRAME_IDX = 15  # absolute frame 45, segment-relative
FACE1 = (520, 80, 820, 380)  # x0, y0, x1, y1
FACE2 = (1150, 180, 1500, 530)

DATA_DIR = Path("data/nl3d_visual_comparison")
NL3D_ARGS = [
    "--variant", "hq", "--temporal-radius", "2", "--channel-mode", "luma,chroma",
]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def extract_gray_frame(clip: Path, frame_idx: int) -> np.ndarray:
    proc = run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(clip),
            "-vf", f"select='eq(n\\,{frame_idx})',format=gray",
            "-vframes", "1",
            "-f", "rawvideo",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    frame = np.frombuffer(proc.stdout, dtype=np.uint8)
    assert frame.size == W * H, f"expected {W * H} bytes, got {frame.size} from {clip}"
    return frame.reshape(H, W).astype(np.float64)


def denoise_nl3d(noisy: Path, device: str, target: Path) -> None:
    device_args = ["--device", device] if device else []
    p1 = subprocess.Popen(
        [
            "target/release/av-denoise", "nl3d", *NL3D_ARGS,
            "-A", "vulkan", *device_args,
            "--workers", "2",
            "--input", str(noisy),
        ],
        stdout=subprocess.PIPE,
    )
    with target.open("wb") as out:
        p2 = subprocess.Popen(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "yuv4mpegpipe", "-i", "-", "-c:v", "ffv1", "-f", "matroska", "-"],
            stdin=p1.stdout,
            stdout=out,
        )
        assert p1.stdout is not None
        p1.stdout.close()
        p2.communicate()
    rc1 = p1.wait()
    if rc1 != 0 or p2.returncode != 0:
        raise RuntimeError(f"nl3d denoise failed for {noisy}: rc1={rc1} rc2={p2.returncode}")


LAPLACIAN = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def laplacian_variance(field: np.ndarray) -> float:
    h, w = field.shape
    resp = np.zeros((h - 2, w - 2))
    for dy in range(3):
        for dx in range(3):
            k = LAPLACIAN[dy, dx]
            if k == 0.0:
                continue
            resp += k * field[dy : dy + h - 2, dx : dx + w - 2]
    return float(np.var(resp))


def crop(field: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = region
    return field[y0:y1, x0:x1]


def split_lr_ratio(clean: np.ndarray, other: np.ndarray, region: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = region
    mid = (x0 + x1) // 2
    left = (x0, y0, mid, y1)
    right = (mid, y0, x1, y1)
    ratios = []
    for r in (left, right):
        c_var = laplacian_variance(crop(clean, r))
        o_var = laplacian_variance(crop(other, r))
        ratios.append(o_var / c_var)
    return ratios[0], ratios[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="discrete:1")
    ap.add_argument("--levels", default="4,6,8")
    ap.add_argument("--skip-fresh-run", action="store_true", help="reuse a cached fresh nl3d clip if present")
    args = ap.parse_args()

    clean = extract_gray_frame(DATA_DIR / "00_clean_reference.mkv", FRAME_IDX)

    print(f"{'level':>5}  {'face':>5}  {'old L/R':>15}  {'new L/R':>15}")
    for level in [int(x) for x in args.levels.split(",")]:
        old_clip = DATA_DIR / f"03_nl3d_r2_n{level}.mkv"
        noisy_clip = DATA_DIR / f"01_noisy_n{level}.mkv"
        new_clip = Path(f"/tmp/nl3d_shaped_n{level}.mkv")

        if not (args.skip_fresh_run and new_clip.exists()):
            denoise_nl3d(noisy_clip, args.device, new_clip)

        old_frame = extract_gray_frame(old_clip, FRAME_IDX)
        new_frame = extract_gray_frame(new_clip, FRAME_IDX)

        for name, region in [("face1", FACE1), ("face2", FACE2)]:
            old_l, old_r = split_lr_ratio(clean, old_frame, region)
            new_l, new_r = split_lr_ratio(clean, new_frame, region)
            print(f"{level:>5}  {name:>5}  {old_l:>7.4f}/{old_r:<7.4f}  {new_l:>7.4f}/{new_r:<7.4f}")


if __name__ == "__main__":
    main()
