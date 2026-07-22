#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Queue denoise commands, run them sequentially, and extract the same frame
numbers from every output for side-by-side visual comparison.

Frames are pulled by *decoded frame index* (ffmpeg's `select=eq(n,N)`), so
container start_times / B-frame reorder / CFR rounding never shift the
selected frame.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path



# ---------------------------------------------------------------------------
# Edit these to define a comparison.
# ---------------------------------------------------------------------------

OUT_DIR = Path("./data/compare_frames")
COMPARE_VIDEOS_DIR = Path("./data/compare_videos")

# Frame indices (0-based) extracted from every queued output.
FRAMES: list[int] = [
    100,
    150,
    200,
    250,
    300,
]


@dataclass(frozen=True)
class Run:
    """One queue entry.

    `command` of `None` means "don't denoise, just extract from `output`"
    — useful for a source/reference entry.

    `output` is optional; if omitted, defaults to
    `COMPARE_VIDEOS_DIR / f"{name}.mkv"`.

    Inside `command`, the placeholder `{out_file}` is replaced with the
    resolved output path and `{av_flags}` with the backend flags built
    from `--accelerators` / `--device`. (When writing the command as an
    f-string, escape as `{{out_file}}` and `{{av_flags}}`.)
    """

    name: str
    command: str | None
    output: Path | None = None

    def resolved_output(self) -> Path:
        return self.output if self.output is not None else COMPARE_VIDEOS_DIR / f"{self.name}.mkv"


SOURCE_VIDEO = Path("./data/91-days.mkv")

RUNS: list[Run] = [
    Run("source", None, SOURCE_VIDEO),
    Run(
        "fast_t0",
        f"just denoise-file -i {SOURCE_VIDEO} -o {{out_file}} -- {{av_flags}} "
        f"--preset veryfast --temporal-radius 0 --channel-mode luma,chroma",
    ),
    Run(
        "fast_t2_mc_off",
        f"just denoise-file -i {SOURCE_VIDEO} -o {{out_file}} -- {{av_flags}} "
        f"--preset veryfast --temporal-radius 2 --channel-mode luma,chroma",
    ),
    Run(
        "fast_t2_mc_on",
        f"just denoise-file -i {SOURCE_VIDEO} -o {{out_file}} -- {{av_flags}} "
        f"--preset veryfast --temporal-radius 2 --channel-mode luma,chroma --motion-compensation",
    ),
    Run(
        "fast_t4_mc_off",
        f"just denoise-file -i {SOURCE_VIDEO} -o {{out_file}} -- {{av_flags}} "
        f"--preset veryfast --temporal-radius 4 --channel-mode luma,chroma",
    ),
    Run(
        "fast_t4_mc_on",
        f"just denoise-file -i {SOURCE_VIDEO} -o {{out_file}} -- {{av_flags}} "
        f"--preset veryfast --temporal-radius 4 --channel-mode luma,chroma --motion-compensation",
    ),
    Run(
        "hq_t0",
        f"just denoise-file -i {SOURCE_VIDEO} -o {{out_file}} -- {{av_flags}} "
        f"--preset base --temporal-radius 0 --channel-mode luma,chroma",
    ),
    Run(
        "hq_t2_mc_off",
        f"just denoise-file -i {SOURCE_VIDEO} -o {{out_file}} -- {{av_flags}} "
        f"--preset base --temporal-radius 2 --channel-mode luma,chroma",
    ),
    Run(
        "hq_t2_mc_on",
        f"just denoise-file -i {SOURCE_VIDEO} -o {{out_file}} -- {{av_flags}} "
        f"--preset base --temporal-radius 2 --channel-mode luma,chroma --motion-compensation",
    ),
    Run(
        "hq_t4_mc_off",
        f"just denoise-file -i {SOURCE_VIDEO} -o {{out_file}} -- {{av_flags}} "
        f"--preset base --temporal-radius 4 --channel-mode luma,chroma",
    ),
    Run(
        "hq_t4_mc_on",
        f"just denoise-file -i {SOURCE_VIDEO} -o {{out_file}} -- {{av_flags}} "
        f"--preset base --temporal-radius 4 --channel-mode luma,chroma --motion-compensation",
    ),
    Run(
        "hq_t4_heavy",
        f"just denoise-file -i {SOURCE_VIDEO} -o {{out_file}} -- {{av_flags}} "
        f"--preset slow --temporal-radius 4 --search-radius 4 --patch-radius 6 "
        f"--channel-mode luma,chroma --hq-thsad-scale 2.0",
    ),
    Run(
        "ffmpeg_opencl_nlmeans",
        f"just denoise-file-ffmpeg -i {SOURCE_VIDEO} -o {{out_file}} "
        f"--patch 9 --search 5 --strength 1.2",
    ),
    # BM3D is CPU-only and slow. `just denoise-file-bm3d` splits the clip
    # across one process per four cores, since the filter's own slice
    # threading stops scaling around eight threads. Its sigma is on
    # ffmpeg's own scale rather than the true noise sigma, so it needs
    # sweeping per clip.
    Run(
        "ffmpeg_bm3d",
        f"just denoise-file-bm3d -i {SOURCE_VIDEO} -o {{out_file}} --sigma 15",
    ),
]

# ---------------------------------------------------------------------------
# Implementation — usually no need to edit below.
# ---------------------------------------------------------------------------


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="re-run every command even if its output file already exists",
    )
    p.add_argument(
        "--skip-extract",
        action="store_true",
        help="run denoise commands only; do not extract comparison frames",
    )
    p.add_argument(
        "--only",
        default="",
        help="comma-separated run names to include (default: all)",
    )
    p.add_argument(
        "--accelerators",
        default="",
        help="av-denoise --accelerators value applied to every av-denoise run",
    )
    p.add_argument(
        "--device",
        default="",
        help="av-denoise --device value applied to every av-denoise run",
    )
    return p.parse_args()


def backend_flags(accelerators: str, device: str) -> str:
    """Backend selection flags shared by every av-denoise run."""
    flags = []
    if accelerators:
        flags.append(f"--accelerators {accelerators}")
    if device:
        flags.append(f"--device {device}")
    return " ".join(flags)


def filtered_runs(only: str) -> list[Run]:
    if not only:
        return RUNS
    wanted = {n.strip() for n in only.split(",") if n.strip()}
    unknown = wanted - {r.name for r in RUNS}
    if unknown:
        sys.exit(f"--only references unknown run(s): {sorted(unknown)}")
    return [r for r in RUNS if r.name in wanted]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def run_command(run: Run, force: bool, av_flags: str = "") -> None:
    output = run.resolved_output()

    if run.command is None:
        if not output.exists():
            sys.exit(
                f"run {run.name!r} has no command and its output "
                f"{output} does not exist"
            )
        print(f"[{run.name}] using existing {output}", flush=True)
        return

    if output.exists() and not force:
        print(
            f"[{run.name}] skipping (output exists; pass --force to re-run): "
            f"{output}",
            flush=True,
        )
        return

    # Substituting "{av_flags} " rather than "{av_flags}" keeps the command
    # free of a stray double space when no backend flags are set.
    rendered = run.command.replace("{out_file}", str(output)).replace(
        "{av_flags} ", f"{av_flags} " if av_flags else ""
    )
    print(f"[{run.name}] running: {rendered}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(rendered, shell=True, check=True)


def extract_frames(run: Run, frames: list[int], force: bool) -> None:
    if not frames:
        return

    sorted_frames = sorted(set(frames))
    final_paths = [OUT_DIR / f"{run.name}_{n:03d}.png" for n in sorted_frames]

    if not force and all(p.exists() for p in final_paths):
        print(
            f"[{run.name}] skipping extraction (all {len(final_paths)} PNGs "
            f"present; pass --force to re-extract)",
            flush=True,
        )
        return

    eq_chain = "+".join(f"eq(n\\,{n})" for n in sorted_frames)
    tmp_template = OUT_DIR / f".{run.name}_tmp_%05d.png"
    output = run.resolved_output()

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(output),
        "-vf",
        f"select='{eq_chain}'",
        "-fps_mode",
        "passthrough",
        "-frames:v",
        str(len(sorted_frames)),
        str(tmp_template),
    ]

    print(
        f"[{run.name}] extracting frames {sorted_frames} via "
        f"`{shlex.join(cmd)}`",
        flush=True,
    )
    subprocess.run(cmd, check=True)

    # Verify count, then rename in select-order (ascending frame index).
    produced = sorted(OUT_DIR.glob(f".{run.name}_tmp_*.png"))
    if len(produced) != len(sorted_frames):
        # Clean up partials before bailing so the next run doesn't trip over them.
        for p in produced:
            p.unlink(missing_ok=True)
        missing = sorted_frames[len(produced):]
        sys.exit(
            f"[{run.name}] expected {len(sorted_frames)} frames but ffmpeg "
            f"produced {len(produced)}; likely out-of-range index(es) "
            f"{missing} in {output}"
        )

    for tmp, frame_idx in zip(produced, sorted_frames):
        final = OUT_DIR / f"{run.name}_{frame_idx:03d}.png"
        shutil.move(str(tmp), str(final))
        print(f"  -> {final}", flush=True)


def main() -> None:
    args = parse_cli()
    ensure_dirs()
    runs = filtered_runs(args.only)
    av_flags = backend_flags(args.accelerators, args.device)

    for run in runs:
        run_command(run, force=args.force, av_flags=av_flags)
        if not args.skip_extract:
            extract_frames(run, FRAMES, force=args.force)


if __name__ == "__main__":
    main()
