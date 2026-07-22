#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Extract the same frame numbers from several videos, each with its own
frame offset.

Frames are pulled by decoded frame index (ffmpeg's `select=eq(n,N)`), so
container start_times, B-frame reorder, and CFR rounding never shift the
selected frame.

A file's offset is added to every requested index before extraction, which
lines up sources that disagree about where frame zero sits. One case that
needs it is a source whose first coded picture displays before its first
keyframe, since ffmpeg drops that picture and ffms2 keeps it, leaving
av-denoise output one frame ahead of the same clip decoded by ffmpeg.

PNGs are named by the requested index rather than the shifted one, so the
same number always means the same moment in every file.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUT_DIR = Path("./data/compare_frames")


@dataclass(frozen=True)
class Video:
    """One input file, its frame offset, and the stem its PNGs get."""

    path: Path
    offset: int
    name: str

    def effective(self, requested: int) -> int:
        """Decoded frame index to pull for a requested index."""
        return requested + self.offset


def parse_video_spec(spec: str) -> tuple[Path, int]:
    """Split `PATH` or `PATH:OFFSET`.

    The suffix counts as an offset only when it parses as an integer, so a
    path that contains a colon is left alone.
    """
    head, sep, tail = spec.rpartition(":")
    if sep and head:
        try:
            return Path(head), int(tail)
        except ValueError:
            pass
    return Path(spec), 0


def parse_frames(raw: str) -> list[int]:
    """Comma-separated frame indices, de-duplicated and sorted."""
    frames = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            frames.append(int(part))
        except ValueError:
            sys.exit(f"--frames expects comma-separated integers, got {part!r}")
    if not frames:
        sys.exit("--frames must contain at least one frame index")
    negative = [n for n in frames if n < 0]
    if negative:
        sys.exit(f"--frames indices must be >= 0, got {negative}")
    return sorted(set(frames))


def resolve_videos(specs: list[str]) -> list[Video]:
    """Name each file by its stem.

    Two inputs sharing a stem would otherwise write over each other's PNGs,
    so a repeated stem is prefixed with its parent directory name, and
    anything still clashing after that gets a numeric suffix.
    """
    parsed = [parse_video_spec(s) for s in specs]

    stem_counts: dict[str, int] = {}
    for path, _ in parsed:
        stem_counts[path.stem] = stem_counts.get(path.stem, 0) + 1

    videos: list[Video] = []
    taken: set[str] = set()
    for path, offset in parsed:
        base = path.stem if stem_counts[path.stem] == 1 else f"{path.parent.name}_{path.stem}"
        name = base
        suffix = 2
        while name in taken:
            name = f"{base}_{suffix}"
            suffix += 1
        taken.add(name)
        videos.append(Video(path=path, offset=offset, name=name))
    return videos


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--video",
        action="append",
        default=[],
        metavar="PATH[:OFFSET]",
        help="input file, repeatable. OFFSET is added to every requested "
        "frame index for this file and may be negative (e.g. out.mkv:1)",
    )
    p.add_argument(
        "--frames",
        required=True,
        help="comma-separated frame indices to extract from every file",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory the PNGs are written to (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="re-extract even when the PNGs already exist",
    )
    args = p.parse_args()
    if not args.video:
        sys.exit("at least one --video is required")
    return args


def extract(video: Video, frames: list[int], out_dir: Path, force: bool) -> None:
    """Pull every requested frame from one file in a single ffmpeg pass."""
    if not video.path.exists():
        sys.exit(f"[{video.name}] input {video.path} does not exist")

    effective = [video.effective(n) for n in frames]
    below_zero = [(n, e) for n, e in zip(frames, effective) if e < 0]
    if below_zero:
        pairs = ", ".join(f"{n}{video.offset:+d} = {e}" for n, e in below_zero)
        sys.exit(
            f"[{video.name}] offset {video.offset:+d} puts {len(below_zero)} "
            f"index(es) before the start of the clip ({pairs})"
        )

    final_paths = [out_dir / f"{video.name}_{n:03d}.png" for n in frames]
    if not force and all(p.exists() for p in final_paths):
        print(
            f"[{video.name}] skipping ({len(final_paths)} PNGs present, "
            f"pass --force to re-extract)",
            flush=True,
        )
        return

    # ffmpeg emits select matches in ascending decoded-index order. The
    # offset is constant per file, so that order matches `frames` ascending
    # and the two lists zip together directly.
    eq_chain = "+".join(f"eq(n\\,{e})" for e in effective)
    tmp_template = out_dir / f".{video.name}_tmp_%05d.png"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video.path),
        "-vf",
        f"select='{eq_chain}'",
        "-fps_mode",
        "passthrough",
        "-frames:v",
        str(len(effective)),
        str(tmp_template),
    ]

    mapping = (
        f"{frames}" if video.offset == 0 else f"{frames} -> {effective} (offset {video.offset:+d})"
    )
    print(f"[{video.name}] extracting {mapping}", flush=True)
    print(f"  `{shlex.join(cmd)}`", flush=True)

    # Clear stale temporaries so a previous partial run cannot be counted
    # as this run's output.
    for stale in out_dir.glob(f".{video.name}_tmp_*.png"):
        stale.unlink(missing_ok=True)

    subprocess.run(cmd, check=True)

    produced = sorted(out_dir.glob(f".{video.name}_tmp_*.png"))
    if len(produced) != len(effective):
        for p in produced:
            p.unlink(missing_ok=True)
        missing = effective[len(produced) :]
        sys.exit(
            f"[{video.name}] expected {len(effective)} frames but ffmpeg "
            f"produced {len(produced)}, likely out-of-range index(es) "
            f"{missing} in {video.path}"
        )

    for tmp, requested in zip(produced, frames):
        final = out_dir / f"{video.name}_{requested:03d}.png"
        shutil.move(str(tmp), str(final))
        print(f"  -> {final}", flush=True)


def main() -> None:
    args = parse_cli()
    frames = parse_frames(args.frames)
    videos = resolve_videos(args.video)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"extracting frames {frames} from {len(videos)} file(s):", flush=True)
    for v in videos:
        print(f"  {v.name:<24} offset {v.offset:+d}  {v.path}", flush=True)
    print(flush=True)

    for video in videos:
        extract(video, frames, args.out_dir, args.force)


if __name__ == "__main__":
    main()
