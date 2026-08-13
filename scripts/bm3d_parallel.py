#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Denoise a clip with ffmpeg's CPU BM3D filter across several processes.

ffmpeg's `bm3d` filter slice-threads badly. On a 32-core box a single
process peaks near 750% CPU, and raising `-filter_threads` past ~8 makes
it slower rather than faster. BM3D as ffmpeg implements it is purely
spatial, so the clip splits into contiguous frame ranges that denoise
independently and concatenate back without seams.

Each worker seeks to the exact presentation timestamp of its first frame
and writes a lossless FFV1 part. The parts are stream-copied into the
final file, and both the per-part and total frame counts are checked
against the source so a bad seek fails loudly instead of silently
shifting frame indices. At a fixed `--filter-threads` the concatenated
result is bit-identical to denoising the clip in one process.

Note that ffmpeg's `bm3d` gives a different result for every thread
count, because slice boundaries change how overlapping block estimates
aggregate. Keep `--filter-threads` fixed across runs that get compared.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Throughput peaks with a handful of threads per process and degrades
# past that, so cores are spent on more processes instead.
THREADS_PER_JOB = 4


@dataclass(frozen=True)
class Chunk:
    """One contiguous frame range handled by a single ffmpeg process."""

    index: int
    start_frame: int
    frames: int
    start_pts: str
    part: Path


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", required=True, type=Path)
    p.add_argument("--output", "-o", required=True, type=Path)
    p.add_argument("--sigma", default="15")
    p.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=f"concurrent ffmpeg processes (default: cores // {THREADS_PER_JOB})",
    )
    p.add_argument(
        "--filter-threads",
        type=int,
        default=THREADS_PER_JOB,
        help=f"-filter_threads per process (default: {THREADS_PER_JOB})",
    )
    p.add_argument(
        "--keep-parts",
        action="store_true",
        help="leave the per-chunk files behind for inspection",
    )
    return p.parse_args()


def default_jobs() -> int:
    cores = os.cpu_count() or THREADS_PER_JOB
    return max(1, cores // THREADS_PER_JOB)


def probe_pts(path: Path) -> list[str]:
    """Presentation timestamps of every video packet, in display order."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    pts = [line.strip().rstrip(",") for line in out.splitlines() if line.strip()]
    if not pts:
        sys.exit(f"no video packets found in {path}")
    try:
        pts.sort(key=float)
    except ValueError:
        sys.exit(f"{path} has non-numeric packet timestamps, cannot split it")
    return pts


def count_frames(path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return int(out)


def plan_chunks(pts: list[str], jobs: int, parts_dir: Path) -> list[Chunk]:
    """Split the frame range into `jobs` near-equal contiguous chunks."""
    total = len(pts)
    jobs = max(1, min(jobs, total))
    base, extra = divmod(total, jobs)

    chunks: list[Chunk] = []
    start = 0
    for i in range(jobs):
        frames = base + (1 if i < extra else 0)
        chunks.append(
            Chunk(
                index=i,
                start_frame=start,
                frames=frames,
                start_pts=pts[start],
                part=parts_dir / f"part_{i:03d}.mkv",
            )
        )
        start += frames
    return chunks


def worker_command(chunk: Chunk, source: Path, sigma: str, threads: int) -> list[str]:
    """Full two-stage BM3D over one chunk. The basic estimate feeds the
    final estimate as its reference stream.

    ffmpeg's bm3d defaults `group` (the max number of blocks collected
    into one 3D group) to 1, which skips block matching entirely and
    denoises each block on its own. That is not collaborative filtering,
    the step that makes BM3D BM3D. The reference implementation groups
    up to 16 blocks in the hard-thresholding stage and up to 32 in the
    Wiener stage, so both stages are set explicitly here."""
    graph = (
        f"split[a][b];"
        f"[b]bm3d=sigma={sigma}:group=16:estim=basic[basic];"
        f"[a][basic]bm3d=sigma={sigma}:group=32:estim=final:ref=1"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-filter_threads",
        str(threads),
        "-ss",
        chunk.start_pts,
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        str(chunk.frames),
        "-vf",
        graph,
        "-fps_mode",
        "passthrough",
        "-c:v",
        "ffv1",
        str(chunk.part),
    ]


def run_chunks(chunks: list[Chunk], source: Path, sigma: str, threads: int) -> None:
    """Start every chunk at once and wait for all of them."""
    procs = [
        (chunk, subprocess.Popen(worker_command(chunk, source, sigma, threads)))
        for chunk in chunks
    ]

    failures: list[str] = []
    for chunk, proc in procs:
        if proc.wait() != 0:
            failures.append(
                f"chunk {chunk.index} (frames {chunk.start_frame}.."
                f"{chunk.start_frame + chunk.frames - 1}) exited {proc.returncode}"
            )
    if failures:
        sys.exit("BM3D chunk failed:\n  " + "\n  ".join(failures))


def verify_chunks(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        if not chunk.part.exists():
            sys.exit(f"chunk {chunk.index} produced no output at {chunk.part}")
        produced = count_frames(chunk.part)
        if produced != chunk.frames:
            sys.exit(
                f"chunk {chunk.index} produced {produced} frames but "
                f"{chunk.frames} were requested starting at frame "
                f"{chunk.start_frame} (pts {chunk.start_pts}); the source "
                f"cannot be split by timestamp, rerun with --jobs 1"
            )


def concat_parts(chunks: list[Chunk], output: Path, parts_dir: Path) -> None:
    listing = parts_dir / "parts.txt"
    listing.write_text(
        "".join(f"file '{chunk.part.resolve()}'\n" for chunk in chunks)
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listing),
        "-c",
        "copy",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_cli()
    if not args.input.exists():
        sys.exit(f"input {args.input} does not exist")

    jobs = args.jobs if args.jobs > 0 else default_jobs()
    pts = probe_pts(args.input)
    total = len(pts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output.parent / f".{args.output.stem}_bm3d_parts"
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True)

    chunks = plan_chunks(pts, jobs, parts_dir)
    print(
        f"[bm3d] {total} frames -> {len(chunks)} chunks x "
        f"{args.filter_threads} filter threads, sigma {args.sigma}",
        flush=True,
    )
    first = worker_command(chunks[0], args.input, args.sigma, args.filter_threads)
    print(f"[bm3d] chunk 0 runs `{shlex.join(first)}`", flush=True)

    started = time.monotonic()
    try:
        run_chunks(chunks, args.input, args.sigma, args.filter_threads)
        verify_chunks(chunks)
        concat_parts(chunks, args.output, parts_dir)
    finally:
        if not args.keep_parts:
            shutil.rmtree(parts_dir, ignore_errors=True)

    elapsed = time.monotonic() - started
    produced = count_frames(args.output)
    if produced != total:
        sys.exit(
            f"{args.output} has {produced} frames but the source has {total}"
        )
    print(
        f"[bm3d] {produced} frames in {elapsed:.1f}s "
        f"({produced / elapsed:.1f} fps) -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
