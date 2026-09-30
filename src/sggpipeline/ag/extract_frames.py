"""Dump the annotated Action Genome keyframes out of the Charades videos.

AG ships annotations but not pixels: ``frame_list.txt`` names 288,782 frames of
the form ``<video>.mp4/<number>.png`` that have to be decoded from Charades.

**Frame numbers are 1-based.**  They come from ffmpeg's ``image2`` numbering,
which starts at 1, so frame ``000089.png`` is decode position 88.  The evidence
is checkable rather than assumed: across all 288,782 entries the smallest index
is 3 and no index is 0, while no index exceeds its video's decoded frame count.
An off-by-one here would silently misalign every box in the benchmark, so
``one_based`` is exposed rather than buried.

Videos are decoded **sequentially** and needed frames picked off as they pass.
Seeking per frame is both slower and less reliable on these files, since not
every target is a keyframe.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def read_frame_list(path: Path) -> dict[str, list[int]]:
    """Group ``frame_list.txt`` entries into ``{video_file: [frame numbers]}``."""
    grouped: dict[str, list[int]] = defaultdict(list)
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            video, _, frame = line.partition("/")
            try:
                grouped[video].append(int(Path(frame).stem))
            except ValueError:
                continue
    return {video: sorted(set(frames)) for video, frames in grouped.items()}


@dataclass(slots=True)
class ExtractionResult:
    video: str
    written: int
    skipped: int
    missing: int
    error: str | None = None


def extract_video(
    video_path: Path,
    frame_numbers: list[int],
    out_dir: Path,
    one_based: bool = True,
    image_format: str = "png",
    overwrite: bool = False,
    decoder_threads: int = 2,
) -> ExtractionResult:
    """Decode one video and write the requested frames.

    Resumable: frames already on disk are skipped, so an interrupted run can be
    restarted without redoing finished videos.
    """
    import av
    from PIL import Image

    video_path, out_dir = Path(video_path), Path(out_dir)
    wanted = {n - 1 if one_based else n: n for n in frame_numbers}

    targets = {}
    skipped = 0
    for position, number in wanted.items():
        destination = out_dir / f"{number:06d}.{image_format}"
        if destination.exists() and not overwrite:
            skipped += 1
            continue
        targets[position] = destination

    if not targets:
        return ExtractionResult(video_path.name, 0, skipped, 0)
    if not video_path.exists():
        return ExtractionResult(video_path.name, 0, skipped, len(targets), "video missing")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    last_needed = max(targets)

    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            # FFmpeg defaults to one decoder thread per core. With many worker
            # processes that multiplies past container thread limits (EAGAIN);
            # parallelism comes from the workers, so each decoder stays small.
            stream.codec_context.thread_count = decoder_threads
            for position, frame in enumerate(container.decode(stream)):
                destination = targets.pop(position, None)
                if destination is not None:
                    image: Image.Image = frame.to_image()
                    image.save(destination)
                    written += 1
                if position >= last_needed:
                    break
    except Exception as exc:  # a handful of Charades files are truncated
        return ExtractionResult(video_path.name, written, skipped, len(targets), str(exc))

    return ExtractionResult(video_path.name, written, skipped, len(targets))


def _worker(job):
    video_path, numbers, out_dir, one_based, image_format, overwrite, threads = job
    return extract_video(video_path, numbers, out_dir, one_based, image_format, overwrite,
                         threads)


def extract_all(
    frame_list: Path,
    videos_dir: Path,
    frames_dir: Path,
    videos: set[str] | None = None,
    workers: int | None = None,
    one_based: bool = True,
    image_format: str = "png",
    overwrite: bool = False,
    progress: bool = True,
    decoder_threads: int = 2,
) -> dict:
    """Extract every listed frame, one process per video."""
    from multiprocessing import Pool

    grouped = read_frame_list(frame_list)
    if videos is not None:
        grouped = {v: f for v, f in grouped.items() if v in videos}

    videos_dir, frames_dir = Path(videos_dir), Path(frames_dir)
    jobs = [
        (videos_dir / video, numbers, frames_dir / video, one_based, image_format, overwrite,
         decoder_threads)
        for video, numbers in sorted(grouped.items())
    ]
    workers = workers or max(1, (os.cpu_count() or 4) - 2)

    results: list[ExtractionResult] = []
    with Pool(processes=workers) as pool:
        iterator = pool.imap_unordered(_worker, jobs, chunksize=4)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(jobs), desc="extracting", unit="video")
        results.extend(iterator)

    errors = [r for r in results if r.error]
    return {
        "videos_processed": len(results),
        "frames_written": sum(r.written for r in results),
        "frames_already_present": sum(r.skipped for r in results),
        "frames_missing": sum(r.missing for r in results),
        "videos_with_errors": len(errors),
        "errors": [{"video": r.video, "error": r.error} for r in errors[:20]],
    }
