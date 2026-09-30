"""How often does Charades show two objects of the same class at once?

Identity only matters when there is something to confuse: two cups, two chairs.
Action Genome never labels that case (it annotates at most one instance per class
per frame), so its videos are measured directly with the detector instead.

Every AG video (train + test) is sampled at 1 frame/s and run through the 960px
unmerged engine. Only confident detections count (score >= 0.3), with stricter
duplicate suppression than tracking uses (same-class IoU > 0.5 merged), so a part
and a whole of one object are not counted as two objects. Two measures:

``any``       >= 2 same-class detections in the frame
``separate``  >= 2 same-class detections that barely overlap (IoU < 0.1) - clearly
              distinct objects, robust to residual double detections

The raw rates overstate the identity problem: most coexisting same-class objects
are fixed (cabinets, pictures, doorknobs), which position alone separates. The
case that matters is reported separately: >= 2 separate same-class objects of a
*movable* class with at least one overlapping a person box (a proxy for being
interacted with).

Detector counts are not ground truth: example frames are saved with boxes drawn
for a human spot check before any conclusion rests on these numbers.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import box_iou
from tqdm import tqdm

from sggpipeline.ag.dataset import ActionGenome
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.owlv2 import postprocess
from sggpipeline.detect.trt_runner import TRTRunner
from sggpipeline.pipeline import Workspace, load_queries, write_report

SCORE, NMS_IOU, SEPARATE_IOU = 0.3, 0.5, 0.1

# Fixed furniture and fittings: several can coexist (kitchen cabinets, pictures on
# a wall) but they do not move, so position alone keeps them apart and they pose
# no real identity problem. Everything else except person counts as movable.
FIXED = {"bed", "chair", "closet/cabinet", "door", "doorknob", "doorway", "floor", "light",
         "mirror", "picture", "refrigerator", "shelf", "sofa/couch", "table", "television",
         "window"}


def sample_video(path: str) -> tuple[str, list[np.ndarray], float]:
    """Decode one video on a worker process, keeping one frame per second."""
    frames, fps = [], 30.0
    try:
        with av.open(path) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            s.codec_context.thread_count = 1
            fps = float(s.average_rate or 30.0)
            step = max(1, round(fps))
            for i, f in enumerate(c.decode(s)):
                if i % step == 0:
                    frames.append(f.to_ndarray(format="rgb24"))
    except Exception:  # a truncated video should not stop the survey
        pass
    return path, frames, fps


def same_class_groups(det, separate: bool) -> dict[int, int]:
    """Class -> number of instances, for classes present at least twice."""
    out = {}
    for cls, n in Counter(det.labels.tolist()).items():
        if n < 2:
            continue
        if not separate:
            out[cls] = n
            continue
        boxes = torch.from_numpy(det.boxes[det.labels == cls])
        iou = box_iou(boxes, boxes).numpy()
        np.fill_diagonal(iou, 1.0)  # a box is never "apart" from itself
        # count boxes that are clearly apart from at least one other same-class box
        apart = (iou < SEPARATE_IOU).any(axis=1).sum()
        if apart >= 2:
            out[cls] = int(apart)
    return out


def touches_person(det, cls: int, person: int) -> bool:
    """Any detection of ``cls`` overlapping a person box: a proxy for interaction."""
    people = det.boxes[det.labels == person]
    objects = det.boxes[det.labels == cls]
    if not len(people) or not len(objects):
        return False
    iou = box_iou(torch.from_numpy(objects), torch.from_numpy(people)).numpy()
    return bool((iou > 0).any())


def draw(image: np.ndarray, det, classes, highlight: int) -> Image.Image:
    im = Image.fromarray(image)
    d = ImageDraw.Draw(im)
    for box, lab, sc in zip(det.boxes, det.labels, det.scores, strict=True):
        colour = (255, 40, 40) if lab == highlight else (60, 200, 60)
        d.rectangle(box.tolist(), outline=colour, width=2)
        d.text((box[0] + 2, box[1] + 2), f"{classes[lab]} {sc:.2f}", fill=colour)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--examples", type=int, default=24)
    args = ap.parse_args()

    ws = Workspace(Path(args.artifacts))
    queries = load_queries(ws, "base")
    query = torch.from_numpy(queries.embeds).cuda()
    runner = TRTRunner(ws.engine("base_fp16.plan"))
    pre = GpuOwlv2Preprocessor(960, device="cuda")

    split_of = {}
    classes = None
    for split in ("train", "test"):
        ag = ActionGenome(root=Path(args.ag_root), split=split).load()
        classes = ag.classes
        split_of.update({f.video_id: split for f in ag.frames})
    videos = sorted(split_of)[: args.max_videos]
    videos_dir = Path(args.ag_root) / "Charades_v1_480"
    paths = [str(videos_dir / v) for v in videos]
    print(f"{len(videos)} videos, sampling 1 frame/s", flush=True)

    frames_total = 0
    frames_any, frames_sep, frames_mov, frames_live = 0, 0, 0, 0
    class_any, class_sep, class_mov, class_live = Counter(), Counter(), Counter(), Counter()
    live_candidates = []
    person = classes.index("person")
    per_video = {}
    candidates = []  # (video, second, class) for example frames
    rng = random.Random(0)

    with Pool(args.workers) as pool:
        for path, frames, fps in tqdm(pool.imap_unordered(sample_video, paths, chunksize=4),
                                      total=len(paths), desc="videos", mininterval=30):
            video = Path(path).name
            run_any = run_sep = best_any = best_sep = 0
            n_any = n_sep = 0
            run_live = best_live = n_live = 0
            for second, frame in enumerate(frames):
                chw = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
                out = runner.infer({"pixel_values": pre.preprocess_tensor(chw),
                                    "query_embeds": query})
                a = {k: v.float().cpu().numpy() for k, v in out.items()}
                det = postprocess(a["pred_logits"], a["pred_boxes"], a["objectness"],
                                  queries.owner, len(classes), (frame.shape[1], frame.shape[0]),
                                  SCORE, 100, NMS_IOU)
                g_any, g_sep = same_class_groups(det, False), same_class_groups(det, True)
                g_mov = {c: n for c, n in g_sep.items() if classes[c] not in FIXED
                         and classes[c] != "person"}
                g_live = {c: n for c, n in g_mov.items() if touches_person(det, c, person)}
                frames_total += 1
                if g_mov:
                    frames_mov += 1
                    class_mov.update(g_mov.keys())
                if g_live:
                    frames_live += 1
                    n_live += 1
                    class_live.update(g_live.keys())
                    if rng.random() < 0.05:
                        live_candidates.append((video, second, max(g_live, key=g_live.get),
                                                frame, det))
                run_live = run_live + 1 if g_live else 0
                best_live = max(best_live, run_live)
                if g_any:
                    frames_any += 1
                    n_any += 1
                    class_any.update(g_any.keys())
                if g_sep:
                    frames_sep += 1
                    n_sep += 1
                    class_sep.update(g_sep.keys())
                    if rng.random() < 0.02:
                        candidates.append((video, second, max(g_sep, key=g_sep.get), frame, det))
                run_any = run_any + 1 if g_any else 0
                run_sep = run_sep + 1 if g_sep else 0
                best_any, best_sep = max(best_any, run_any), max(best_sep, run_sep)
            per_video[video] = {"split": split_of[video], "seconds": len(frames),
                                "frames_any": n_any, "frames_separate": n_sep,
                                "frames_movable_near_person": n_live,
                                "longest_run_any_s": best_any, "longest_run_separate_s": best_sep,
                                "longest_run_movable_near_person_s": best_live}

    # Example frames for a human spot check, spread over classes.
    out_dir = ws.root / "results" / "same_class_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_class = defaultdict(list)
    for c in (live_candidates or candidates):
        by_class[c[2]].append(c)
    picked = []
    while len(picked) < args.examples and any(by_class.values()):
        for cls in list(by_class):
            if by_class[cls] and len(picked) < args.examples:
                picked.append(by_class[cls].pop(rng.randrange(len(by_class[cls]))))
    for video, second, cls, frame, det in picked:
        draw(frame, det, classes, cls).save(out_dir / f"{video[:-4]}_{second:03d}s_{classes[cls].replace('/', '-')}.jpg")

    vids = list(per_video.values())

    def share(pred):
        return sum(1 for v in vids if pred(v)) / max(len(vids), 1)

    summary = {
        "videos": len(vids), "sampled_frames": frames_total,
        "frames_with_same_class_any": frames_any / max(frames_total, 1),
        "frames_with_same_class_separate": frames_sep / max(frames_total, 1),
        "videos_with_any_separate_frame": share(lambda v: v["frames_separate"] > 0),
        "videos_with_separate_run_ge_3s": share(lambda v: v["longest_run_separate_s"] >= 3),
        "videos_with_separate_run_ge_10s": share(lambda v: v["longest_run_separate_s"] >= 10),
        "frames_with_movable_same_class": frames_mov / max(frames_total, 1),
        "frames_with_movable_same_class_near_person": frames_live / max(frames_total, 1),
        "videos_with_movable_near_person_frame": share(lambda v: v["frames_movable_near_person"] > 0),
        "videos_with_movable_near_person_run_ge_3s": share(
            lambda v: v["longest_run_movable_near_person_s"] >= 3),
        "top_classes_movable_near_person": {classes[k]: n for k, n in class_live.most_common(15)},
        "top_classes_movable": {classes[k]: n for k, n in class_mov.most_common(15)},
        "top_classes_separate": {classes[k]: n for k, n in class_sep.most_common(15)},
        "fixed_classes": sorted(FIXED),
        "top_classes_any": {classes[k]: n for k, n in class_any.most_common(15)},
        "thresholds": {"score": SCORE, "nms_iou": NMS_IOU, "separate_iou": SEPARATE_IOU},
        "examples_dir": str(out_dir), "examples": len(picked),
    }
    write_report({"summary": summary, "per_video": per_video}, ws.result("same_class_mining.json"))
    for k, v in summary.items():
        print(f"{k}: {v}", flush=True)
    print("MINING_DONE", flush=True)


if __name__ == "__main__":
    main()
