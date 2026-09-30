"""Smoke test: does a larger zero-shot OWLv2 lift the relationship ceiling?

Detection is what limits the relationship head (object recall 0.62 on Action
Genome, 0.51 on VG150), and most of the loss is wrong labels on well-placed
boxes. Training a classifier would fix labels but puts the open-vocabulary
claim at risk; a stronger *zero-shot* detector would not. This compares OWLv2
checkpoints on the same frames, with the same postprocessing as the caches the
head trains on (Action Genome: score >= 0.05, NMS 0.7, top 32; VG150: score >=
0.01, NMS 0.5, top 64), each at its native resolution.

Reported per dataset: object recall (label and IoU >= 0.5), the class-agnostic
share (any label, IoU >= 0.5), and the pair ceiling - true subject-object pairs
whose two boxes are both recovered, which bounds relationship recall.

Both models run eagerly in PyTorch fp16 at batch 1; the timings are for
comparing the two, not deployment numbers (TensorRT is several times faster).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm import tqdm

from sggpipeline.ag.classes import AG_OBJECT_CLASSES, build_prompt_index
from sggpipeline.ag.relations import load_relation_frames
from sggpipeline.detect.fast_preprocess import GpuOwlv2Preprocessor
from sggpipeline.detect.owlv2 import Owlv2DetectionGraph, encode_text_queries, load_owlv2, postprocess_device
from sggpipeline.detect.preprocess import load_image
from sggpipeline.pipeline import Workspace, write_report
from sggpipeline.vg.data import iter_split, vocabulary

PERSON = AG_OBJECT_CLASSES.index("person")
SETTINGS = {"ag": (0.05, 0.7, 32), "vg": (0.01, 0.5, 64)}


def samples(args):
    """(dataset, image loader, gt boxes, gt labels, gt pairs) for a fixed frame sample."""
    frames = load_relation_frames(Path(args.ag_root), "test")
    step = max(1, len(frames) // args.frames)
    for fr in frames[::step][: args.frames]:
        boxes = np.vstack([fr.person_box[None], fr.object_boxes]).astype(np.float32)
        labels = np.concatenate([[PERSON], fr.object_labels])
        pairs = np.stack([np.zeros(len(fr.object_labels), int), 1 + np.arange(len(fr.object_labels))], 1)
        yield "ag", (lambda p=fr.image_path: load_image(p)), boxes, labels, pairs
    for i, im in enumerate(iter_split(Path(args.vg_root), "test")):
        if i >= args.frames:
            break
        yield "vg", im.image, im.boxes, im.labels, np.unique(im.relations[:, :2], axis=0)


class Tally:
    def __init__(self):
        self.gt = self.hit = self.agnostic = self.pairs = self.pair_hit = self.images = 0
        self.dets = 0

    def add(self, packed, boxes, labels, pairs):
        det = packed.cpu()
        iou = box_iou(torch.from_numpy(boxes), det[:, :4])
        same = torch.from_numpy(labels)[:, None] == det[None, :, 5].long()
        match = (iou >= 0.5) & same
        found = match.any(dim=1).numpy()
        self.images += 1
        self.dets += len(det)
        self.gt += len(boxes)
        self.hit += int(found.sum())
        self.agnostic += int((iou >= 0.5).any(dim=1).sum())
        self.pairs += len(pairs)
        self.pair_hit += int((found[pairs[:, 0]] & found[pairs[:, 1]]).sum()) if len(pairs) else 0

    def summary(self):
        return {"images": self.images, "gt_boxes": self.gt, "object_recall": self.hit / max(self.gt, 1),
                "class_agnostic_recall": self.agnostic / max(self.gt, 1),
                "pair_ceiling": self.pair_hit / max(self.pairs, 1),
                "mean_detections": self.dets / max(self.images, 1)}


@torch.no_grad()
def run(name: str, items: list, vg_classes) -> dict:
    model, processor = load_owlv2(name, device="cuda", dtype=torch.float16)
    size = model.config.vision_config.image_size
    graph = Owlv2DetectionGraph(model).eval()
    pre = GpuOwlv2Preprocessor(size, device="cuda", dtype=torch.float16)
    vocab = {"ag": AG_OBJECT_CLASSES, "vg": vg_classes}
    queries = {}
    for ds, classes in vocab.items():
        prompts, owner = build_prompt_index(classes)
        queries[ds] = (encode_text_queries(model, processor, prompts, "cuda").half(),
                       torch.as_tensor(owner, device="cuda"), len(classes))
    tallies = {"ag": Tally(), "vg": Tally()}
    times = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(load) for _, load, *_ in items[:64]]
        for i, (ds, load, boxes, labels, pairs) in enumerate(tqdm(items, desc=name, mininterval=60)):
            image = futures[i].result()
            if i + 64 < len(items):
                futures.append(pool.submit(items[i + 64][1]))
            futures[i] = None
            w, h = image.size
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            query, owner, n_cls = queries[ds]
            logits, pred_boxes, objectness = graph(pre([image]), query)
            thr, nms, top = SETTINGS[ds]
            packed = postprocess_device(logits, pred_boxes, objectness, owner, n_cls, (w, h), thr, top, nms)
            sides = packed[:, 2:4] - packed[:, 0:2]
            packed = packed[(sides >= 1.0).all(dim=1)][:top]
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            tallies[ds].add(packed, boxes, labels, pairs)
    del model, graph
    torch.cuda.empty_cache()
    out = {ds: t.summary() for ds, t in tallies.items()}
    out["image_size"] = size
    out["eager_fp16_ms_median"] = 1000 * float(np.median(times[20:]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--ag-root", default="acgdataset")
    ap.add_argument("--vg-root", default="/workspace/vg150")
    ap.add_argument("--frames", type=int, default=2000, help="per dataset")
    ap.add_argument("--models", nargs="+", default=["base", "large"])
    args = ap.parse_args()
    ws = Workspace(Path(args.artifacts))
    vg_classes, _ = vocabulary(Path(args.vg_root))
    items = list(samples(args))
    report = {"frames_per_dataset": args.frames, "settings": {k: dict(zip(("score", "nms", "top"), v))
                                                            for k, v in SETTINGS.items()}}
    for name in args.models:
        report[name] = r = run(name, items, vg_classes)
        for ds in ("ag", "vg"):
            s = r[ds]
            print(f"{name:<6} {ds}: object recall {s['object_recall']:.3f}  class-agnostic "
                  f"{s['class_agnostic_recall']:.3f}  pair ceiling {s['pair_ceiling']:.3f}  "
                  f"({s['images']} images, {s['mean_detections']:.1f} dets)", flush=True)
        print(f"{name:<6} {r['image_size']}px eager fp16 {r['eager_fp16_ms_median']:.1f} ms/image", flush=True)
    write_report(report, ws.result("detector_smoke.json"))
    print("DETECTOR_SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
