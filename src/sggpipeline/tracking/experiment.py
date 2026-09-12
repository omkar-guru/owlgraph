"""CPU-first training and fixed-detection identity controls.

Run with ``python -m sggpipeline.tracking --help``. Metrics are conditional on
trusted correspondences already matched to detection rows, not full MOT scores.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from .cache import DetectionCache, load_caches
from .identity import IdentityHead, identity_loss
from .tracker import AssociationTracker, TrackerConfig


def train_identity(caches: list[DetectionCache], *, steps: int = 500,
                   tracks_per_batch: int = 16, learning_rate: float = 1e-3,
                   hidden_dim: int = 256, output_dim: int = 128, seed: int = 0) -> tuple[IdentityHead, dict]:
    """Train on reviewed train-split correspondences, using same-class negatives.

    Each batch samples multiple distinct instances of one class, with two
    different frames per instance. This prevents category-only discrimination
    from solving the task. Cached visual features always remain frozen.
    """
    if steps <= 0 or tracks_per_batch < 2 or not 0 < learning_rate < float("inf"):
        raise ValueError("Need positive steps/rate and at least two tracks per batch")
    groups: dict[tuple[int, int, int], dict[int, int]] = {}
    features, instances, frames = [], [], []
    instance_index: dict[tuple[int, int], int] = {}
    row_offset = frame_offset = 0
    for video, cache in enumerate(caches):
        cache.validate(require_identity=True)
        if cache.metadata["split"] != "train":
            raise ValueError("Identity training accepts only train-split videos")
        features.append(cache.features)
        for frame, (start, end) in enumerate(zip(cache.offsets[:-1], cache.offsets[1:], strict=True)):
            for row in range(int(start), int(end)):
                track_id, label = int(cache.instance_ids[row]), int(cache.labels[row])
                global_id = -1
                if track_id >= 0:
                    key = (video, track_id)
                    global_id = instance_index.setdefault(key, len(instance_index))
                    groups.setdefault((video, track_id, label), {})[frame] = row_offset + row
                instances.append(global_id)
                frames.append(frame_offset + frame)
        row_offset += len(cache.features)
        frame_offset += len(cache.timestamps)
    by_class: dict[int, list[list[int]]] = {}
    for (_, _, label), observations in groups.items():
        if len(observations) >= 2:
            by_class.setdefault(label, []).append(list(observations.values()))
    by_class = {label: tracks for label, tracks in by_class.items() if len(tracks) >= 2}
    if not by_class:
        raise ValueError("Need at least two same-class instances with two labelled frames each")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(np.concatenate(features).astype(np.float32))
    instance_tensor, frame_tensor = torch.tensor(instances), torch.tensor(frames)
    model = IdentityHead(x.shape[1], hidden_dim, output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    start_time = time.perf_counter()
    for _ in range(steps):
        label = int(rng.choice(list(by_class)))
        tracks = by_class[label]
        chosen = rng.choice(len(tracks), size=min(tracks_per_batch, len(tracks)), replace=False)
        rows = np.concatenate([rng.choice(tracks[i], size=2, replace=False) for i in chosen])
        optimizer.zero_grad(set_to_none=True)
        loss = identity_loss(model(x[rows]), instance_tensor[rows], frame_tensor[rows])
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite identity loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return model.eval(), {
        "steps": steps, "tracks_per_batch": tracks_per_batch, "learning_rate": learning_rate,
        "seed": seed, "train_seconds": time.perf_counter() - start_time,
        "first_loss": losses[0], "last_loss": losses[-1],
        "eligible_class_count": len(by_class),
        "eligible_instance_class_groups": sum(map(len, by_class.values())),
        "training_video_ids": [c.metadata["video_id"] for c in caches],
        "identity_sources": sorted({c.metadata["identity_source"] for c in caches}),
        "feature_source": caches[0].metadata["feature_source"],
    }


def association_metrics(cache: DetectionCache, predicted_ids: np.ndarray) -> dict:
    """Count ID changes and cross-instance reuse on known detection rows only.

    Switches compare to the last labelled observation (including gaps).
    Cross-instance transfers count a predicted ID changing its known physical
    owner. Fragmentation counts additional predicted IDs per true instance.
    """
    if predicted_ids.shape != cache.instance_ids.shape:
        raise ValueError("Predictions must align with cache detection rows")
    last_prediction: dict[int, int] = {}
    last_owner: dict[int, int] = {}
    identities: dict[int, set[int]] = {}
    switches = transfers = links = observations = 0
    for truth, predicted in zip(cache.instance_ids, predicted_ids, strict=True):
        truth, predicted = int(truth), int(predicted)
        if truth < 0:
            continue
        observations += 1
        if truth in last_prediction:
            links += 1
            switches += last_prediction[truth] != predicted
        if predicted in last_owner:
            transfers += last_owner[predicted] != truth
        last_prediction[truth], last_owner[predicted] = predicted, truth
        identities.setdefault(truth, set()).add(predicted)
    return {"labelled_observations": observations, "identity_links": links,
            "id_switches": switches, "cross_instance_transfers": transfers,
            "extra_ids_per_instance_total": sum(len(ids) - 1 for ids in identities.values()),
            "id_switch_rate": switches / links if links else None}


def compare(caches: list[DetectionCache], config: TrackerConfig,
            model: IdentityHead | None = None) -> dict:
    """Replay identical detections with frozen appearance and learned identity."""
    if model is not None:
        model.eval()
    result = {"config": asdict(config), "metrics_scope": "association conditional on labelled detection rows",
              "timing_scope": "CPU descriptor transform and association; excludes detection and cache I/O",
              "variants": {}}
    for name in (["appearance", "identity"] if model is not None else ["appearance"]):
        videos, latencies = [], []
        for cache in caches:
            tracker = AssociationTracker(config)
            ids, uncertainty, expired_count = [], 0, 0
            for f, (start, end) in enumerate(zip(cache.offsets[:-1], cache.offsets[1:], strict=True)):
                detections = SimpleNamespace(boxes=cache.boxes[start:end], labels=cache.labels[start:end])
                features = cache.features[start:end]
                tick = time.perf_counter()
                if name == "identity":
                    with torch.inference_mode():
                        features = model(torch.from_numpy(features.astype(np.float32))).numpy()
                tracked = tracker.update(detections, features, float(cache.timestamps[f]))
                latencies.append((time.perf_counter() - tick) * 1000)
                ids.extend(tracked.track_ids.tolist())
                uncertainty += int(tracked.uncertain.sum())
                expired_count += len(tracked.expired_ids)
            videos.append({"video_id": cache.metadata["video_id"], "split": cache.metadata["split"],
                           "identity_source": cache.metadata["identity_source"],
                           "metrics": association_metrics(cache, np.asarray(ids, dtype=np.int64)),
                           "uncertain_observations": uncertainty, "retired_tracks": expired_count,
                           "track_ids": ids})
        totals = {key: sum(v["metrics"][key] for v in videos) for key in
                  ("labelled_observations", "identity_links", "id_switches",
                   "cross_instance_transfers", "extra_ids_per_instance_total")}
        totals["id_switch_rate"] = totals["id_switches"] / totals["identity_links"] if totals["identity_links"] else None
        result["variants"][name] = {"totals": totals, "videos": videos,
                                    "median_frame_ms": float(np.median(latencies)),
                                    "p95_frame_ms": float(np.percentile(latencies, 95))}
    return result


def _fingerprint(path: str) -> dict:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train identity adapter on trusted train caches (CPU)")
    train.add_argument("--caches", nargs="+", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--steps", type=int, default=500)
    train.add_argument("--tracks-per-batch", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--output-dim", type=int, default=128)
    train.add_argument("--seed", type=int, default=0)
    evaluate = sub.add_parser("evaluate", help="Compare fixed-detection tracking on held-out caches")
    evaluate.add_argument("--caches", nargs="+", required=True)
    evaluate.add_argument("--checkpoint", help="Omit for appearance baseline only")
    evaluate.add_argument("--config", help="JSON mapping of TrackerConfig settings, fixed across variants")
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    caches = load_caches(args.caches, require_identity=True)
    output = Path(args.output)
    protected = [Path(p).resolve() for p in args.caches]
    if args.command == "evaluate":
        protected += [Path(p).resolve() for p in (args.checkpoint, args.config) if p]
    if output.resolve() in protected:
        raise ValueError("Output must not overwrite an input")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        model, report = train_identity(caches, steps=args.steps, tracks_per_batch=args.tracks_per_batch,
                                       learning_rate=args.learning_rate, hidden_dim=args.hidden_dim,
                                       output_dim=args.output_dim, seed=args.seed)
        report["caches"] = [_fingerprint(p) for p in args.caches]
        torch.save({"schema_version": 1, "dimensions": model.dimensions,
                    "state_dict": model.state_dict(), "training": report}, output)
        print(json.dumps(report, indent=2))
    else:
        if any(c.metadata["split"] == "train" for c in caches):
            raise ValueError("Evaluate on held-out val or test videos")
        if len({c.metadata["split"] for c in caches}) != 1:
            raise ValueError("Do not pool validation and test results")
        config = TrackerConfig(**json.loads(Path(args.config).read_text())) if args.config else TrackerConfig()
        model = None
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if checkpoint["schema_version"] != 1:
                raise ValueError("Unsupported identity checkpoint version")
            training = checkpoint["training"]
            if set(training["training_video_ids"]) & {c.metadata["video_id"] for c in caches}:
                raise ValueError("Evaluation videos overlap identity training videos")
            if training["feature_source"] != caches[0].metadata["feature_source"]:
                raise ValueError("Checkpoint feature extractor does not match evaluation cache")
            if checkpoint["dimensions"]["input_dim"] != caches[0].features.shape[1]:
                raise ValueError("Checkpoint feature dimension does not match evaluation cache")
            model = IdentityHead(**checkpoint["dimensions"])
            model.load_state_dict(checkpoint["state_dict"])
        report = compare(caches, config, model)
        report["caches"] = [_fingerprint(p) for p in args.caches]
        report["checkpoint"] = _fingerprint(args.checkpoint) if args.checkpoint else None
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({name: value["totals"] for name, value in report["variants"].items()}, indent=2))
