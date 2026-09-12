"""Versioned, pickle-free exchange format for fixed Stage 1 detections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tracker import normalize


@dataclass
class DetectionCache:
    metadata: dict
    timestamps: np.ndarray  # (F,), seconds, including frames with no detections
    offsets: np.ndarray  # (F+1,), frame f occupies [offsets[f]:offsets[f+1]]
    boxes: np.ndarray  # (N,4), native xyxy pixels, stable image coordinate system
    scores: np.ndarray
    labels: np.ndarray
    objectness: np.ndarray
    features: np.ndarray  # (N,D), frozen visual features in detection order
    instance_ids: np.ndarray  # (N,), trusted ID scoped to video, -1 unknown

    def validate(self, require_identity: bool = False) -> None:
        required = {"schema_version", "video_id", "split", "feature_source", "identity_source"}
        if not required <= self.metadata.keys() or self.metadata["schema_version"] != 1:
            raise ValueError("Cache requires version 1 metadata: " + ", ".join(sorted(required)))
        if self.metadata["split"] not in {"train", "val", "test"}:
            raise ValueError("Cache split must be train, val or test")
        if not all(isinstance(self.metadata[k], str) and self.metadata[k] for k in required - {"schema_version"}):
            raise ValueError("Metadata names must be nonempty strings")
        if self.metadata["identity_source"] not in {"none", "human", "reviewed_pseudo"}:
            raise ValueError("Identity source must be none, human or reviewed_pseudo")
        f, n = len(self.timestamps), len(self.boxes)
        if self.timestamps.shape != (f,) or f == 0 or not np.isfinite(self.timestamps).all() or (np.diff(self.timestamps) <= 0).any():
            raise ValueError("Cache needs at least one strictly ordered finite timestamp")
        if (self.offsets.shape != (f + 1,) or self.offsets.dtype.kind not in "iu"
                or self.offsets[0] != 0 or self.offsets[-1] != n or (np.diff(self.offsets) < 0).any()):
            raise ValueError("Invalid frame offsets")
        if self.boxes.shape != (n, 4) or not np.isfinite(self.boxes).all() or (self.boxes[:, 2:] <= self.boxes[:, :2]).any():
            raise ValueError("Boxes must be finite positive-area xyxy")
        for name in ("scores", "labels", "objectness", "instance_ids"):
            value = getattr(self, name)
            if value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must have one finite value per detection")
        for name in ("labels", "instance_ids"):
            if getattr(self, name).dtype.kind not in "iu":
                raise ValueError(f"{name} must contain integers")
        if (self.labels < 0).any() or (self.instance_ids < -1).any():
            raise ValueError("Labels must be nonnegative and instance IDs >= -1")
        for name in ("scores", "objectness"):
            if ((getattr(self, name) < 0) | (getattr(self, name) > 1)).any():
                raise ValueError(f"{name} must be probabilities in [0,1]")
        normalize(self.features)
        if len(self.features) != n:
            raise ValueError("Features must align with detection rows")
        if self.metadata["identity_source"] == "none" and (self.instance_ids >= 0).any():
            raise ValueError("Unreviewed tracks cannot supply trusted instance IDs")
        if require_identity and (self.metadata["identity_source"] == "none" or not (self.instance_ids >= 0).any()):
            raise ValueError("Trusted instance correspondences are required")
        for start, end in zip(self.offsets[:-1], self.offsets[1:], strict=True):
            known = self.instance_ids[start:end]
            known = known[known >= 0]
            if len(np.unique(known)) != len(known):
                raise ValueError("An instance may match at most one detection per frame; leave duplicate proposals unknown")

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(handle, metadata=np.array(json.dumps(self.metadata)),
                                **{k: v for k, v in vars(self).items() if k != "metadata"})

    @classmethod
    def load(cls, path: str | Path) -> DetectionCache:
        with np.load(path, allow_pickle=False) as data:
            cache = cls(metadata=json.loads(str(data["metadata"].item())),
                        **{name: data[name] for name in cls.__dataclass_fields__ if name != "metadata"})
        cache.validate()
        return cache


def load_caches(paths: list[str | Path], require_identity: bool = False) -> list[DetectionCache]:
    caches = [DetectionCache.load(p) for p in paths]
    if not caches:
        raise ValueError("At least one video cache is required")
    videos = [c.metadata["video_id"] for c in caches]
    if len(set(videos)) != len(videos):
        raise ValueError("Each video must have exactly one cache")
    for cache in caches:
        cache.validate(require_identity=require_identity)
    if len({c.features.shape[1] for c in caches}) != 1 or len({c.metadata["feature_source"] for c in caches}) != 1:
        raise ValueError("Caches must share the same feature extractor and dimension")
    return caches
