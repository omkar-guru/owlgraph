"""Small causal tracker with global assignment and timestamp-based expiry.

Only observed detections are returned. Retained tracks across a gap are memory,
not evidence of an object or relationship being observed during that gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] == 0 or not np.isfinite(features).all():
        raise ValueError("Features must be a finite (N, D) array with D > 0")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if (norms < 1e-8).any():
        raise ValueError("Appearance descriptors must be nonzero")
    return features / np.maximum(norms, 1e-8)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(hi - lo, 0).prod(axis=-1)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=-1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=-1)
    return intersection / np.maximum(area_a[:, None] + area_b - intersection, 1e-8)


def assign(cost: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost maximum-cardinality gated matching (Hungarian algorithm).

    Private dummy columns permit every row to remain unmatched. Their cost is
    larger than any possible change in the sum of real costs, so valid match
    cardinality takes precedence. No optional SciPy dependency is needed.
    """
    n, m = cost.shape
    if not n or not m:
        return []
    penalty = (n + 1) * (float(np.max(np.abs(cost[valid]))) + 1) if valid.any() else n + 1
    matrix = np.concatenate([np.where(valid, cost, penalty * (n + 2)),
                             np.full((n, n), penalty)], axis=1)
    columns = m + n
    u, v = np.zeros(n + 1), np.zeros(columns + 1)
    p, way = np.zeros(columns + 1, dtype=int), np.zeros(columns + 1, dtype=int)
    for row in range(1, n + 1):
        p[0] = row
        j0 = 0
        minimum = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, columns + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minimum[j]:
                    minimum[j], way[j] = cur, j0
                if minimum[j] < delta:
                    delta, j1 = minimum[j], j
            u[p[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return [(int(p[j] - 1), j - 1) for j in range(1, m + 1)
            if p[j] and valid[p[j] - 1, j - 1]]


@dataclass(frozen=True)
class TrackerConfig:
    max_age_seconds: float = 2.0
    max_center_distance: float = 3.0  # in predicted box diagonals
    max_appearance_distance: float = 0.5  # 1 - cosine similarity
    max_cost: float = 0.7
    appearance_weight: float = 0.5
    motion_weight: float = 0.25
    descriptor_momentum: float = 0.8
    ambiguity_margin: float = 0.03
    class_gate: bool = False  # category flicker should not automatically change ID

    def __post_init__(self):
        values = [self.max_age_seconds, self.max_center_distance,
                  self.max_appearance_distance, self.max_cost,
                  self.appearance_weight, self.motion_weight,
                  self.descriptor_momentum, self.ambiguity_margin]
        if not np.isfinite(values).all():
            raise ValueError("Tracker settings must be finite")
        if self.max_age_seconds <= 0 or self.max_center_distance <= 0:
            raise ValueError("Age and center-distance limits must be positive")
        if not 0 <= self.max_appearance_distance <= 2 or not 0 <= self.max_cost <= 1:
            raise ValueError("Invalid appearance or cost threshold")
        if min(self.appearance_weight, self.motion_weight, self.ambiguity_margin) < 0:
            raise ValueError("Weights and ambiguity margin must be nonnegative")
        if self.appearance_weight + self.motion_weight > 1:
            raise ValueError("Appearance and motion weights must sum to at most one")
        if not 0 <= self.descriptor_momentum < 1:
            raise ValueError("Descriptor momentum must be in [0, 1)")


@dataclass
class _Track:
    track_id: int
    box: np.ndarray
    velocity: np.ndarray
    feature: np.ndarray
    label: int
    timestamp: float


@dataclass
class TrackingResult:
    track_ids: np.ndarray  # aligned exactly with input detection rows
    is_new: np.ndarray
    uncertain: np.ndarray  # ambiguous association rejected; fresh ID assigned
    expired_ids: tuple[int, ...]


class AssociationTracker:
    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self.reset()

    def reset(self) -> None:
        """Start another video; IDs are scoped to that video."""
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0
        self._timestamp: float | None = None
        self._feature_dim: int | None = None

    def update(self, detections, features: np.ndarray, timestamp: float) -> TrackingResult:
        """Accept Stage 1's Detections and one appearance vector per row.

        Boxes use original image xyxy pixels; timestamps use seconds. Call on
        empty frames too. Image coordinates must remain consistent within a video.
        Labels and scores are never rewritten by association.
        """
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or (self._timestamp is not None and timestamp <= self._timestamp):
            raise ValueError("Timestamps must be finite and strictly increasing")
        boxes = np.asarray(detections.boxes, dtype=np.float32)
        labels = np.asarray(detections.labels)
        features = normalize(features)
        n = len(features)
        if boxes.shape != (n, 4) or labels.shape != (n,):
            raise ValueError("Boxes, labels and features must have matching detection rows")
        if not np.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
            raise ValueError("Boxes must be finite xyxy with positive area")
        if self._feature_dim is not None and features.shape[1] != self._feature_dim:
            raise ValueError("Feature dimension changed within a video")
        self._feature_dim = features.shape[1]
        self._timestamp = timestamp
        cfg = self.config
        expired = tuple(k for k, t in self._tracks.items() if timestamp - t.timestamp > cfg.max_age_seconds)
        for key in expired:
            del self._tracks[key]
        tracks = list(self._tracks.values())
        ids = np.full(n, -1, dtype=np.int64)
        uncertain = np.zeros(n, dtype=bool)
        is_new = np.ones(n, dtype=bool)
        if tracks and n:
            predicted = np.stack([t.box + t.velocity * (timestamp - t.timestamp) for t in tracks])
            centers = (predicted[:, :2] + predicted[:, 2:]) / 2
            det_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            scale = np.linalg.norm(predicted[:, 2:] - predicted[:, :2], axis=1)
            distance = np.linalg.norm(centers[:, None] - det_centers, axis=-1) / np.maximum(scale[:, None], 1)
            appearance = np.clip(1 - np.stack([t.feature for t in tracks]) @ features.T, 0, 2)
            cost = ((1 - cfg.appearance_weight - cfg.motion_weight) * (1 - box_iou(predicted, boxes))
                    + cfg.motion_weight * np.minimum(distance / cfg.max_center_distance, 1)
                    + cfg.appearance_weight * appearance / 2)
            valid = (distance <= cfg.max_center_distance) & (cost <= cfg.max_cost)
            if cfg.appearance_weight > 0:
                valid &= appearance <= cfg.max_appearance_distance
            if cfg.class_gate:
                valid &= np.array([t.label for t in tracks])[:, None] == labels
            # Reject near-ties on either side before assignment. A new identity
            # avoids silently carrying downstream pair memory across an uncertain match.
            if cfg.ambiguity_margin > 0:
                ambiguous_rows, ambiguous_cols = set(), set()
                for i in range(len(tracks)):
                    options = np.sort(cost[i, valid[i]])
                    if len(options) > 1 and options[1] - options[0] < cfg.ambiguity_margin:
                        ambiguous_rows.add(i)
                for j in range(n):
                    options = np.sort(cost[valid[:, j], j])
                    if len(options) > 1 and options[1] - options[0] < cfg.ambiguity_margin:
                        ambiguous_cols.add(j)
                for i in ambiguous_rows:
                    uncertain |= valid[i]
                for j in ambiguous_cols:
                    uncertain[j] = True
                # Retire every plausible old identity for rejected observations,
                # so it cannot later revive with stale pair state.
                retired_rows = np.flatnonzero(valid[:, uncertain].any(axis=1))
                for i in retired_rows:
                    del self._tracks[tracks[i].track_id]
                expired += tuple(tracks[i].track_id for i in retired_rows)
                valid[retired_rows, :] = False
                valid[:, uncertain] = False
            for i, j in assign(cost, valid):
                track = tracks[i]
                dt = timestamp - track.timestamp
                # Translation-only velocity keeps predicted box dimensions positive.
                shift = ((boxes[j, :2] + boxes[j, 2:]) - (track.box[:2] + track.box[2:])) / (2 * dt)
                track.velocity = np.tile(shift, 2)
                mixed = cfg.descriptor_momentum * track.feature + (1 - cfg.descriptor_momentum) * features[j]
                track.feature = features[j].copy() if np.linalg.norm(mixed) < 1e-8 else normalize(mixed[None])[0]
                track.box, track.label, track.timestamp = boxes[j].copy(), int(labels[j]), timestamp
                ids[j], is_new[j] = track.track_id, False
        for j in np.flatnonzero(is_new):
            key = self._next_id
            self._next_id += 1
            self._tracks[key] = _Track(key, boxes[j].copy(), np.zeros(4, dtype=np.float32),
                                       features[j].copy(), int(labels[j]), timestamp)
            ids[j] = key
        return TrackingResult(ids, is_new, uncertain, expired)
