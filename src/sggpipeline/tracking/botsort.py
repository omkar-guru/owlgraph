"""BoT-SORT driven exclusively by supplied OWL visual features.

Uses Ultralytics' native-feature pass-through. No YOLO detector or separate
ReID network is constructed. Tested with ultralytics 8.4.165.
"""

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from threading import RLock
from types import SimpleNamespace

import numpy as np

from .tracker import TrackingResult, normalize

_ID_LOCK = RLock()  # Upstream BaseTrack's allocator is process-global.


@dataclass(frozen=True)
class BoTSORTConfig:
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30  # number of updates, NOT seconds
    match_thresh: float = 0.8
    fuse_score: bool = True
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.8

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name not in {"track_buffer", "fuse_score"} and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be in [0,1]")
        if not self.track_low_thresh < self.track_high_thresh <= self.new_track_thresh:
            raise ValueError("Require low < high <= new-track confidence thresholds")
        if isinstance(self.track_buffer, bool) or not isinstance(self.track_buffer, int) or self.track_buffer < 1:
            raise ValueError("track_buffer must be a positive integer number of updates")
        if not isinstance(self.fuse_score, bool):
            raise ValueError("fuse_score must be boolean")


@dataclass
class _Rows:
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, key):
        return _Rows(self.xyxy[key], self.conf[key], self.cls[key])

    @property
    def xywh(self):
        return np.concatenate(((self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2,
                               self.xyxy[:, 2:] - self.xyxy[:, :2]), axis=1)


class OWLBoTSORT:
    """Track fixed OWL detections using one supplied descriptor per detection.

    Returns IDs in input order; -1 means filtered or not yet confirmed.
    Camera-motion compensation is disabled because this path needs no images.
    Regular frame spacing is required, including empty detection frames. This
    adapter checks cadence rather than pretending the upstream Kalman filter
    uses elapsed seconds. It does not provide ambiguity estimates.
    """

    def __init__(self, config: BoTSORTConfig | None = None):
        if find_spec("lap") is None:
            # Prevent Ultralytics' optional-dependency auto-installer at import.
            raise ImportError("BoT-SORT needs lap; install the project's tracking extra (uv sync --extra tracking)")
        from ultralytics.trackers.basetrack import BaseTrack, TrackState
        from ultralytics.trackers.bot_sort import BOTSORT
        import ultralytics

        self.config = config or BoTSORTConfig()
        self.backend_version = ultralytics.__version__
        self._base_track = BaseTrack
        self._removed_state = TrackState.Removed
        args = SimpleNamespace(**asdict(self.config), with_reid=True, model="auto", gmc_method="none")
        with _ID_LOCK:
            saved = BaseTrack._count
            try:
                self._backend = BOTSORT(args)
            finally:
                BaseTrack._count = saved
        self.reset()

    def reset(self):
        with _ID_LOCK:
            saved = self._base_track._count
            try:
                self._backend.reset()
            finally:
                self._base_track._count = saved
        self._counter = 0
        self._last_timestamp = None
        self._interval = None
        self._feature_dim = None
        self._emitted = set()

    def update(self, detections, features: np.ndarray, timestamp: float) -> TrackingResult:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or (self._last_timestamp is not None and timestamp <= self._last_timestamp):
            raise ValueError("Timestamps must be finite and strictly increasing")
        interval = None if self._last_timestamp is None else timestamp - self._last_timestamp
        if self._interval is not None and not np.isclose(interval, self._interval, rtol=1e-3, atol=1e-6):
            raise ValueError("BoT-SORT requires regularly spaced frames; include empty frames or use the timestamp-aware baseline")
        features = normalize(features)
        n = len(features)
        boxes = np.asarray(detections.boxes, dtype=np.float32)
        scores = np.asarray(detections.scores, dtype=np.float32)
        labels = np.asarray(detections.labels)
        if boxes.shape != (n, 4) or scores.shape != (n,) or labels.shape != (n,):
            raise ValueError("Boxes, scores, labels and features must align by detection row")
        if not np.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
            raise ValueError("Boxes must be finite positive-area xyxy")
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Scores must be probabilities in [0,1]")
        if labels.dtype.kind not in "iu" or (labels < 0).any():
            raise ValueError("Labels must be nonnegative integers")
        if self._feature_dim is not None and features.shape[1] != self._feature_dim:
            raise ValueError("Feature dimension changed within a video")
        prior = {t.track_id for t in self._backend.tracked_stracks + self._backend.lost_stracks}
        with _ID_LOCK:
            saved = self._base_track._count
            self._base_track._count = self._counter
            try:
                output = self._backend.update(_Rows(boxes.copy(), scores.copy(), labels.copy()),
                                              img=None, feats=features.copy())
            finally:
                self._counter = self._base_track._count
                self._base_track._count = saved
        self._feature_dim = features.shape[1]
        self._last_timestamp = timestamp
        if interval is not None:
            self._interval = interval
        # Upstream retains newly removed tracks in the lost pool for one update.
        # Prune them now so a retired identity cannot be reactivated next frame.
        self._backend.lost_stracks = [t for t in self._backend.lost_stracks if t.state != self._removed_state]
        live = {t.track_id for t in self._backend.tracked_stracks + self._backend.lost_stracks}
        expired = tuple(sorted(prior - live))
        ids = np.full(n, -1, dtype=np.int64)
        is_new = np.zeros(n, dtype=bool)
        if output.size:
            indices = output[:, 7].astype(np.int64)
            if (indices < 0).any() or (indices >= n).any() or len(np.unique(indices)) != len(indices):
                raise RuntimeError("Upstream BoT-SORT returned invalid detection-row indices")
            ids[indices] = output[:, 4].astype(np.int64)
            is_new[indices] = [int(key) not in self._emitted for key in ids[indices]]
        self._emitted.update(int(key) for key in ids if key >= 0)
        self._emitted.intersection_update(live)
        return TrackingResult(ids, is_new, np.zeros(n, dtype=bool), expired)
