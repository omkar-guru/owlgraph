"""Loader for Action Genome object-detection ground truth.

AG annotates a sparse set of keyframes from Charades videos.  Two pickles carry
everything this stage needs:

``person_bbox.pkl``
    ``{frame_key: {"bbox": (N,4) xyxy, "bbox_size": (w,h), ...}}``.  Person boxes
    come from a detector run at ``bbox_size``, which is **not** always the native
    frame size, so they are rescaled to the real image on load.

``object_bbox_and_relationship.pkl``
    ``{frame_key: [{"class": str, "bbox": (x,y,w,h)|None, "visible": bool,
    "metadata": {"set": "train"|"test"}, ...}]}``.  Object boxes are xywh in
    native frame coordinates and are absent when the object is not visible.

``frame_key`` looks like ``"001YG.mp4/000089.png"``.

Nothing here assumes the layout is correct: :func:`ActionGenome.validate`
inspects the real files and reports what it actually found.
"""

from __future__ import annotations

import pickle
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classes import AG_OBJECT_CLASSES


@dataclass(slots=True)
class AGFrame:
    """One annotated keyframe with its ground-truth boxes."""

    frame_key: str
    video_id: str
    frame_index: int
    boxes: np.ndarray  # (K, 4) float32, xyxy, native frame pixels
    labels: np.ndarray  # (K,) int64, index into the class vocabulary
    image_path: Path | None = None
    width: int | None = None
    height: int | None = None

    @property
    def num_boxes(self) -> int:
        return int(self.boxes.shape[0])


@dataclass
class ActionGenome:
    """Indexes AG keyframes and serves ground truth in a detector-friendly form."""

    root: Path
    split: str = "test"
    include_person: bool = True
    frames_dirname: str = "frames"
    # The AG v1.0 download unpacks to a version-named directory rather than
    # "annotations"; both layouts are accepted (see _resolve_annotation_dir).
    annotations_dirname: str = "action_genome_v1.0"
    classes: tuple[str, ...] = AG_OBJECT_CLASSES
    frames: list[AGFrame] = field(default_factory=list, repr=False)
    _class_to_idx: dict[str, int] = field(default_factory=dict, repr=False)

    # -- paths ---------------------------------------------------------------
    @property
    def annotation_dir(self) -> Path:
        """Locate the annotations, tolerating either published directory name."""
        candidates = [
            self.root / self.annotations_dirname,
            self.root / "annotations",
            self.root / "action_genome_v1.0",
            self.root,
        ]
        for candidate in candidates:
            if (candidate / "person_bbox.pkl").exists():
                return candidate
        return self.root / self.annotations_dirname

    @property
    def frames_dir(self) -> Path:
        return self.root / self.frames_dirname

    @property
    def person_pkl(self) -> Path:
        return self.annotation_dir / "person_bbox.pkl"

    @property
    def object_pkl(self) -> Path:
        return self.annotation_dir / "object_bbox_and_relationship.pkl"

    # -- loading -------------------------------------------------------------
    def load(self, max_frames: int | None = None) -> ActionGenome:
        """Read both annotation pickles and build the keyframe index."""
        for path in (self.person_pkl, self.object_pkl):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing AG annotation file: {path}\n"
                    "Expected the AG 'annotations' directory under --ag-root."
                )

        with self.person_pkl.open("rb") as fh:
            person = pickle.load(fh)
        with self.object_pkl.open("rb") as fh:
            objects = pickle.load(fh)

        self.classes = self._resolve_vocabulary(objects)
        self._class_to_idx = {name: i for i, name in enumerate(self.classes)}

        self.frames = []
        for frame_key in sorted(objects.keys()):
            entries = objects[frame_key]
            if not self._in_split(entries):
                continue
            frame = self._build_frame(frame_key, entries, person.get(frame_key))
            if frame is not None and frame.num_boxes > 0:
                self.frames.append(frame)
            if max_frames is not None and len(self.frames) >= max_frames:
                break
        return self

    def _in_split(self, entries: list[dict]) -> bool:
        for entry in entries:
            meta = entry.get("metadata") or {}
            if meta.get("set") is not None:
                return str(meta["set"]) == self.split
        return False

    def _resolve_vocabulary(self, objects: dict) -> tuple[str, ...]:
        """Derive the label set from the annotations, cross-checking the constant."""
        found = {
            str(entry["class"])
            for entries in objects.values()
            for entry in entries
            if entry.get("class")
        }
        if self.include_person:
            found.add("person")

        expected = set(self.classes)
        if found != expected:
            missing = sorted(expected - found)
            extra = sorted(found - expected)
            warnings.warn(
                "AG vocabulary differs from the hardcoded list; using the "
                f"annotations. Not present in data: {missing}. "
                f"Unexpected in data: {extra}.",
                stacklevel=2,
            )
        # Preserve the canonical order for classes we know, append novelties.
        ordered = [c for c in self.classes if c in found]
        ordered += sorted(found - set(ordered))
        return tuple(ordered)

    def _build_frame(
        self, frame_key: str, entries: list[dict], person_rec: dict | None
    ) -> AGFrame | None:
        video_id, _, frame_name = frame_key.partition("/")
        try:
            frame_index = int(Path(frame_name).stem)
        except ValueError:
            frame_index = -1

        image_path = self.frames_dir / frame_key
        boxes: list[np.ndarray] = []
        labels: list[int] = []

        for entry in entries:
            if not entry.get("visible", False):
                continue
            bbox = entry.get("bbox")
            name = entry.get("class")
            if bbox is None or name not in self._class_to_idx:
                continue
            x, y, w, h = (float(v) for v in np.asarray(bbox).reshape(-1)[:4])
            if w <= 0 or h <= 0:
                continue
            boxes.append(np.array([x, y, x + w, y + h], dtype=np.float32))
            labels.append(self._class_to_idx[name])

        width = height = None
        if self.include_person and person_rec is not None:
            pboxes = np.asarray(person_rec.get("bbox", []), dtype=np.float32)
            pboxes = pboxes.reshape(-1, 4)
            if pboxes.size:
                # Person boxes live in the detector's coordinate frame; rescale
                # them if that differs from the frame the object boxes use.
                size = person_rec.get("bbox_size")
                scale = self._person_scale(size, image_path)
                if scale is not None:
                    pboxes = pboxes * np.array(
                        [scale[0], scale[1], scale[0], scale[1]], dtype=np.float32
                    )
                person_idx = self._class_to_idx.get("person")
                if person_idx is not None:
                    for pb in pboxes:
                        boxes.append(pb.astype(np.float32))
                        labels.append(person_idx)

        if not boxes:
            return None
        return AGFrame(
            frame_key=frame_key,
            video_id=video_id,
            frame_index=frame_index,
            boxes=np.stack(boxes).astype(np.float32),
            labels=np.asarray(labels, dtype=np.int64),
            image_path=image_path if image_path.exists() else None,
            width=width,
            height=height,
        )

    def _person_scale(
        self, bbox_size, image_path: Path
    ) -> tuple[float, float] | None:
        """Ratio that maps detector coordinates onto native frame coordinates."""
        if bbox_size is None or not image_path.exists():
            return None
        try:
            from PIL import Image

            with Image.open(image_path) as im:
                real_w, real_h = im.size
        except Exception:
            return None
        det_w, det_h = (float(v) for v in np.asarray(bbox_size).reshape(-1)[:2])
        if det_w <= 0 or det_h <= 0:
            return None
        if abs(det_w - real_w) < 1 and abs(det_h - real_h) < 1:
            return None
        return real_w / det_w, real_h / det_h

    # -- diagnostics ---------------------------------------------------------
    def validate(self, sample: int = 25) -> dict:
        """Report what the annotations actually contain, for eyeballing."""
        if not self.frames:
            raise RuntimeError("Call load() before validate().")

        label_counts = Counter()
        boxes_per_frame = []
        out_of_bounds = 0
        checked = 0
        missing_images = 0

        for frame in self.frames:
            label_counts.update(self.classes[i] for i in frame.labels)
            boxes_per_frame.append(frame.num_boxes)
            if frame.image_path is None:
                missing_images += 1

        for frame in self.frames[:sample]:
            if frame.image_path is None:
                continue
            from PIL import Image

            with Image.open(frame.image_path) as im:
                w, h = im.size
            checked += 1
            b = frame.boxes
            if (b[:, 0] < -1).any() or (b[:, 1] < -1).any():
                out_of_bounds += 1
            elif (b[:, 2] > w + 1).any() or (b[:, 3] > h + 1).any():
                out_of_bounds += 1

        return {
            "split": self.split,
            "num_frames": len(self.frames),
            "num_classes": len(self.classes),
            "classes": list(self.classes),
            "total_boxes": int(sum(boxes_per_frame)),
            "boxes_per_frame_mean": float(np.mean(boxes_per_frame)),
            "boxes_per_frame_max": int(np.max(boxes_per_frame)),
            "frames_missing_image": missing_images,
            "geometry_checked": checked,
            "frames_with_out_of_bounds_boxes": out_of_bounds,
            "label_counts": dict(label_counts.most_common()),
        }

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> AGFrame:
        return self.frames[idx]
