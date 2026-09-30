"""Action Genome person-object relationships, in a form a pair head can train on.

Every visible object in an AG keyframe is related to the frame's person by up to
three kinds of predicate, stored per object entry:

``attention``   exactly one of 3 (looking at, not looking at, unsure)
``spatial``     one or more of 6
``contacting``  one or more of 17

The subject is always the person, whose box comes from ``person_bbox.pkl`` (at
most one person per annotated frame). Predicate order follows
``relationship_classes.txt``; the pickles spell names with underscores.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classes import AG_OBJECT_CLASSES

ATTENTION = ("looking_at", "not_looking_at", "unsure")
SPATIAL = ("above", "beneath", "in_front_of", "behind", "on_the_side_of", "in")
CONTACTING = ("carrying", "covered_by", "drinking_from", "eating", "have_it_on_the_back",
              "holding", "leaning_on", "lying_on", "not_contacting", "other_relationship",
              "sitting_on", "standing_on", "touching", "twisting", "wearing", "wiping",
              "writing_on")
PREDICATES = ATTENTION + SPATIAL + CONTACTING  # the 26 in relationship_classes.txt order


@dataclass(slots=True)
class RelationFrame:
    frame_key: str
    video_id: str
    image_path: Path
    person_box: np.ndarray  # (4,) xyxy, native pixels
    object_boxes: np.ndarray  # (K, 4) xyxy
    object_labels: np.ndarray  # (K,) index into AG_OBJECT_CLASSES
    attention: np.ndarray  # (K,) index into ATTENTION
    spatial: np.ndarray  # (K, 6) bool
    contacting: np.ndarray  # (K, 17) bool


def _multi_hot(names, vocab) -> np.ndarray:
    out = np.zeros(len(vocab), dtype=bool)
    for name in names or ():
        out[vocab.index(name)] = True
    return out


def load_relation_frames(ag_root: Path, split: str, frames_dirname: str = "frames",
                         annotations_dirname: str = "action_genome_v1.0") -> list[RelationFrame]:
    """Keyframes of ``split`` with a person and at least one labelled visible object.

    Frames whose image has not been extracted are skipped, so callers see only
    what they can actually run.
    """
    root = Path(ag_root)
    ann = root / annotations_dirname
    with (ann / "person_bbox.pkl").open("rb") as fh:
        people = pickle.load(fh)
    with (ann / "object_bbox_and_relationship.pkl").open("rb") as fh:
        objects = pickle.load(fh)
    class_index = {name: i for i, name in enumerate(AG_OBJECT_CLASSES)}

    frames = []
    for key in sorted(objects):
        entries = objects[key]
        if not any((e.get("metadata") or {}).get("set") == split for e in entries):
            continue
        person = np.asarray(people.get(key, {}).get("bbox", []), dtype=np.float32).reshape(-1, 4)
        if len(person) != 1:
            continue
        image_path = root / frames_dirname / key
        if not image_path.exists():
            continue
        boxes, labels, att, spa, con = [], [], [], [], []
        for e in entries:
            if not e.get("visible") or e.get("bbox") is None or not e.get("attention_relationship"):
                continue
            x, y, w, h = (float(v) for v in np.asarray(e["bbox"]).reshape(-1)[:4])
            if w <= 0 or h <= 0 or e["class"] not in class_index:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(class_index[e["class"]])
            att.append(ATTENTION.index(e["attention_relationship"][0]))
            spa.append(_multi_hot(e.get("spatial_relationship"), SPATIAL))
            con.append(_multi_hot(e.get("contacting_relationship"), CONTACTING))
        if not boxes:
            continue
        frames.append(RelationFrame(
            frame_key=key, video_id=key.split("/")[0], image_path=image_path,
            person_box=person[0], object_boxes=np.asarray(boxes, dtype=np.float32),
            object_labels=np.asarray(labels, dtype=np.int64),
            attention=np.asarray(att, dtype=np.int64),
            spatial=np.stack(spa), contacting=np.stack(con)))
    return frames
