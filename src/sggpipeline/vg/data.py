"""VG150 from the Hugging Face parquet release (``maelic/VG150-coco-format``).

That release is the standard Xu et al. split (150 object classes, 50
predicates) with the canonical Neural-Motifs / Scene-Graph-Benchmark validation
set. Once images without relations are dropped - as the reference loader does
for every split - it gives the canonical 57,723 train / 5,000 val / 26,446 test
images. Object and predicate ids are 1-based and alphabetical, the reference
dictionaries' order; here they are 0-based.

Boxes arrive as COCO xywh and are returned as xyxy (x, y, x + w, y + h), the
pixel-inclusive convention the reference evaluator's +1 IoU expects.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class VGImage:
    image_id: int
    width: int
    height: int
    boxes: np.ndarray  # (G, 4) xyxy
    labels: np.ndarray  # (G,) 0-based object class
    relations: np.ndarray  # (M, 3) subject index, object index, 0-based predicate
    image_bytes: bytes | None = None

    def image(self):
        from PIL import Image

        return Image.open(io.BytesIO(self.image_bytes)).convert("RGB")


def vocabulary(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(object classes, predicates), 0-based, in the reference order."""
    meta = json.loads((Path(root) / "categories.json").read_text())
    objects = sorted(meta["categories"], key=lambda c: c["id"])
    predicates = sorted(meta["rel_categories"], key=lambda c: c["id"])
    assert [c["id"] for c in objects] == list(range(1, len(objects) + 1))
    assert [c["id"] for c in predicates] == list(range(1, len(predicates) + 1))
    return tuple(c["name"] for c in objects), tuple(c["name"] for c in predicates)


def iter_split(root: Path, split: str, images: bool = True, relations_only: bool = True,
               batch_size: int = 64) -> Iterator[VGImage]:
    """Images of one split in file order; by default only those with relations."""
    import pyarrow.parquet as pq

    columns = ["image_id", "width", "height", "objects", "relations"] + (["image"] if images else [])
    for path in sorted((Path(root) / "data").glob(f"{split}-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                if relations_only and not row["relations"]:
                    continue
                objects = row["objects"]
                index = {o["id"]: i for i, o in enumerate(objects)}
                xywh = np.array([o["bbox"] for o in objects], np.float32).reshape(-1, 4)
                boxes = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], axis=1)
                labels = np.array([o["category_id"] - 1 for o in objects], np.int64)
                rels = np.array([(index[r["subject_id"]], index[r["object_id"]], r["predicate_id"] - 1)
                                 for r in row["relations"]], np.int64).reshape(-1, 3)
                yield VGImage(row["image_id"], row["width"], row["height"], boxes, labels, rels,
                              row["image"]["bytes"] if images else None)
