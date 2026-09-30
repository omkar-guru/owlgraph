"""Relationship inference on a live stream, on top of ``StreamingDetector``.

Reads the detector's GPU-side detections and features directly (no round trip
through the host), keeps the highest-scoring ``max_objects`` detections (the
plan's 32-object budget), routes every ordered pair, classifies the top
``pair_budget`` pairs and brings the result to the host in one copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..ag.classes import AG_OBJECT_CLASSES
from ..ag.relations import PREDICATES
from .model import RelationshipHead, all_pair_geometry, predicate_probabilities
from .sgdet import constrained_mask


@dataclass
class FrameRelations:
    """Classified pairs for one frame; indices refer to the frame's detections."""

    subject: np.ndarray  # (P,) detection index
    object: np.ndarray  # (P,) detection index
    pair_score: np.ndarray  # (P,) router probability
    predicate_probs: np.ndarray  # (P, 26), PREDICATES order

    def triplets(self, detection_scores: np.ndarray, top: int = 50,
                 constrained: bool = True) -> list[tuple[int, int, str, float]]:
        """Highest-scoring (subject, object, predicate, score) triplets.

        Score = router x predicate x subject score x object score, as evaluated.
        """
        if not len(self.subject):
            return []
        probs = torch.from_numpy(self.predicate_probs)
        allowed = constrained_mask(probs).numpy() if constrained else np.ones_like(self.predicate_probs, bool)
        score = (self.pair_score[:, None] * self.predicate_probs
                 * detection_scores[self.subject, None] * detection_scores[self.object, None])
        pi, pp = np.nonzero(allowed)
        order = np.argsort(-score[pi, pp], kind="stable")[:top]
        return [(int(self.subject[pi[i]]), int(self.object[pi[i]]), PREDICATES[pp[i]],
                 float(score[pi[i], pp[i]])) for i in order]


class RelationshipPredictor:
    def __init__(self, checkpoint: Path, predicate_embeds: np.ndarray, classifier: str = "text",
                 max_objects: int = 32, pair_budget: int = 128, device: str = "cuda"):
        self.model = RelationshipHead(768, len(AG_OBJECT_CLASSES),
                                      torch.from_numpy(predicate_embeds), classifier=classifier)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        self.model.to(device).eval()
        self.max_objects = max_objects
        self.pair_budget = pair_budget
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, result) -> FrameRelations:
        packed, feats = result.device_packed, result.device_features
        if packed is None:
            raise ValueError("FrameResult has no device tensors; use StreamingDetector output")
        n = min(len(packed), self.max_objects)  # rows are already in descending score order
        if n < 2:
            empty = np.zeros(0, np.int64)
            return FrameRelations(empty, empty, np.zeros(0, np.float32),
                                  np.zeros((0, len(PREDICATES)), np.float32))
        packed, feats = packed[:n], feats[:n]
        boxes, labels = packed[None, :, :4], packed[None, :, 5].long()
        size = torch.tensor([result.image_size], device=self.device)
        subj, obj = self.model.roles(feats[None].float(), labels)
        geo = all_pair_geometry(boxes, size)
        route = self.model.route(subj, obj, geo)[0]
        route = route.masked_fill(torch.eye(n, dtype=torch.bool, device=self.device), float("-inf"))
        k = min(self.pair_budget, n * (n - 1))
        top = torch.topk(route.flatten(), k)
        si, oi = top.indices // n, top.indices % n
        probs = predicate_probabilities(self.model.classify(subj[0, si], obj[0, oi], geo[0, si, oi]))
        host = torch.cat([si[:, None].float(), oi[:, None].float(),
                          torch.sigmoid(top.values)[:, None], probs], dim=1).cpu().numpy()
        return FrameRelations(host[:, 0].astype(np.int64), host[:, 1].astype(np.int64),
                              host[:, 2].copy(), np.ascontiguousarray(host[:, 3:]))
