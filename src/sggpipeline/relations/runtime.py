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
from .model import RelationshipHead, all_pair_geometry
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
    """The relationship head on ``StreamingDetector`` output, one frame at a time.

    The head runs at a fixed shape - detections padded to ``max_objects``, a
    fixed ``pair_budget`` - so it can be captured once as a CUDA graph and
    replayed per frame. Eagerly it is ~100 small kernels whose launch cost,
    not arithmetic, dominates (the same ~1.75 ms at 64 or 992 pairs on an RTX
    5090); replayed, it is one launch. Padded detections are masked out of
    routing, and pairs the router could only fill with padding are dropped on
    the host.
    """

    def __init__(self, checkpoint: Path, predicate_embeds: np.ndarray, classifier: str = "text",
                 max_objects: int = 32, pair_budget: int = 128, device: str = "cuda",
                 cuda_graph: bool = True):
        self.model = RelationshipHead(768, len(AG_OBJECT_CLASSES),
                                      torch.from_numpy(predicate_embeds), classifier=classifier)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        self.model.to(device).eval()
        self.max_objects = max_objects
        self.pair_budget = pair_budget
        self.device = torch.device(device)
        self.cuda_graph = cuda_graph and self.device.type == "cuda"
        n = max_objects
        self._eye = torch.eye(n, dtype=torch.bool, device=self.device)
        self._arange = torch.arange(n, device=self.device)
        self._packed = torch.zeros(n, 8, device=self.device)
        self._feats = None  # allocated on first use, in the detector's feature dtype
        self._valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._size = torch.zeros(1, 2, dtype=torch.long, device=self.device)
        self._host_size = None
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}

    def _head(self, packed: torch.Tensor, feats: torch.Tensor, valid: torch.Tensor,
              size: torch.Tensor, budget: int) -> torch.Tensor:
        """Fixed shapes in, (budget, 3 + P) rows out: subject, object, router logit, probs."""
        n = packed.shape[0]
        boxes, labels = packed[None, :, :4], packed[None, :, 5].long()
        subj, obj = self.model.roles(feats[None].float(), labels)
        geo = all_pair_geometry(boxes, size)
        route = self.model.route(subj, obj, geo)[0]
        blocked = ~(valid[:, None] & valid[None, :]) | self._eye
        top = torch.topk(route.masked_fill(blocked, float("-inf")).flatten(), budget)
        si, oi = top.indices // n, top.indices % n
        probs = self.model.probabilities(self.model.classify(subj[0, si], obj[0, oi], geo[0, si, oi]))
        return torch.cat([si[:, None].float(), oi[:, None].float(), top.values[:, None], probs], dim=1)

    def _replay(self, budget: int) -> torch.Tensor:
        args = (self._packed, self._feats, self._valid, self._size, budget)
        if budget not in self._graphs:
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):  # warm up off the capture stream, as capture requires
                for _ in range(3):
                    self._head(*args)
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._head(*args)
            self._graphs[budget] = (graph, out)
        graph, out = self._graphs[budget]
        graph.replay()
        return out

    @torch.no_grad()
    def __call__(self, result) -> FrameRelations:
        packed, feats = result.device_packed, result.device_features
        if packed is None:
            raise ValueError("FrameResult has no device tensors; use StreamingDetector output")
        n = min(len(packed), self.max_objects)  # rows are already in descending score order
        budget = min(self.pair_budget, self.max_objects * (self.max_objects - 1))
        if n < 2:
            empty = np.zeros(0, np.int64)
            return FrameRelations(empty, empty, np.zeros(0, np.float32),
                                  np.zeros((0, len(PREDICATES)), np.float32))
        if self._feats is None:
            self._feats = torch.zeros(self.max_objects, feats.shape[1], dtype=feats.dtype, device=self.device)
        self._packed[:n].copy_(packed[:n])
        self._feats[:n].copy_(feats[:n])
        torch.lt(self._arange, n, out=self._valid)
        if result.image_size != self._host_size:
            self._size.copy_(torch.tensor([result.image_size]))
            self._host_size = result.image_size
        out = self._replay(budget) if self.cuda_graph else self._head(
            self._packed, self._feats, self._valid, self._size, budget)
        host = out.cpu().numpy()
        host = host[np.isfinite(host[:, 2])]  # budget slots only padding could fill
        return FrameRelations(host[:, 0].astype(np.int64), host[:, 1].astype(np.int64),
                              (1 / (1 + np.exp(-host[:, 2]))).astype(np.float32),
                              np.ascontiguousarray(host[:, 3:]))
