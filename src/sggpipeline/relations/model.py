"""The relationship head: SG-ViT-style routing and classification over detections.

Per frame, given up to N detections (boxes, labels, per-detection features from
the frozen detector):

1. **Role features.** Each detection gets a *subject* and an *object* projection
   of its feature plus a label embedding. Roles are separate because the pair is
   directed: person-holds-cup is not cup-holds-person. The label enters through
   the *text embedding* of its class name (projected), so an object class never
   seen in training - or any name queried at deployment - still has one. The
   older per-class learned vector (``class_embeds=None``) is kept for loading
   earlier checkpoints and as the closed control.
2. **Router.** Every ordered pair (i, j), i != j, gets a cheap score: a dot
   product between i's subject key and j's object key, plus a small term on
   their relative geometry. Only the top-K pairs go further; the rest are never
   classified, which is where the saving is - and why pair-selection recall has
   to be measured.
3. **Pair embedding.** Selected pairs combine both role features and a richer
   geometry encoding into one embedding.
4. **Predicate classifier.** ``text``: cosine similarity between the projected
   pair embedding and predicate text embeddings, so a predicate never seen as a
   label can still be scored. ``closed``: a fixed linear layer, the control that
   cannot score unseen predicates at all.

Predicates are described by a :class:`PredicateSchema`. Action Genome has three
groups: attention (exactly one of 3, softmax), spatial (any of 6) and contacting
(any of 17), both sigmoid. Visual Genome (VG150) has one group of 50, sigmoid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..ag.relations import ATTENTION, CONTACTING, PREDICATES, SPATIAL
from .pair_head import GEOMETRY_DIM, pair_geometry

NUM_PREDICATES = len(PREDICATES)
GROUP_SLICES = {
    "attention": slice(0, len(ATTENTION)),
    "spatial": slice(len(ATTENTION), len(ATTENTION) + len(SPATIAL)),
    "contacting": slice(len(ATTENTION) + len(SPATIAL), NUM_PREDICATES),
}


@dataclass(frozen=True)
class PredicateSchema:
    """A predicate vocabulary split into contiguous groups.

    ``single`` groups hold exactly one true predicate per pair (softmax,
    cross-entropy); the others are multi-label (sigmoid, BCE). The graph
    constraint keeps one predicate per group.
    """

    names: tuple[str, ...]
    groups: tuple[tuple[str, slice], ...]
    single: frozenset[str] = frozenset()

    def __len__(self) -> int:
        return len(self.names)

    @property
    def multi_label(self) -> list[slice]:
        return [sl for name, sl in self.groups if name not in self.single]


AG_SCHEMA = PredicateSchema(tuple(PREDICATES), tuple(GROUP_SLICES.items()), frozenset({"attention"}))


def all_pair_geometry(boxes: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
    """(B, N, N, G) geometry of every ordered pair (subject i, object j)."""
    b, n, _ = boxes.shape
    subj = boxes[:, :, None, :].expand(b, n, n, 4).reshape(-1, 4)
    obj = boxes[:, None, :, :].expand(b, n, n, 4).reshape(-1, 4)
    size = image_size[:, None, :].expand(b, n * n, 2).reshape(-1, 2)
    return pair_geometry(subj, obj, size).reshape(b, n, n, GEOMETRY_DIM)


def predicate_probabilities(logits: torch.Tensor, schema: PredicateSchema = AG_SCHEMA) -> torch.Tensor:
    """Softmax within single-label groups, sigmoid elsewhere (AG: attention softmax)."""
    return torch.cat([logits[..., sl].softmax(dim=-1) if name in schema.single else logits[..., sl].sigmoid()
                      for name, sl in schema.groups], dim=-1)


class RelationshipHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, predicate_embeds: torch.Tensor | None,
                 classifier: str = "text", hidden: int = 512, key_dim: int = 128,
                 dropout: float = 0.1, schema: PredicateSchema = AG_SCHEMA,
                 class_embeds: torch.Tensor | None = None):
        super().__init__()
        self.schema = schema
        if classifier not in ("text", "closed"):
            raise ValueError("classifier must be 'text' or 'closed'")
        if classifier == "text" and predicate_embeds is None:
            raise ValueError("the text classifier needs predicate embeddings")
        self.classifier = classifier
        self.key_dim = key_dim
        self.role_subject = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden))
        self.role_object = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden))
        if class_embeds is None:
            self.label = nn.Embedding(num_classes, hidden)  # closed: one learned vector per class
        else:
            self.register_buffer("class_embeds", F.normalize(class_embeds.float(), dim=1))
            self.label_text = nn.Linear(class_embeds.shape[1], hidden)
        self.subject_key = nn.Linear(hidden, key_dim)
        self.object_key = nn.Linear(hidden, key_dim)
        self.router_geometry = nn.Sequential(nn.Linear(GEOMETRY_DIM, 64), nn.GELU(), nn.Linear(64, 1))
        self.geometry = nn.Sequential(nn.Linear(GEOMETRY_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.trunk = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        if classifier == "text":
            self.register_buffer("predicate_embeds", F.normalize(predicate_embeds.float(), dim=1))
            self.to_text = nn.Linear(hidden, predicate_embeds.shape[1])
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            # One bias per group, never per predicate: a per-predicate bias would
            # learn each seen predicate's frequency and leave unseen ones at an
            # arbitrary value, biasing any held-out comparison. Softmax groups
            # need none (a shared shift cancels).
            self.group_bias = nn.Parameter(torch.zeros(len(schema.multi_label)))
        else:
            self.closed = nn.Linear(hidden, len(schema))

    # -- per detection ---------------------------------------------------------
    @property
    def text_labels(self) -> bool:
        return hasattr(self, "label_text")

    def set_vocabulary(self, class_embeds: torch.Tensor) -> None:
        """Swap in another object vocabulary (text-label heads only); labels then
        index into it."""
        if not self.text_labels:
            raise ValueError("a head with learned per-class label vectors has a fixed vocabulary")
        self.class_embeds = F.normalize(class_embeds.float(), dim=1).to(self.class_embeds.device)

    def roles(self, features: torch.Tensor, labels: torch.Tensor):
        lab = self.label_text(self.class_embeds[labels]) if self.text_labels else self.label(labels)
        return self.role_subject(features) + lab, self.role_object(features) + lab

    # -- router ------------------------------------------------------------------
    def route(self, subj: torch.Tensor, obj: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        """(B, N, N) pair logits from (B, N, H) roles and (B, N, N, G) geometry."""
        keys = self.subject_key(subj) @ self.object_key(obj).transpose(1, 2)
        return keys / math.sqrt(self.key_dim) + self.router_geometry(geometry).squeeze(-1)

    # -- classifier --------------------------------------------------------------
    def classify(self, subj: torch.Tensor, obj: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        """(P, num_predicates) logits for P selected pairs."""
        h = self.trunk(subj + obj + self.geometry(geometry))
        if self.classifier == "closed":
            return self.closed(h)
        z = F.normalize(self.to_text(h), dim=-1)
        logits = self.logit_scale.exp() * z @ self.predicate_embeds.T
        bias = torch.zeros(len(self.schema), device=logits.device, dtype=logits.dtype)
        for value, sl in zip(self.group_bias, self.schema.multi_label):
            bias[sl] = value
        return logits + bias

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return predicate_probabilities(logits, self.schema)


def classification_loss(logits: torch.Tensor, targets: torch.Tensor,
                        seen: torch.Tensor | None = None,
                        schema: PredicateSchema = AG_SCHEMA) -> torch.Tensor:
    """Cross-entropy per single-label group plus BCE over the multi-label ones,
    over seen predicates only.

    Held-out predicates contribute no gradient anywhere: their columns are
    dropped from the BCE, and a pair whose true single-label predicate is held
    out is dropped from that group's term, which is taken over seen columns only.
    """
    if seen is None:
        seen = torch.ones(len(schema), dtype=torch.bool, device=logits.device)
    loss = logits.new_zeros(())
    for name, sl in schema.groups:
        if name in schema.single:
            group_logits = logits[:, sl].masked_fill(~seen[sl], float("-inf"))
            group_target = targets[:, sl].float().argmax(dim=1)
            keep = seen[sl][group_target] & targets[:, sl].any(dim=1)
            if keep.any():
                loss = loss + F.cross_entropy(group_logits[keep], group_target[keep])
    multi = torch.zeros(len(schema), dtype=torch.bool, device=logits.device)
    for sl in schema.multi_label:
        multi[sl] = True
    cols = multi & seen
    if cols.any():
        loss = loss + F.binary_cross_entropy_with_logits(logits[:, cols], targets[:, cols].float())
    return loss


def router_loss(pair_logits: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor,
                pos_weight: float = 50.0) -> torch.Tensor:
    """BCE over valid ordered pairs; positives are rare (~2 in ~1,000), so upweighted."""
    weight = torch.where(positive, torch.full_like(pair_logits, pos_weight),
                         torch.ones_like(pair_logits))
    return F.binary_cross_entropy_with_logits(pair_logits[valid], positive[valid].float(),
                                              weight=weight[valid])


def load_relationship_head(path, num_classes: int, predicate_embeds: torch.Tensor, classifier: str = "text",
                           schema: PredicateSchema = AG_SCHEMA, device: str = "cuda") -> RelationshipHead:
    """Rebuild a saved head, text-label or learned-label, from its state dict."""
    state = torch.load(path, map_location=device, weights_only=True)
    model = RelationshipHead(768, num_classes, predicate_embeds, classifier=classifier, schema=schema,
                             class_embeds=state.get("class_embeds"))
    model.load_state_dict(state)
    return model.to(device).eval()
