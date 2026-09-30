"""Predicate phrases and their text embeddings, for open-vocabulary classification.

The relationship head scores a pair against *text embeddings* of predicate
phrases rather than a fixed classifier, following SG-ViT: a predicate it never
saw a label for can still be scored if its phrase can be embedded. Phrases are
embedded with OWLv2's own text tower, so predicates live in the same space the
detector already uses for object names.

That tower was pretrained on web image-text pairs, so these phrases are not
unseen *language* - only unseen as supervised predicates here. Held-out results
measure transfer from that pretraining, and are reported as such.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..ag.relations import PREDICATES

# Natural-language phrase for each AG predicate, in PREDICATES order. Several
# templates are embedded and averaged, so no single wording decides the result.
PHRASES: dict[str, str] = {
    "looking_at": "looking at",
    "not_looking_at": "not looking at",
    "unsure": "possibly looking at",
    "above": "above",
    "beneath": "beneath",
    "in_front_of": "in front of",
    "behind": "behind",
    "on_the_side_of": "beside",
    "in": "inside",
    "carrying": "carrying",
    "covered_by": "covered by",
    "drinking_from": "drinking from",
    "eating": "eating",
    "have_it_on_the_back": "carrying on their back",
    "holding": "holding",
    "leaning_on": "leaning on",
    "lying_on": "lying on",
    "not_contacting": "not touching",
    "other_relationship": "interacting with",
    "sitting_on": "sitting on",
    "standing_on": "standing on",
    "touching": "touching",
    "twisting": "twisting",
    "wearing": "wearing",
    "wiping": "wiping",
    "writing_on": "writing on",
}
TEMPLATES = ("a person {} something", "a photo of a person {} an object", "someone {} a thing")
# Visual Genome subjects are anything (a man, a window, a tree), so its
# predicates - already plain phrases such as "parked on" - get neutral templates.
GENERIC_TEMPLATES = ("something {} something", "a photo of an object {} another object",
                     "a thing {} a thing")


def predicate_prompts(phrases: list[str] | None = None,
                      templates: tuple[str, ...] = TEMPLATES) -> tuple[list[str], np.ndarray]:
    """All prompt strings, and the predicate index each belongs to (AG by default)."""
    if phrases is None:
        phrases = [PHRASES[name] for name in PREDICATES]
    prompts, owner = [], []
    for index, phrase in enumerate(phrases):
        for template in templates:
            prompts.append(template.format(phrase))
            owner.append(index)
    return prompts, np.asarray(owner)


@torch.no_grad()
def embed_predicates(model, processor, device: str = "cuda", phrases: list[str] | None = None,
                     templates: tuple[str, ...] = TEMPLATES) -> np.ndarray:
    """(P, D) unit-norm predicate embeddings: template embeddings averaged."""
    from ..detect.owlv2 import encode_text_queries

    prompts, owner = predicate_prompts(phrases, templates)
    embeds = encode_text_queries(model, processor, prompts, device).cpu().numpy()
    out = np.stack([embeds[owner == i].mean(axis=0) for i in range(owner.max() + 1)])
    return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)


def load_or_build(cache: Path, device: str = "cuda", phrases: list[str] | None = None,
                  templates: tuple[str, ...] = TEMPLATES) -> np.ndarray:
    """Predicate embeddings from ``cache``, computing them once if missing."""
    cache = Path(cache)
    if cache.exists():
        return np.load(cache)
    from ..detect.owlv2 import load_owlv2

    model, processor = load_owlv2("base", device=device, dtype=torch.float32)
    embeds = embed_predicates(model, processor, device, phrases, templates)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeds)
    return embeds
