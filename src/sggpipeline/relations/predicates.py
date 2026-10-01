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


def class_embeddings(prompt_embeds: np.ndarray, owner: np.ndarray, num_classes: int) -> np.ndarray:
    """(C, D) unit-norm class embeddings: each class's prompt embeddings averaged,
    from the same text queries the detector uses."""
    out = np.stack([prompt_embeds[owner == c].mean(axis=0) for c in range(num_classes)])
    return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)


def load_or_build_classes(cache: Path, classes: tuple[str, ...], device: str = "cuda") -> np.ndarray:
    """Class-name embeddings for a vocabulary, built with the detector's prompts."""
    cache = Path(cache)
    if cache.exists():
        return np.load(cache)
    from ..ag.classes import build_prompt_index
    from ..detect.owlv2 import encode_text_queries, load_owlv2

    model, processor = load_owlv2("base", device=device, dtype=torch.float32)
    prompts, owner = build_prompt_index(classes)
    embeds = encode_text_queries(model, processor, prompts, device).cpu().numpy()
    out = class_embeddings(embeds, np.asarray(owner), len(classes))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


# Text encoders tried for predicate phrases. OWLv2's own tower was trained on
# object noun phrases and barely separates predicates (mean cosine 0.90 across
# AG's 26); sentence encoders are trained to tell phrases apart.
PREDICATE_ENCODERS = {
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge": "BAAI/bge-large-en-v1.5",
    "clip-l": "openai/clip-vit-large-patch14",
}


@torch.no_grad()
def encode_phrases(encoder: str, prompts: list[str], device: str = "cuda") -> np.ndarray:
    """(N, D) unit-norm embeddings of ``prompts`` from one of PREDICATE_ENCODERS."""
    model_id = PREDICATE_ENCODERS[encoder]
    if encoder == "owlv2":
        from ..detect.owlv2 import encode_text_queries, load_owlv2

        model, processor = load_owlv2(model_id, device=device, dtype=torch.float32)
        return encode_text_queries(model, processor, prompts, device).cpu().numpy()
    if encoder == "clip-l":
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        tok = CLIPTokenizer.from_pretrained(model_id)
        model = CLIPTextModelWithProjection.from_pretrained(model_id).to(device).eval()
        out = model(**tok(prompts, padding=True, return_tensors="pt").to(device)).text_embeds
    else:
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        batch = tok(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        hidden = model(**batch).last_hidden_state
        if encoder == "bge":  # bge pools with its CLS token
            out = hidden[:, 0]
        else:  # sentence-transformers mpnet: attention-masked mean
            mask = batch["attention_mask"][..., None].float()
            out = (hidden * mask).sum(1) / mask.sum(1)
    return torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy()


def load_or_build_encoder(cache: Path, encoder: str, phrases: list[str] | None = None,
                          templates: tuple[str, ...] = TEMPLATES, device: str = "cuda") -> np.ndarray:
    """Template-averaged predicate embeddings from any PREDICATE_ENCODERS entry."""
    cache = Path(cache)
    if cache.exists():
        return np.load(cache)
    prompts, owner = predicate_prompts(phrases, templates)
    embeds = encode_phrases(encoder, prompts, device)
    out = np.stack([embeds[owner == i].mean(axis=0) for i in range(owner.max() + 1)])
    out = (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out
