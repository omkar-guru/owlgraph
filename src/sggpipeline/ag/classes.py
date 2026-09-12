"""Action Genome object vocabulary and open-vocabulary prompt construction.

AG ships its label set in ``object_classes.txt`` (index 0 is ``__background__``).
The list below is the canonical 36-category vocabulary, but it is only a
*fallback*: :mod:`sggpipeline.ag.dataset` derives the true vocabulary from the
downloaded annotations and cross-checks it against this list, warning on any
mismatch.  Never silently trust a hardcoded label set for a scored benchmark.
"""

from __future__ import annotations

# Canonical AG vocabulary, in annotation-file order (background excluded).
AG_OBJECT_CLASSES: tuple[str, ...] = (
    "person",
    "bag",
    "bed",
    "blanket",
    "book",
    "box",
    "broom",
    "chair",
    "closet/cabinet",
    "clothes",
    "cup/glass/bottle",
    "dish",
    "door",
    "doorknob",
    "doorway",
    "floor",
    "food",
    "groceries",
    "laptop",
    "light",
    "medicine",
    "mirror",
    "paper/notebook",
    "phone/camera",
    "picture",
    "pillow",
    "refrigerator",
    "sandwich",
    "shelf",
    "shoe",
    "sofa/couch",
    "table",
    "television",
    "towel",
    "vacuum",
    "window",
)

# A few AG labels are disjunctions ("cup/glass/bottle") or terse tokens that read
# poorly as detection prompts.  Scoring a single awkward string would understate
# open-vocabulary quality for reasons that have nothing to do with the detector,
# so each class expands to several prompts whose scores are max-pooled.
_PROMPT_OVERRIDES: dict[str, tuple[str, ...]] = {
    "closet/cabinet": ("a closet", "a cabinet", "a cupboard"),
    "cup/glass/bottle": ("a cup", "a glass", "a bottle", "a mug"),
    "paper/notebook": ("a piece of paper", "a notebook"),
    "phone/camera": ("a phone", "a mobile phone", "a camera"),
    "sofa/couch": ("a sofa", "a couch"),
    "clothes": ("clothes", "a piece of clothing"),
    "groceries": ("groceries", "a bag of groceries"),
    "light": ("a light", "a lamp"),
    "medicine": ("medicine", "a bottle of medicine", "a pill bottle"),
    "picture": ("a picture", "a framed picture on the wall"),
    "dish": ("a dish", "a plate"),
    "vacuum": ("a vacuum cleaner",),
    "floor": ("the floor",),
    "doorway": ("a doorway",),
    "doorknob": ("a doorknob",),
    "television": ("a television", "a tv screen"),
}

PROMPT_TEMPLATE = "a photo of {}"


def prompts_for_class(name: str) -> tuple[str, ...]:
    """Return the surface strings to embed for one AG class."""
    if name in _PROMPT_OVERRIDES:
        variants = _PROMPT_OVERRIDES[name]
    else:
        article = "an" if name[0] in "aeiou" else "a"
        variants = (f"{article} {name}",)
    return tuple(PROMPT_TEMPLATE.format(v) for v in variants)


def build_prompt_index(
    classes: tuple[str, ...] = AG_OBJECT_CLASSES,
) -> tuple[list[str], list[int]]:
    """Flatten every class's prompts into one list for a single text-tower pass.

    Returns the prompt strings and a parallel list mapping each prompt back to
    its class index, so per-prompt logits can be max-pooled into per-class ones.
    """
    prompts: list[str] = []
    owner: list[int] = []
    for class_idx, name in enumerate(classes):
        for prompt in prompts_for_class(name):
            prompts.append(prompt)
            owner.append(class_idx)
    return prompts, owner
