"""How well does each text encoder separate Action Genome's predicate phrases?

Builds (and caches) the template-averaged predicate embeddings for every
``PREDICATE_ENCODERS`` entry and reports their spread. Raw mean cosine is not
comparable across encoders - some embed everything in a narrow cone - so the
spread is also given after centring (subtracting the mean of the 26), which is
what separating predicates from each other depends on. Nearest neighbours of
the held-out predicates show whether the geometry is sensible. Held-out AP
after training (``heldout_ranking.py``) is the measure that counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sggpipeline.ag.relations import PREDICATES
from sggpipeline.pipeline import Workspace, write_report
from sggpipeline.relations.predicates import PREDICATE_ENCODERS

from train_relationships import HOLDOUTS, predicate_embeddings


def spread(e: np.ndarray) -> tuple[float, float]:
    off = ~np.eye(len(e), dtype=bool)
    c = e - e.mean(axis=0)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    return float((e @ e.T)[off].mean()), float((c @ c.T)[off].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()
    ws = Workspace(Path(args.artifacts))
    report = {}
    for encoder in PREDICATE_ENCODERS:
        e = predicate_embeddings(ws, encoder)
        raw, centred = spread(e)
        sims = e @ e.T
        neighbours = {}
        for p in HOLDOUTS["four"]:
            i = PREDICATES.index(p)
            order = [j for j in np.argsort(-sims[i]) if j != i][:3]
            neighbours[p] = [(PREDICATES[j], round(float(sims[i, j]), 3)) for j in order]
        report[encoder] = {"dim": int(e.shape[1]), "mean_cosine": raw, "mean_cosine_centred": centred,
                           "held_out_neighbours": neighbours}
        print(f"{encoder:<7} dim {e.shape[1]:>4}  mean cosine {raw:.3f}  centred {centred:+.3f}", flush=True)
        for p, nn in neighbours.items():
            print(f"         {p:<14} -> " + ", ".join(f"{n} {s:.2f}" for n, s in nn), flush=True)
    write_report(report, ws.result("predicate_encoders.json"))
    print("PREDICATE_ENCODERS_DONE", flush=True)


if __name__ == "__main__":
    main()
