"""Assemble per-variant results into one comparison table.

Kept separate from measurement so the comparison can be regenerated from saved
JSON without re-running any engine.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_results(results_dir: Path, variants: list[str]) -> dict:
    """Read whatever bench/eval records exist for the named variants."""
    results_dir = Path(results_dir)
    out: dict[str, dict] = {}
    for name in variants:
        record: dict = {}
        for kind in ("bench", "eval"):
            path = results_dir / f"{kind}_{name}.json"
            if path.exists():
                record[kind] = json.loads(path.read_text())
        if record:
            out[name] = record
    return out


def _fmt(value, spec: str = ".2f", missing: str = "-") -> str:
    return format(value, spec) if isinstance(value, (int, float)) else missing


def render_table(results: dict) -> str:
    """Markdown table of the numbers that decide the Stage 1 question."""
    header = (
        "| Variant | Engine median ms | Engine p95 ms | FPS | GPU mem MB | "
        "Preproc median ms | End-to-end ms | mAP | mAP@50 | AR@100 |"
    )
    sep = "| --- " * 10 + "|"
    rows = [header, sep]

    for name, record in results.items():
        bench = record.get("bench", {})
        engine = bench.get("engine", {})
        pre = bench.get("preprocess", {})
        metrics = record.get("eval", {}).get("metrics", {})
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                _fmt(engine.get("median_ms")),
                _fmt(engine.get("p95_ms")),
                _fmt(engine.get("fps_from_median"), ".1f"),
                _fmt(engine.get("peak_gpu_mem_mb"), ".0f"),
                _fmt(pre.get("median_ms")),
                _fmt(bench.get("end_to_end_median_ms")),
                _fmt(metrics.get("mAP"), ".4f"),
                _fmt(metrics.get("mAP_50"), ".4f"),
                _fmt(metrics.get("AR_100"), ".4f"),
            )
        )
    return "\n".join(rows)


def render_caveats(results: dict) -> str:
    """State what the table does and does not support, next to the table."""
    notes = [
        "- AG annotates only interaction-relevant objects, so absolute mAP is "
        "understated for every variant. Only the between-variant comparison is "
        "meaningful.",
        "- AG's `person` boxes are detector output, so `person` AP measures "
        "agreement with that detector, not with human annotation.",
        "- Engine time excludes CPU preprocessing and the host-to-device copy, "
        "both reported separately; `End-to-end ms` is their sum at the median.",
    ]
    names = list(results)
    if "base_fp16" in names and "large_int8" in names:
        notes.append(
            "- `base_fp16` vs `large_int8` varies model size *and* precision at "
            "once. Neither can be credited for a difference without a "
            "`large_fp16` control."
        )
    return "\n".join(notes)


def render_report(results: dict) -> str:
    return "\n\n".join(
        ["## Stage 1 detector comparison", render_table(results), "### Reading these numbers", render_caveats(results)]
    )
