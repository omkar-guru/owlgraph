"""How far can early (previous-frame-guided) merging go before accuracy drops?

50% of windows merged before block 1 was indistinguishable from no merging at
960px, so the knee lies somewhere above it. Same harness, frames and gates as
``cascade_benchmark.py``; only the schedules differ. 70% is also run without
dilation, because the probe suggested halos around large objects crowd out
small ones once the budget gets tight.
"""

from __future__ import annotations

from cascade_benchmark import run

CONFIGS = {
    "none": dict(),
    "prev50": dict(early_fraction=0.50),
    "prev60": dict(early_fraction=0.60),
    "prev70": dict(early_fraction=0.70),
    "prev70_nodil": dict(early_fraction=0.70, early_dilate=False),
    "prev80": dict(early_fraction=0.80),
}

if __name__ == "__main__":
    run(CONFIGS, {960: list(CONFIGS), 640: list(CONFIGS)}, num_frames=600,
        out_name="merge_fraction_sweep.json")
