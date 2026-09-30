# Stage 1 results log

Every experiment run so far, with method and caveats. Numbers are meaningless
without the conditions that produced them, so each entry states what was
measured, on what, and what the result does **not** support.

**Hardware for all measurements below:** RTX 5070 Ti Laptop (sm_120, 12 GB),
WSL2, 15 GB host RAM. torch 2.14.0+cu130, TensorRT 11.3.0.99, modelopt 0.46.1.
**These numbers do not transfer to other hardware** — TensorRT engines are built
per-GPU. This laptop GPU measures 41.9-46.9 TFLOPS peak fp16; that is its real
capability, not throttling (re-measured plugged in, section 16). Re-measure on
every new machine.

Raw JSON for each is under `artifacts/results/`.

---

## 1. Correctness gates (passed)

Not results in themselves, but everything downstream is void without them.

| Check | Result |
| --- | --- |
| Decomposed `Owlv2DetectionGraph` vs stock HF forward | logits 7.6e-06, boxes and objectness **bit-exact** |
| TRT fp16 engine vs eager fp32 reference | cosine ≥0.9997; detection match rate 0.86–1.00; mean best IoU 0.86–0.99 |
| GPU preprocessing vs stock `Owlv2ImageProcessor` | max abs diff **1.4e-06**, 0% of pixels above 1e-3 |
| TRT YOLO path vs ultralytics PyTorch | mAP within ~3% (0.1521/0.1480, 0.1605/0.1583, 0.1966/0.1913) |
| AG frame indexing (1-based) | no zeros in 288,782 entries, global min 3, boxes render correctly on people/chairs/tables |
| AG split disjointness | 7,787 train vs 1,814 test videos; **zero** video, frame, or label overlap |

The YOLO parity check is slightly *high* rather than exact, consistent with this
project using a full 640×640 letterbox where ultralytics uses minimal-padding
rect inference. No drift indicating a broken transform.

---

## 2. Preprocessing optimization

Profiling found ~70% of preprocessing time in a single `torch.conv2d`: OWLv2
anti-aliases before downsampling (reproducing skimage `anti_aliasing=True`), and
the blur ran on the **padded** 1920×1920 fp32 tensor on CPU.

| Path | ms/frame (1080p) | Note |
| --- | --- | --- |
| Stock HF `Owlv2ImageProcessor` | 78.9–101.3 | CPU; ~70% in the anti-alias blur |
| **GPU, identical ordering** | **4.2** | **18.7× faster**, matches to 1.4e-06 |
| GPU, resize-before-pad | 3.0 | 26× but 0.2% of pixels differ at the seam — **rejected** |

Breakdown of the remaining 4.2 ms: PIL→numpy 0.91, H2D 0.65, GPU compute 2.11.

End-to-end effect: **87.5 ms → 23.8 ms** (11.4 → 42 FPS) at 960px.

`pad_last` was rejected deliberately: 1.2 ms saved is not worth changing output
for a stage that is no longer the bottleneck.

---

## 3. Resolution sweep (fp16)

`artifacts/results/resolution_sweep.json`. Accuracy on 1,200 AG test frames
across 1,143 videos, **all 36 AG classes**.

| Resolution | Patches | Latency | FPS | mAP | mAP@50 | AR@100 |
| --- | --- | --- | --- | --- | --- | --- |
| 960 (native) | 3600 | 19.00 ms | 50.7 | 0.1077 | 0.1512 | 0.3985 |
| 768 | 2304 | 10.63 ms | 87.3 | 0.0831 (−23%) | 0.1492 (**−1.3%**) | 0.3270 (−18%) |
| 640 | 1600 | 6.70 ms | 122.8 | 0.0568 (−47%) | 0.1417 (**−6.3%**) | 0.2537 (−36%) |

Latency was predicted from FLOPs within 5% before measuring, so the analytic
model is trustworthy for planning.

**The headline is the divergence, not the averages.** mAP@50 barely moves while
mAP@[.5:.95] collapses: reduced resolution still finds and classifies objects but
localises them coarsely. Averaging the two would have hidden this entirely.

Lower resolution requires `retarget_resolution()` — resampling the learned
position grid into the weights at export time. The runtime
`interpolate_pos_encoding` path is unusable for TensorRT (emits a `Range` op
required in fp32, colliding with fp16 neighbours).

---

## 4. Roofline — why engine-side optimization was abandoned

| Measurement | Value |
| --- | --- |
| Peak fp16 dense matmul, this GPU | **46.9 TFLOPS** |
| OWLv2-B/16 @960 analytic cost | 1.09 TFLOP/frame |
| Engine effective throughput | **52.9 TFLOPS** |

The engine runs *above* the dense-matmul ceiling, because TensorRT fuses
attention. Consequences, each measured rather than assumed:

- **CUDA graphs: no gain.** Python/binding overhead measured at ~0 ms — the
  engine is compute-bound, not launch-bound.
- **Builder tactics / larger batches: no gain.** No idle capacity to reclaim.
- **FlashAttention: already in use.** Engine device memory is 58 MB, while a
  materialised fp16 score matrix at 960px would be 25.9 MB per head × 12 heads ≈
  311 MB for one layer. It cannot fit, so TRT is necessarily using a fused tiled
  kernel.
- **Structural sparsity: inert.** `BuilderFlag.SPARSE_WEIGHTS` exists but does
  nothing to dense weights; benefiting requires 2:4 pruning *and* fine-tuning.

46.9 TFLOPS was first suspected to be power throttling; a later plugged-in
re-measurement gave 41.9 TFLOPS with unchanged engine timings (section 16), so it
is simply this laptop GPU's capability.

---

## 5. YOLO26 vs OWLv2 — first attempt (superseded)

Kept as a record of a methodological error worth not repeating.

Run with OWLv2 in TensorRT and YOLO in ultralytics PyTorch, and with OWLv2 timed
engine-only while YOLO's timing included Python preprocessing, NMS and result
construction. It produced the conclusion "OWLv2@960 is faster than yolo26x"
(16.99 ms vs 22.67 ms), which **normalization reversed**. Three separate biases
all favoured OWLv2.

Accuracy from this run was valid (same frames, GT and scoring) and matched the
normalized re-run within 3%.

---

## 6. YOLO26 vs OWLv2 — normalized

`artifacts/results/normalized_compare.json`. Both models as strongly-typed fp16
TensorRT engines, GPU preprocessing, GPU postprocessing, timed at two scopes.
1,144 frames, 2,400 GT boxes.

**Scored only on the 14 AG classes reachable from COCO**: bag, bed, book, chair,
cup/glass/bottle, dish, laptop, person, phone/camera, refrigerator, sandwich,
sofa/couch, table, television. Each model keeps its own full vocabulary at
inference (YOLO all 80 COCO, OWLv2 all 36 AG prompts); out-of-set predictions are
discarded rather than the vocabularies trimmed, so neither gets an easier problem
than deployment.

| Model | Res | mAP | mAP@50 | AR@100 | engine ms | e2e ms | e2e FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OWLv2-B/16 | 640 | 0.1019 | 0.2545 | 0.3134 | 6.60 | 7.63 | 131.0 |
| OWLv2-B/16 | 960 | **0.2001** | **0.2782** | **0.5123** | 18.62 | 19.12 | 52.3 |
| yolo26s | 640 | 0.1521 | 0.2091 | 0.4008 | **3.96** | **5.60** | **178.6** |
| yolo26m | 640 | 0.1605 | 0.2203 | 0.4234 | 4.41 | 6.02 | 166.2 |
| yolo26x | 640 | 0.1966 | 0.2618 | 0.4493 | 7.33 | 9.08 | 110.2 |

Findings:

- **At matched 640, YOLO wins outright.** yolo26s is faster (5.60 vs 7.63 ms)
  *and* better on strict mAP (0.152 vs 0.102). OWLv2@640 keeps only the mAP@50
  lead — it finds objects but boxes them loosely.
- **yolo26x is 2.1× faster than OWLv2@960** (9.08 vs 19.12 ms) at **statistically
  tied mAP** (0.1966 vs 0.2001, a 1.8% gap).
- **OWLv2's real margin is recall**: AR@100 0.512 vs 0.449, **+14%**. That matters
  more than mAP for this pipeline — a relationship cannot be built for an object
  never detected, so AR feeds the pair-head recall ceiling.
- **NMS costs ~1.6 ms** for YOLO (5.60 − 3.96) vs ~1.0 ms for OWLv2. YOLO26
  exports with `end2end=False`, so engine-only measurement flatters it — it wins
  anyway.

**What this does not show:** the other 22 AG classes, where YOLO26 scores zero by
construction — including `doorway`, `broom`, `vacuum`, `blanket`, `towel`, which
carry interactions. It also excludes OWLv2's text-aligned features (needed by the
semantic head) and shared visual tokens (needed by the SG-ViT pair head). The
detector decision is not reducible to this table.

---

## 7. Quantization

| Item | State |
| --- | --- |
| int8 toolchain | **Proven** — smoke test 195 s, 202 Q/DQ pairs inserted |
| `base_fp8` | **Blocked** — calibration crashed twice on host RAM (since fixed by streaming) |
| `large_int8` | **Never built** — the original Stage 1 goal |

TensorRT 11 builds strongly-typed networks only: `BuilderFlag.FP16`,
`BuilderFlag.INT8` and `IInt8EntropyCalibrator2` do not exist. Precision is a
property of the ONNX graph — fp16 by dtype conversion, int8/fp8 by Q/DQ nodes
from calibrated PTQ.

Calibration is guarded structurally: `prepare` refuses when `--calib-split`
equals `--eval-split`, writes a manifest of every frame used, and `evaluate`
cross-checks it and reports `calibration_leakage.clean`.

---

## 8. Standing caveats for every accuracy number here

1. **Absolute mAP is a floor, not a quality statement.** AG annotates only
   objects involved in an annotated interaction, so correctly finding an
   unlabelled real object scores as a false positive. Between-variant comparison
   is valid because all variants are penalised identically; the absolute value is
   not a detector-quality claim.
2. **`person` AP measures agreement with a detector**, not with human annotation
   — AG's person boxes are themselves detector output.
3. **Subset size.** Most runs use 800–1,200 of 68,183 available test frames,
   spread across videos. Indicative, not final.
4. **Untested confound: padding value.** transformers pads with `0.0`; original
   OWLv2 used `0.5` grey. Verified by measurement, never A/B'd. If HF's default
   is a regression it costs real mAP and would look like quantization damage.
5. **`base_fp16` vs `large_int8` would confound size with precision.** A
   `large_fp16` control is required before attributing any difference.

---

## 9. Box-error diagnostic and affine calibration

`artifacts/results/box_error_diagnostic.json`, `artifacts/results/box_calibration.json`.

### Diagnosis: the low-resolution error is systematic, not random

800 test frames. For each GT box, the best same-class prediction's residual was
measured in scale-invariant form (centre offset as a fraction of GT size, log
size ratio). The **mean** is bias; the **standard deviation** is jitter.

| Variant | matches | mean IoU | IoU>=.75 | log_w bias | log_w jitter | log_h bias | log_h jitter |
| --- | --- | --- | --- | --- | --- | --- | --- |
| owlv2@960 | 1753 | 0.833 | 0.798 | **-0.005** | 0.160 | **+0.004** | 0.147 |
| owlv2@768 | 1763 | 0.767 | 0.659 | **+0.083** | 0.174 | **+0.068** | 0.154 |
| owlv2@640 | 1746 | 0.697 | 0.342 | **+0.149** | 0.189 | **+0.108** | 0.164 |

At 640 the boxes are **16% too wide and 11% too tall, consistently**, while
jitter grows only 11-18% and match counts stay flat (~1750). Detection is
unaffected; only regression degrades. A uniform inflation tips otherwise-correct
boxes past strict IoU thresholds, which is why the IoU>=0.75 fraction collapses
(0.798 -> 0.342) while mean IoU only falls 0.83 -> 0.70.

Mechanically: a token covers 16x16 source pixels at 640 versus 10.7x10.7 at 960,
so both the grid prior from `compute_box_bias` and the regressed feature are 1.5x
coarser.

### Fix: per-class affine correction, fitted on train, applied to test

Size scale and centre shift per class, global fallback below 15 matches. Applied
at postprocessing; cost is numpy on <=100 boxes, unmeasurable against a 6.7 ms
engine. No training, no architecture change.

| Res | mAP before | mAP after | delta | mAP@75 before | mAP@75 after | AR@100 |
| --- | --- | --- | --- | --- | --- | --- |
| 640 | 0.0568 | **0.0816** | **+43.6%** | 0.0314 | **0.0877** (+179%) | 0.254 -> 0.308 |
| 768 | 0.0831 | **0.0955** | **+14.9%** | 0.0890 | 0.1091 (+22.6%) | 0.327 -> 0.352 |
| 960 | 0.1077 | 0.1048 | **-2.7%** | 0.1187 | 0.1180 | 0.399 -> 0.389 |

**The gain tracks the measured bias, which is the validation.** 640 had the
largest inflation and gains most; 768 had roughly half and gains proportionally;
960 had none and the correction slightly *hurts*, fitting noise where there is no
signal. A spurious result would not behave this way.

Calibrated 640 recovers from 53% to **76% of native-resolution mAP at 2.8x the
speed** (0.0816 at 6.70 ms vs 0.1077 at 19.00 ms).

**Apply at 640 and 768; do not apply at 960.** Constants are resolution-specific
and must be re-fitted per engine, always on the train split - fitting them on the
evaluation boxes would tune the correction on what it is then scored against.

Remaining gap to 960 is the jitter component, which no calibration can address; a
learned refinement head pooling features at the predicted box would be the next
step, and `tracking/features.py:pool_box_features` already provides the pooling.

---

## 10. Early objectness probe (for selective token merging)

`artifacts/results/early_objectness_960.json`, `scripts/early_objectness_probe.py`.

Question: can anything identify background tokens *before* the last layer, so
they can be merged early while object tokens stay at full resolution? OWLv2
computes objectness only at the end. Candidates: the final objectness head
applied to each intermediate block's features, and the previous video frame's
final objectness. Merging simulated as 2x2 windows (window = max token score),
lowest X% merged. An object "survives" if at least one token that detects it at
the final layer (box IoU >= 0.5, class score >= 0.05) is in an unmerged window.
Correctness gate: per-layer replication matches the real head exactly (0.0).

299 frames / 299 videos, 960px, 829 detectable objects (171 undetectable even
unmerged, excluded). Merging 25/50/75% of windows removes 19/38/56% of tokens.

| Score | survive @25% | @50% | @75% |
| --- | --- | --- | --- |
| random | 0.829 | 0.600 | 0.346 |
| blocks 0-6 | 0.91-1.00 | 0.77-0.91 | 0.48-0.60 |
| block 7 | 0.999 | 0.970 | 0.824 |
| **block 8** | **1.000** | **0.998** | **0.989** |
| blocks 9-12 | 1.000 | 1.000 | 0.992-0.999 |
| previous frame | 0.986 | 0.955 | 0.882 |
| previous frame, dilated | 0.996 | 0.970 | 0.872 |
| 5 frames back, dilated | 0.998 | 0.959 | 0.836 |

- The head becomes reliable at **block 8**; earlier blocks lose 9-23% of objects
  at 50% merge.
- The previous frame ranks tokens closer to final than block 11 does (Spearman
  0.884 vs 0.803) but still loses 3-5% of objects at 50% - motion and detection
  flicker near threshold.
- Rank correlation is a poor guide here (block 5: rho 0.26, survival 0.88);
  survival is the decision metric.
- Dilation helps at 50% but slightly hurts at 75%: a fixed budget spent on halos
  around large objects leaves small ones exposed.
- Stuff classes (floor/door/window/light/doorway) are not penalised relative to
  things.

Analytic compute estimate (not measured): merging only at block 8 saves ~16-22%
of backbone FLOPs; a cascade (25% from block 1 via the previous frame, up to 75%
at block 8) ~39% at an estimated 98-99% survival. Benchmark to beat: 768px +
calibration, ~46% saved at mAP 0.0955.

**Ceiling estimate only**: features are held fixed. Real merging changes the
surviving tokens through attention; that needs an actual merging experiment.

---

## 11. Selective token merging, actual (not simulated)

`artifacts/results/cascade_benchmark.json`, `scripts/cascade_benchmark.py`,
`src/sggpipeline/detect/token_merging.py`.

2x2 windows averaged into one token; proportional attention (log size added to
key logits, folded into an extra head dimension so fused SDPA kernels still
apply); unmerged before the heads. Gates: folded attention vs masked reference
6e-7; rewritten forward vs real graph with merging off 0.0 at all resolutions.

Schedules: `prev50` = 50% of windows before block 1 by previous-frame objectness
(dilated); `b8_75` = 75% before block 9 by block-8 objectness; `cascade` = 25%
early + up to 75% at block 8. 600 test frames, one per video, 36 classes,
uncalibrated, eager fp16.

| Res | Config | Compute vs 960 | Eager ms | mAP | mAP@75 | AP small | AR@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 960 | none | 1.00 | 126.9 | 0.1200 | 0.1302 | 0.0219 | 0.4036 |
| 960 | **prev50** | **0.52** | 71.5 | **0.1206** | **0.1319** | 0.0216 | 0.4032 |
| 960 | b8_75 | 0.78 | 100.6 | 0.1124 | 0.1240 | 0.0207 | 0.3970 |
| 960 | cascade | 0.61 | 84.2 | 0.1200 | 0.1314 | 0.0208 | 0.3934 |
| 768 | none | 0.54 | 63.5 | 0.0912 | 0.0953 | 0.0122 | 0.3312 |
| 640 | none | 0.34 | 34.0 | 0.0620 | 0.0366 | 0.0072 | 0.2555 |
| 640 | prev50 | 0.19 | 27.7 | 0.0634 | 0.0372 | 0.0072 | 0.2506 |
| 640 | b8_75 | 0.27 | 32.4 | 0.0537 | 0.0318 | 0.0065 | 0.2515 |
| 640 | cascade | 0.22 | 30.5 | 0.0598 | 0.0326 | 0.0082 | 0.2526 |

- **Early merging via the previous frame is near-lossless at 960** *(Correction: on the full test split merging costs 3.3% mAP / 1.2% AR, concentrated in large low-texture objects - see section 18. The 600-frame samples could not resolve a gap that small.)* (0.52x compute,
  mAP/AR within noise) and beats 768px at matched compute by +32% mAP, +38%
  mAP@75, +22% AR, +77% AP small. Compressing background beats shrinking the image.
- **Late merging hurts** (b8_75: -6% mAP at 960, -13% at 640); the cascade is not
  better than prev50. This contradicts the probe-based prediction: the survival
  metric treated merged windows as deleted (they still predict after unmerge)
  and could not see feature damage from averaging specialised late tokens. Use
  the probe to screen ideas, not to predict accuracy.
- At 640 prev50 halves compute again at ~no mAP cost, but eager speedup is only
  1.23x: bookkeeping dominates at 1,600 tokens.
- Eager latency overstates merging overhead (960 prev50 71.5 ms vs 768 63.5 ms at
  similar FLOPs); TensorRT latency unmeasured. prev50 keeps static shapes (fixed
  2,251 tokens, indices supplied from outside), the easiest TRT case.
- **Optimistic in one respect:** the prior came from an unmerged pass. In a stream
  it comes from the previous *merged* pass, risking lock-in (an object that
  appears in a merged region stays low-scored and merged). Needs a
  consecutive-frame test and likely a periodic unmerged refresh.

---

## 12. Merge fraction sweep and streaming lock-in test

`artifacts/results/merge_fraction_sweep.json`, `artifacts/results/merge_streaming_test.json`.

### Fraction sweep (previous-frame prior, 600 frames, one per video)

Eager latencies are within-run only: the GPU ran unthrottled here (960 unmerged
30.2 ms vs 126.9 ms in section 11).

| Res | Config | Compute | Eager ms | mAP | mAP@75 | AR@100 |
| --- | --- | --- | --- | --- | --- | --- |
| 960 | none | 1.00 | 30.2 | 0.1200 | 0.1302 | 0.4036 |
| 960 | prev50 | 0.52 | 20.5 | 0.1206 | 0.1319 | 0.4032 |
| 960 | prev60 | 0.44 | 18.1 | 0.1200 | 0.1302 | 0.3886 |
| 960 | prev70 (dilated) | 0.37 | 14.9 | 0.1121 | 0.1218 | 0.3786 |
| 960 | prev70 no dilation | 0.37 | 15.0 | 0.1215 | 0.1315 | 0.3900 |
| 960 | prev80 | 0.29 | 13.7 | 0.1009 | 0.1078 | 0.3380 |
| 640 | none | 0.34 | 11.1 | 0.0620 | 0.0366 | 0.2555 |
| 640 | prev50 | 0.19 | 9.2 | 0.0634 | 0.0372 | 0.2506 |
| 640 | prev70 no dilation | 0.14 | 7.2 | 0.0590 | 0.0349 | 0.2441 |

Knee between 50% and 70%. Dilation hurts at 70% (budget spent on halos), as the
probe predicted. Merged 960 at 0.37x compute has twice the mAP of plain 640 at
0.34x.

### Streaming (self-fed prior, consecutive frames)

5,821 frames over 40 five-second segments at 960px. Pseudo-GT = unmerged
detections >= 0.3; recall = same-class detection >= 0.1 at IoU >= 0.5. Measures
loss relative to not merging, not absolute accuracy.

| Config | Recall | New-object recall |
| --- | --- | --- |
| self-fed 50% | 0.994 | 0.973 |
| clean prior 50% | 0.994 | 0.973 |
| self-fed 70% no-dil | 0.971 | 0.926 |
| clean prior 70% no-dil | 0.969 | 0.922 |

No lock-in: self-fed equals clean-prior, and recall is flat across frames since
refresh (1-5 through 61-149). No periodic refresh needed within 5 s; longer
horizons untested. 70% misses ~1 in 14 newly appearing objects vs ~1 in 37 at 50%.

**Decision: prev50 (dilated) for the engine** - no measurable loss on 600 frames at 0.52x *(Correction: on the full test split merging costs 3.3% mAP / 1.2% AR, concentrated in large low-texture objects - see section 18. The 600-frame samples could not resolve a gap that small.)* -
encoder compute.

---

## 13. prev50 merged TensorRT engine

`artifacts/engines/base_merged50_fp16.plan`, `artifacts/results/verify_base_merged50.json`,
`artifacts/results/merged_engine_benchmark.json`; code in
`src/sggpipeline/detect/merged_export.py`, `scripts/build_merged_engine.py`,
`scripts/merged_engine_benchmark.py`.

Static graph: 2,251 tokens (450 of 900 windows merged) every frame; the merge plan
enters as three int64 index inputs (`unmerged_idx` 1800, `member_patches` 450x4,
`assign` 3600) computed outside the engine from the previous frame's objectness.
Export graph matches eager `merged_forward` bit-exactly.

**Verification** vs eager fp32, 8 real frames with real priors: logit cosine
0.99967, detection match 0.980, mean IoU 0.975 - PASS. Engine 181 MB, 165 layers,
**41.6 MB device memory**: fused attention survived (materialised scores at 2,251
tokens would be ~120 MB for one layer).

**Benchmark** (TensorRT fp16, same run, 600 frames one per video, 36 classes,
uncalibrated, prior from the 960 engine on the previous frame):

| Engine | Engine ms | p95 | Plan ms | Per-frame ms | FPS | mAP | mAP@75 | AP small | AR@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 960 unmerged | 17.89 | 20.20 | - | 18.07 | 55.3 | 0.1193 | 0.1292 | 0.0213 | 0.4051 |
| **960 merged50** | **14.26** | **15.42** | 0.38 | **15.10** | **66.2** | **0.1207** | **0.1324** | 0.0210 | **0.4038** |
| 768 unmerged | 10.01 | 11.36 | - | 10.45 | 95.7 | 0.0917 | 0.0978 | 0.0120 | 0.3324 |
| 640 unmerged | 6.55 | 7.28 | - | 7.00 | 142.8 | 0.0634 | 0.0379 | 0.0078 | 0.2551 |

- No measurable loss on 600 frames in TensorRT, as in eager. *(Correction: on the full test split merging costs 3.3% mAP / 1.2% AR, concentrated in large low-texture objects - see section 18. The 600-frame samples could not resolve a gap that small.)* Merge plan costs 0.38 ms.
- **Only 20% faster despite 48% less encoder compute**: effective throughput fell
  from ~61 to ~40 TFLOPS. Unprofiled. Leading hypothesis: the proportional-
  attention fold (head dim 64 -> 72, per-layer concats) lands on a slower fused
  kernel; secondary: input/output gathers blocking fusion.
- Per-frame figures here use Charades' 480x270 frames, so preprocessing is far
  cheaper than the 1080p figures in section 2.
- The first frame of a stream has no prior: run the unmerged engine once (or use a
  default plan). No refresh needed afterwards within the 5 s tested.

---

## 14. Kernel profile: where the merged engine's speedup went

`artifacts/results/engine_kernel_profile.json`, `scripts/profile_engines.py`.
Every CUDA kernel recorded via CUPTI (torch.profiler), 50 iterations, both 960
engines in the same run.

| Category | 960 unmerged | 960 merged50 | Ratio |
| --- | --- | --- | --- |
| GEMM (projections + MLP) | 8.97 ms (57 launches) | 6.22 ms (57) | 0.69 |
| **Fused attention (`_gemm_mha_v2`)** | **7.09 ms (12)** | **7.19 ms (12)** | **1.01** |
| Pointwise / norms / concats | 0.87 ms (43) | 1.04 ms (94) | 1.19 |
| Total | 16.96 ms | 14.47 ms | 0.85 |

- GEMMs scale as expected with tokens (0.69 vs 0.625 token ratio).
- **Attention did not shrink at all**, though it should scale ~quadratically to
  ~0.39x (~2.8 ms). Cause: the proportional-attention fold widens heads 64 -> 72;
  the fused MHA kernel is ~2.5x slower per FLOP at that width. Its concats also
  show as the extra pointwise launches.
- Attention is 42% of the unmerged engine at 960, so this is most of the gap.
  With standard 64-dim heads the merged engine is estimated at ~10 ms (the 768
  engine's speed) - an estimate, not a measurement.
- Fixes: drop proportional attention (fastest; accuracy must be re-measured) or
  implement it as exact 4x key/value duplication with 64-dim heads (~0.625x
  attention cost, accuracy unchanged by construction).

---

## 15. Merged engine without proportional attention (current best)

`artifacts/engines/base_merged50_np_fp16.plan`, `artifacts/results/verify_base_merged50_np.json`,
`artifacts/results/merged_engine_benchmark.json`. Built with
`scripts/build_merged_engine.py --no-proportional`: merged tokens attend as one
token, heads keep their native 64-dim width. Graph shrinks from 1,854 to 596 ONNX
nodes and 165 to 118 TensorRT layers; device memory 36.3 MB.

Verification vs eager fp32 (same schedule, no proportional attention), 8 real
frames: logit cosine 0.99972, detection match 0.979, mean IoU 0.971 - PASS.

TensorRT fp16, same run, 600 frames one per video, 36 classes, uncalibrated:

| Engine | Engine ms | p95 | Per-frame ms | FPS | mAP | mAP@75 | AP small | AR@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 960 unmerged | 17.74 | 20.05 | 17.49 | 57.2 | 0.1193 | 0.1292 | 0.0213 | 0.4051 |
| 960 merged50 (proportional) | 14.51 | 15.16 | 14.93 | 67.0 | 0.1207 | 0.1324 | 0.0210 | 0.4038 |
| **960 merged50, no proportional** | **10.29** | **10.70** | **10.70** | **93.4** | **0.1188** | **0.1306** | **0.0217** | **0.4036** |
| 768 unmerged | 9.87 | 11.70 | 10.30 | 97.1 | 0.0917 | 0.0978 | 0.0120 | 0.3324 |
| 640 unmerged | 6.35 | 7.84 | 6.92 | 144.4 | 0.0634 | 0.0379 | 0.0078 | 0.2551 |

- **1.72x faster than unmerged 960 at unchanged accuracy** (all deltas within noise).
  The profile-based estimate (~10 ms) held: fixing the attention width recovered
  the missing speedup.
- **Matches 768's speed with native-960 accuracy**: +30% mAP, +34% mAP@75, +81%
  AP small, +21% AR, and a better p95 (10.70 vs 11.70 ms).
- Proportional attention is unnecessary for this model: with vs without, mAP
  0.1207 vs 0.1188 and AR 0.4038 vs 0.4036 are within noise.
- Deployment: seed the first frame of a stream with one unmerged pass; the merged
  engine then supplies its own prior (no refresh needed within 5 s, section 12).

---

## 16. Speed on the laptop and on the RTX 5090

`scripts/build_engines.py` (rebuild on a new GPU), `scripts/speed_benchmark.py`;
`artifacts/results/speed_benchmark_*.json`. Charades 480x270 frames; no dataset
needed. "Sustained" = decoder thread -> GPU preprocess -> engine back to back,
the merged engine building each plan from its own previous output. Correctness
gate (merged engine vs eager fp32, real consecutive frames) passed on both GPUs:
logit cosine 0.99972.

| Engine | Laptop engine ms | Laptop sustained ms/frame | 5090 engine ms | 5090 p99 | 5090 sustained FPS | 5090 ms/frame | 5090 share of 16 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 960 unmerged | 16.73 | 18.95 | 4.74 | 6.80 | 125.9* | 7.94* | 50%* |
| **960 merged50 (np)** | **10.09** | **11.76** | **2.74** | **2.77** | **318.4** | **3.14** | **20%** |
| 768 unmerged | 9.90 | 11.02 | 2.79 | 2.82 | 351.6 | 2.84 | 18% |
| 640 unmerged | 6.44 | 7.47 | 1.67 | 1.68 | 585.5 | 1.71 | 11% |

\*A benchmark-harness artifact, now confirmed and fixed: the producer allocated
fresh page-locked memory per frame (`pin_memory()`), which the CPU cannot
recycle while running ahead of the GPU. With buffers pinned once and reused
(`stream_overhead_probe.py`, RTX 5090):

| Engine | Per-frame pin | Pinned once, reused | Frames already on GPU |
| --- | --- | --- | --- |
| 960 unmerged | 7.76 ms (129 FPS) | **4.89 ms (204 FPS)** | 4.88 ms |
| 960 merged | 12.91 ms (78 FPS) | **3.17 ms (316 FPS)** | 3.08 ms |

So 960 unmerged really sustains ~204 FPS (31% of 16 ms), and per-frame pinning
is erratic as well as slow. `speed_benchmark.py` now pins once. The laptop's
1-2 ms sustained gap very likely has the same cause (not re-measured: the
laptop GPU was throttled overnight). For Stage 2: decode into a fixed set of
reusable pinned buffers, never allocate per frame.

- fp16 roofline: laptop 41.9 TFLOPS (plugged in - not throttled), 5090 228.2
  TFLOPS (5.4x). Engine speedup laptop -> 5090 is only 3.7x: batch-1 kernels do
  not fill the larger GPU.
- Merging stays 1.73x faster than unmerged at 960 on the 5090 (1.72x on the
  laptop) and again matches 768's speed, so laptop comparisons transfer.
- On the 5090 the current best detector uses ~20% of a 16 ms throughput budget,
  leaving ~12.9 ms for downstream stages or a larger detector.
- On the laptop, sustained streaming runs 1-2 ms/frame slower than isolated
  per-frame timing (merged 11.76 vs 10.67 ms). Causes unconfirmed: per-frame
  host sync in the merge planner (`nonzero`), per-frame pinned allocation,
  Python overhead, heat under continuous load. The 5090 merged engine shows no
  such gap (3.14 sustained vs 3.32 per-frame).
- CPU side: single-thread decode 360 FPS (laptop) / 582 FPS (5090) at 480x270;
  CPU postprocessing ~0.5-1.4 ms/frame.

---

## 17. Stage 2 readiness (review of `tracking/`)

The Stage 2 code (association tracker, identity head, BoT-SORT adapter, cache
format) is well built: 23 tests pass, identity is scoped per video, category
labels are refused as identity supervision, ambiguous matches retire the old ID.
Measured blockers and fixes:

**Action Genome cannot evaluate identity.** Across all 288,782 annotated frames:
zero frames with two visible objects of the same class, zero with two person
boxes, and no instance IDs. The core identity case ("this cup vs an identical
cup") has no test cases in AG; a tracked-instance subset is needed (plan.md
anticipated this). Separately, 12% of AG object pairs sit in frames with no
person box (the person detector missed), so they have no subject and are
excluded from relationship work.

**Raw detector output needs filtering before tracking** (200 test frames, 960px
unmerged, no NMS):

| Score threshold | Detections/frame | Same-class near-duplicates (IoU > 0.7) |
| --- | --- | --- |
| >= 0.05 (Stage 1 evaluation) | 64.9 | 16.0% |
| >= 0.1 | 36.1 | 9.5% |
| >= 0.3 | 6.9 | 0.4% |

`postprocess(..., nms_iou=...)` now provides per-class suppression; the default
path is unchanged (bit-identical on 50 random frames).

**Assignment speed.** The tracker's pure-Python Hungarian solver took 3.66 ms at
32x32 and 40.3 ms at 100x100; SciPy returns the identical result in 0.035 / 0.21
ms. Swapped (tested against exhaustive search). scipy currently arrives via
nvidia-modelopt and should be declared directly.

**Container thread limit.** On the 5090 VM (pids.max 2,816), 90 extraction
workers each running FFmpeg's default one-thread-per-core decoder failed with
EAGAIN. Decoders are now capped at 2 threads (`--decoder-threads`); the rerun
extracted all 288,777 frames with zero errors.

---

## 18. Full test-split evaluation (definitive Stage 1 accuracy)

`artifacts/results/full_split_eval.json`, `scripts/full_split_eval.py`. Every AG
test keyframe with a previous video frame: **68,183 frames from all 1,814 test
videos**. 36 classes, uncalibrated, TensorRT fp16 on the RTX 5090; merged prior
from the 960 unmerged engine on the previous frame. Inference 32.6 min; COCO
scoring ~7 min per engine.

| Engine | mAP | mAP@50 | mAP@75 | AP small | AR@100 |
| --- | --- | --- | --- | --- | --- |
| 960 unmerged | 0.1079 | 0.1523 | 0.1182 | 0.0118 | 0.4088 |
| 960 merged (50%, plain attention) | 0.1043 | 0.1480 | 0.1147 | 0.0120 | 0.4039 |
| 768 unmerged | 0.0822 | 0.1491 | 0.0824 | 0.0091 | 0.3359 |
| 640 unmerged | 0.0546 | 0.1414 | 0.0256 | 0.0046 | 0.2561 |

- **Merging is not lossless**: -3.3% mAP, -2.8% mAP@50, -1.2% AR against
  unmerged 960. Earlier 600-frame runs showed differences within noise and were
  reported as lossless; the full split resolves a real, small gap.
- **The loss is concentrated in large, low-texture objects**: largest per-class
  AP drops are table (-0.024), floor (-0.021), person (-0.019), chair, bed,
  laptop; small objects are flat or slightly up (dish, clothes, sandwich,
  phone). Consistent with interior windows of big uniform surfaces scoring low
  objectness and being merged. Candidate fix (untested): also protect windows
  inside the previous frame's confident detections, not only high-objectness
  patches.
- **The decision holds**: at the same speed, merged 960 beats 768 by +27% mAP,
  +39% mAP@75, +32% AP small and +20% AR.
- Subsample estimates were close for every engine (e.g. 960 unmerged 0.1077 on
  1,200 frames vs 0.1079 here), so the earlier resolution conclusions stand.

---

## 19. How often is identity a real problem? (same-class survey)

`artifacts/results/same_class_mining.json`, `scripts/mine_same_class.py`. All
9,600 AG videos (train + test) sampled at 1 frame/s: 289,861 frames, 960px
unmerged engine, detections >= 0.3 with same-class NMS at 0.5. "Separate" = two
same-class boxes with IoU < 0.1. "Movable" excludes fixed furniture and fittings
(bed, chair, closet/cabinet, door, doorknob, doorway, floor, light, mirror,
picture, refrigerator, shelf, sofa/couch, table, television, window).
"Touching a person" = overlaps a person box, a proxy for interaction.

| Situation | Share of sampled seconds | Share of videos |
| --- | --- | --- |
| >= 2 same-class detections | 64.7% | - |
| >= 2 clearly separate same-class objects | 57.9% | 93.4% (>= 3 s: 77.3%; >= 10 s: 48.7%) |
| ...of a movable class | 32.1% | - |
| ...movable and touching a person | **4.2%** | 31.0% (**>= 3 s: 6.7%**) |

Top movable classes touching a person: shoe (3,134 sampled seconds), clothes
(1,802), cup/glass/bottle (1,721), pillow (1,430), box (1,004), bag (960),
paper/notebook (706), towel (639), phone/camera (591).

- **Spot check** of 12 saved examples: most are genuine multiple instances (two
  cups while drinking, a bag in hand plus one on the wall, paper in hand plus
  papers on the table, pillows, clothes on a rack); a few are doubtful (one
  blanket split into two boxes, tiny towel boxes). Rates are broadly real but
  somewhat inflated, and the person-overlap proxy also counts worn items
  (shoes, clothes), which are not identity confusion.
- **Conclusion: identity confusion is real but a minority case.** Same-class
  coexistence is common, but mostly fixed objects that position alone separates.
  The hard case lasts >= 3 s in only 6.7% of videos. The relationship head, needed
  on every frame and evaluable on AG, should come first; identity can start from
  the conventional tracker with detector features as the baseline.
- **Asset:** the ~640 videos with sustained hard cases (`per_video` in the JSON)
  are the pool to hand-label for an identity test set if identity becomes a
  priority, instead of labelling at random.
- Caveats: detector counts, not ground truth; 1 frame/s sampling; crude
  interaction proxy.

---

## 20. Stage 1 -> Stage 2 bridge (per-detection features, streaming detector)

`src/sggpipeline/detect/stream.py`, `scripts/verify_bridge.py`,
`artifacts/results/verify_bridge_NVIDIA_GeForce_RTX_5090.json`.

Engines optionally export the 768-dim per-patch features the heads read
(`patch_features`); each detection's descriptor is the row of the patch that
produced it. (The 512-dim class embedding was rejected: it is trained to match
category names, the wrong signal for telling identical objects apart.)
`StreamingDetector` seeds each stream with the unmerged engine, then runs the
merged engine with self-fed plans, per-class NMS, and feature gather.

**Correctness** (RTX 5090, TensorRT fp16 vs eager fp32, 8 real frames; mean
cosine):

| Engine | logits | boxes | objectness | patch_features |
| --- | --- | --- | --- | --- |
| 960 unmerged + features | 0.99974 | 0.99985 | 0.99969 | 0.99991 |
| 960 merged + features | 0.99973 | 0.99984 | 0.99961 | 0.99988 |
| merged + features vs merged without | 0.99998 | 1.0 | 1.0 | - |

Adding the output leaves detections unchanged.

**Stream sanity** over 720 frames (717 merged): 33.2 detections/frame at score
>= 0.1 with NMS 0.7; zero same-class pairs above IoU 0.7, zero degenerate boxes,
zero bad feature rows.

**Cost** (sustained, frames on GPU, decode excluded):

| | ms/frame |
| --- | --- |
| Bare merged engine | 3.08-3.11 |
| Streaming detector, numpy postprocess | 5.04 |
| Streaming detector, GPU postprocess (`postprocess_torch`) | **4.32** |

GPU postprocessing (identical output to the numpy reference, tested on CPU and
CUDA) cut the bridge's overhead from 1.96 to 1.21 ms. The rest is mostly several
small device-to-host copies per frame that could be batched into one. The full
detector with the bridge uses ~27% of the 16 ms budget.

---

## 21. Protecting the previous frame's detections during merging

`artifacts/results/merge_protection_eval.json`, `scripts/merge_protection_eval.py`.
Full test split (68,183 frames), same protocol as section 18. The merge planner
merges last any window whose centre lies inside one of the previous frame's
detections at or above a score threshold; the budget stays 450 windows.

| Merge planning | mAP | mAP@50 | mAP@75 | AP medium | AP large | AR@100 |
| --- | --- | --- | --- | --- | --- | --- |
| Unprotected | 0.1043 | 0.1480 | 0.1147 | 0.0769 | 0.1529 | 0.4039 |
| **Protect boxes >= 0.30** | **0.1064** | **0.1503** | **0.1165** | 0.0770 | **0.1561** | 0.4045 |
| Protect boxes >= 0.15 | 0.1062 | 0.1503 | 0.1164 | 0.0765 | 0.1560 | 0.4055 |
| *960 unmerged* | *0.1079* | *0.1523* | *0.1182* | *0.0766* | *0.1589* | *0.4088* |

- The unprotected run reproduces section 18 exactly (0.1043).
- **Protection recovers ~58% of merging's mAP loss**: -3.3% -> -1.4% against
  unmerged, and 53% of the large-object AP gap. Recall barely moves, so the
  remaining recall gap has another cause.
- Per class (unmerged / merged / protected): table 0.297 / 0.273 / 0.291, chair
  0.137 / 0.127 / 0.133, laptop 0.376 / 0.368 / 0.372, bed 0.180 / 0.171 /
  0.175, floor 0.098 / 0.077 / 0.084, person 0.568 / 0.549 / 0.553. Floor
  recovers least, most likely because floor detections are often below the 0.3
  threshold; the reason person recovers little is unexplained.
- Cost in a stream is near zero: the previous frame's detections are that
  frame's own output. `StreamingDetector` now protects boxes >= 0.3 by default
  (`protect_score`).

---

## 22. First relationship (pair) head, PredCls

`artifacts/results/pair_head_predcls.json`, `scripts/cache_pair_features.py`,
`scripts/train_pair_head.py`, `src/sggpipeline/relations/pair_head.py`.

Setting: ground-truth person and object boxes and object labels are given; only
the 26 AG predicates are predicted (attention: one of 3; spatial: any of 6;
contacting: any of 17). Frozen 960px detector features (768-dim), SG-ViT-style
directed head (separate subject/object projections + relative geometry + object
label), 15 epochs, model selection on 395 held-out *training* videos (21,019
pairs), test scored once. Train 355,759 pairs; test 146,656 pairs in 56,923
frames. Frames without a person box (~12% of AG pairs) are excluded.

Feature correspondence: for 97% of ground-truth objects some patch's predicted
box matches at IoU >= 0.5 (median best IoU 0.90), so the deployment descriptor
(the matching patch) is available for nearly every object.

| Variant | R@10 wc | R@20 wc | mR@10 wc | mR@20 wc | R@20 nc | mR@20 nc |
| --- | --- | --- | --- | --- | --- | --- |
| Frequency prior (no model) | 0.619 | 0.643 | 0.254 | 0.275 | 0.940 | 0.684 |
| Geometry + label only | 0.652 | 0.677 | 0.343 | 0.369 | 0.953 | 0.767 |
| **+ box-averaged features** | **0.705** | **0.733** | **0.410** | **0.443** | **0.967** | **0.834** |
| + matched-patch features | 0.703 | 0.731 | 0.408 | 0.438 | 0.966 | 0.827 |

wc = at most one predicate per group per pair; nc = all 26 compete. R = recall
of true (pair, predicate) among the top K per frame, averaged over frames; mR =
per-predicate recall averaged over predicates.

- **Frozen detector features add real relationship information**: +5.6 pts
  R@20 and **+7.4 pts mR@20 (+20% relative)** over the geometry+label control.
  Largest per-predicate gains are visual interaction predicates: twisting
  (+26 pts), writing on, wiping, lying on, carrying.
- **The deployment descriptor loses almost nothing** against box averaging
  (mR@20 0.438 vs 0.443), supporting the bridge's design.
- AG's label priors are strong (frequency alone: R@20 0.643), which is why the
  geometry-only control is the right comparison.
- nc R@50 is ~1.0 for every variant (about 2.6 pairs x 26 predicates per frame
  fits in 50 guesses) - saturated and uninformative, so not reported.
- Metrics follow standard scene-graph definitions but have **not** been checked
  line-for-line against published AG evaluation code; do not compare to
  published numbers until they are.

---

## 23. Streaming detector cost, same-run A/B (RTX 5090)

Alternating configurations in one run, 3 repeats each, frames on GPU:

| Configuration | ms/frame (median) |
| --- | --- |
| Streaming detector, no box protection | 4.30 |
| **Streaming detector, box protection (default)** | **4.53** |
| Bare merged engine (section 20) | 3.08-3.14 |

- Box protection costs 0.23 ms/frame for +2% mAP (section 21).
- Packing detections and features into one host copy made **no measurable
  difference** (4.30-4.39 vs 4.32 with separate copies); the remaining ~1.2 ms
  bridge overhead is elsewhere, likely the many small GPU ops (selection, sort,
  NMS) and their launch overhead. Unprofiled.
- The full streaming detector uses ~28% of the 16 ms throughput budget.

---

## 24. Not yet measured

- Video decode throughput — `bench/streaming.py` written, never run. If CPU
  decode caps below the engine's FPS, the resolution trade-off is moot.
- `large_int8`, `large_fp16`, `base_fp8`.
- Padding-value A/B.
- Calibration re-run against YOLO26 on the 14 shared classes (the normalized
  comparison used uncalibrated OWLv2).
- Learned box-refinement head for the residual jitter.
