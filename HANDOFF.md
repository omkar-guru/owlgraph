# Stage 1 handoff — migrating to a cloud machine

Written for: whoever picks this up on the new box (likely you, later).

Status as of the last local session. Stage 1 of [plan.md](plan.md) — detector
quality and speed on Action Genome — is **partly complete**. The fp16 side is
finished and measured. The quantized side is not: it is blocked on host RAM,
which is why the local machine kept dying.

---

## 0. Before you wipe the local machine

**Nothing is committed. The repo has zero commits.** All 30 source files exist
only in the working tree. Commit and push before shutting the machine down, or
this is all gone.

```bash
cd ~/projects/sggpipeline
git add -A && git commit -m "Stage 1: OWLv2 detector benchmarking on Action Genome"
git remote add origin <your-remote> && git push -u origin master
```

`.gitignore` already excludes `.venv/` (11 GB) and `artifacts/` (4.5 GB).
**Add `acgdataset/` too — it is 27 GB and must not enter git:**

```bash
echo "acgdataset/" >> .gitignore
```

Do not copy `artifacts/engines/*.plan` to the new machine. TensorRT engines are
specific to one GPU, driver and TRT version; they must be rebuilt.

---

## 1. Why the local machine failed

Not the GPU — **host RAM**. The box had 15 GB. Calibration materialised the
whole set as one fp32 array:

```
384 frames x 3 x 960 x 960 x 4 bytes = 3.96 GB
```

on top of modelopt's ONNX graph copies and an ONNX Runtime CUDA session. That
is what killed WSL, twice.

`CalibrationReader` now streams from disk (one frame resident at a time), so the
immediate cause is fixed. But `large_int8` has not been attempted yet and its
ONNX graph is **1.2 GB**, of which modelopt keeps several copies. Size the new
machine for that.

### New machine requirements

| Resource | Minimum | Why |
| --- | --- | --- |
| Host RAM | **64 GB** | The binding constraint. modelopt holds multiple copies of a 1.2 GB graph plus an ORT session. |
| GPU memory | 24 GB+ | `large` at 1008px; 12 GB worked for `base` only. |
| GPU arch | **sm_89+ (Ada/Hopper/Blackwell)** | **fp8 needs sm_89 or newer. A100 (sm_80) has no fp8 tensor cores** — pick L40S/H100/RTX 6000 Ada, not A100, if fp8 matters. |
| Disk | 100 GB+ | 27 GB dataset + ~5 GB artifacts + 11 GB venv. |

---

## 2. Environment — reproduce exactly

**Python 3.13. Not 3.14.** `nvidia-modelopt[onnx]` pins `onnxruntime-gpu <1.25`,
which has no cp314 wheels, so int8/fp8 are unreachable on 3.14. This was
verified, not assumed.

```bash
uv sync          # pyproject.toml pins requires-python = ">=3.13,<3.14"
```

Verified local stack: torch 2.14.0+cu130, TensorRT 11.3.0.99, modelopt 0.46.1,
onnxruntime-gpu 1.24.4, transformers with `Owlv2ForObjectDetection`.

**After `uv sync`, force onnxruntime-gpu to win:**

```bash
uv pip uninstall onnxruntime onnxruntime-gpu
uv pip install --reinstall onnxruntime-gpu==1.24.4
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# must list CUDAExecutionProvider; if it only shows CPU, calibration runs on CPU and is unusably slow
```

Both packages install into the same `onnxruntime` directory and collide; the CPU
build silently wins and removes the CUDA provider.

---

## 3. Data

`acgdataset/` layout (already correct, the loader accepts it):

```
acgdataset/
  action_genome_v1.0/     # person_bbox.pkl, object_bbox_and_relationship.pkl,
                          # frame_list.txt, object_classes.txt, relationship_classes.txt
  Charades_v1_480/        # 9,848 .mp4
  frames/<video>.mp4/<nnnnnn>.png    # extracted keyframes
```

Re-extract on the new machine (frames are ~10 GB, faster to regenerate than to
copy):

```bash
uv run sgg extract-frames --ag-root acgdataset --videos-dir acgdataset/Charades_v1_480 \
                          --split test --workers $(nproc)
uv run sgg extract-frames --ag-root acgdataset --videos-dir acgdataset/Charades_v1_480 \
                          --split train --max-videos 600 --workers $(nproc)
uv run sgg validate-ag --ag-root acgdataset --split test
```

Local run produced 70,329 test frames (1,814 videos) and 17,329 train frames
(600 videos), zero errors.

**Verified facts about AG — do not re-litigate these:**

- Frame numbers are **1-based** (ffmpeg `image2`). No zeros in 288,782 entries,
  global min 3, no index exceeds its video's decoded frame count, and rendered
  boxes land correctly on people/chairs/tables.
- Object boxes are `(x, y, w, h)`; person boxes are `xyxy`.
- Boxes are already in **native frame coordinates** — `bbox_size` matches each
  video's own resolution per-video. No rescaling needed.
- Vocabulary is 35 objects + person = 36. The **pickles use slash forms**
  (`cup/glass/bottle`); `object_classes.txt` strips them (`cupglassbottle`).
  `AG_OBJECT_CLASSES` matches the pickles.
- Splits are **disjoint**: 7,787 train vs 1,814 test videos, zero video overlap,
  zero frame overlap, zero frames with mixed set labels.

---

## 4. What is done and measured

All numbers from an RTX 5070 Ti Laptop (sm_120, 12 GB). **They will change on new
hardware — re-measure, do not carry these forward as baselines.**

### Speed (fp16, batch 1)

| Resolution | Engine latency | FPS | Engine mem |
| --- | --- | --- | --- |
| 960 (native) | 19.00 ms | 50.7 | 58 MB |
| 768 | 10.63 ms | 87.3 | 40 MB |
| 640 | 6.70 ms | 122.8 | 26 MB |

Predicted from FLOPs within 5%, so the analytic model in
`scripts/speed_experiments.py` is trustworthy for planning.

### Accuracy (1,200 test frames across 1,143 videos — indicative, not final)

| Resolution | mAP | mAP@50 | AR@100 |
| --- | --- | --- | --- |
| 960 | 0.1077 | 0.1512 | 0.3985 |
| 768 | 0.0831 (−23%) | 0.1492 (−1.3%) | 0.3270 (−18%) |
| 640 | 0.0568 (−47%) | 0.1417 (−6.3%) | 0.2537 (−36%) |

**Read the divergence, not the averages.** mAP@50 barely moves while
mAP@[.5:.95] collapses — lower resolution still finds and classifies objects but
localises them coarsely. 768 is the defensible default; 640 only if the temporal
module tolerates loose boxes.

**Absolute mAP ~0.11 is a floor, not a quality statement.** AG annotates only
interaction-relevant objects, so correct detections of unlabelled real objects
score as false positives. Between-variant comparison is valid; the absolute
number is not a detector-quality claim.

### Other verified results

- **Graph parity**: the decomposed `Owlv2DetectionGraph` matches stock HF forward
  to 7.6e-06 on logits, bit-exact on boxes and objectness.
- **Engine vs fp32 reference**: cosine ≥0.9997, detection match rate 0.86–1.0.
- **GPU preprocessing**: 78.9 ms → 4.2 ms (**18.7×**), max diff 1.4e-06 vs the
  stock processor. End-to-end 87.5 ms → 23.8 ms.
- **int8 toolchain proven**: smoke test completed in 195 s, 202 Q/DQ pairs
  inserted. Mechanism works; only a real calibration run is missing.
- **Roofline**: peak fp16 matmul measured at 46.9 TFLOPS; the engine achieves
  52.9 TFLOPS effective. The engine is at the hardware limit on that GPU.

---

## 5. What is NOT done

| Item | State | Notes |
| --- | --- | --- |
| **`large_int8`** | **Not built** | The original Stage 1 goal. `large_fp32.onnx` is exported (1.2 GB); calibration and build never ran. |
| **`base_fp8`** | Calibration crashed twice | Code is ready (`to_fp8_onnx`). Needs RAM headroom + sm_89+. |
| Full-split eval | Only 1,200 of 68,183 frames | Subset was for speed of iteration. |
| Decode throughput | **Never measured** | `bench/streaming.py` written but unrun. If CPU decode caps below 87 FPS, the resolution trade-off is moot. |
| pad-value A/B | Never run | See gotcha 8 below. |
| `large_fp16` control | Never built | Needed to separate model size from precision — see gotcha 11. |

---

## 6. Gotchas — each of these cost real time

1. **TensorRT 11 is strongly-typed only.** `BuilderFlag.FP16`, `BuilderFlag.INT8`
   and `IInt8EntropyCalibrator2` **do not exist**. Precision is a property of the
   ONNX graph, not a builder flag: fp16 by dtype conversion, int8/fp8 by explicit
   Q/DQ nodes from calibrated PTQ.

2. **fp16 conversion leaves a stale `Cast(to=float32)`.** HF's
   `pixel_values.to(patch_embedding.weight.dtype)` traces with the dtype frozen at
   export time. A strongly-typed network then refuses to build
   (`input and kernel must be of same type`). Handled by
   `_retarget_stale_fp32_casts`, which only retargets casts whose consumers have
   fp16 weights — deliberately not a blanket rewrite.

3. **Never use `interpolate_pos_encoding` for TRT.** It emits a `Range` op that
   TRT requires in fp32, which then collides with fp16 neighbours. Use
   `retarget_resolution()` instead: it resamples the position grid once into the
   weights at export time, producing a static graph with zero `Range` nodes.

4. **HF caches the patch grid on the model, not in the config.**
   `num_patches_height`, `num_patches_width` and `box_bias` are set in
   `__init__`. Updating only `config.image_size` leaves the graph reshaping to the
   old grid. `retarget_resolution` updates all of them together.

5. **modelopt needs `get_first()` as well as `get_next()`**, and passing
   `calibration_data` as a dict slices **axis 0 of every array** as the batch —
   which turns the fixed rank-2 `query_embeds` into 53 bogus samples. Use
   `calibration_data_reader` instead.

6. **Set `high_precision_dtype="fp16"` when quantizing.** Otherwise unquantized
   regions stay fp32, and in a strongly-typed network that is what actually runs —
   quietly turning an "int8" engine into a mostly-fp32 one.

7. **Calibration must stream.** See section 1.

8. **HF pads with 0.0 (black); original OWLv2 pads with 0.5 grey.** Verified by
   pushing a white frame through and reading back the pad region. Exposed as
   `pad_value` on `GpuOwlv2Preprocessor`. **Untested A/B** — if HF's default is a
   regression it costs real mAP, and it would look like quantization damage.

9. **Calibrate across videos, not the first N frames.** `load(max_frames=N)`
   returns alphabetically-first frames, i.e. a handful of scenes. Use
   `_sample_across_videos`, which round-robins one frame per video.

10. **Calibration/eval split separation is enforced, not defaulted.** `prepare`
    refuses when `--calib-split == --eval-split`, writes a manifest of every frame
    used, and `evaluate` cross-checks it and reports `calibration_leakage.clean`.

11. **`base_fp16` vs `large_int8` confounds model size with precision.** Build
    `large_fp16` as a control before attributing any difference to either.

12. **Structural sparsity and FlashAttention are settled — do not revisit.**
    `BuilderFlag.SPARSE_WEIGHTS` exists but is inert on dense weights; benefiting
    requires 2:4 pruning *and* fine-tuning. FlashAttention is already in use: the
    engine's total device memory is 58 MB while a materialised fp16 score matrix
    at 960px would be 25.9 MB per head × 12 heads ≈ 311 MB, so TRT is necessarily
    using a fused tiled kernel.

---

## 7. Next steps, in priority order

### Step 1 — rebuild the fp16 baseline on the new GPU (~20 min)

Everything downstream compares against this, and engines do not transfer.

```bash
uv run sgg export  --variant base_fp16
uv run sgg prepare --variant base_fp16
uv run sgg build   --variant base_fp16
uv run sgg verify  --variant base_fp16 --video acgdataset/Charades_v1_480/001YG.mp4
uv run sgg bench   --variant base_fp16 --video acgdataset/Charades_v1_480/001YG.mp4
```

`verify` must report `mean_logit_cosine` ≥0.999. If not, stop — something in the
export or precision path is wrong and every later number is meaningless.

### Step 2 — finish the original Stage 1 question: `large_int8` (~1–2 h)

This is what Stage 1 was for, and it is the one thing still missing.

```bash
uv run sgg export  --variant large_int8
uv run sgg prepare --variant large_int8 --ag-root acgdataset \
                   --calib-split train --eval-split test --calib-frames 256
uv run sgg build   --variant large_int8
uv run sgg bench   --variant large_int8 --video acgdataset/Charades_v1_480/001YG.mp4
uv run sgg evaluate --variant large_int8 --ag-root acgdataset --split test
uv run sgg evaluate --variant base_fp16  --ag-root acgdataset --split test
uv run sgg report
```

Watch host RAM during `prepare`. Confirm the result's
`calibration_leakage.clean` is `true`.

Also build `large_fp16` as the control (gotcha 11) — otherwise any base-vs-large
difference cannot be attributed to size or precision.

### Step 3 — measure decode throughput (~15 min)

Cheap, and it may invalidate the whole resolution trade-off.

```python
from sggpipeline.bench.streaming import benchmark_decode_only, benchmark_streaming
```

If decode alone cannot sustain the engine's FPS, the detector is not the
bottleneck and 768/640 buys nothing.

### Step 4 — fp8 (~1 h, needs sm_89+)

Code is ready. The most promising remaining speed lever, and fp8's E4M3 exponent
handles transformer outlier activations far better than int8.

```python
from sggpipeline.detect.quantize import to_fp8_onnx   # see /tmp/fp8_calib.py pattern
```

### Step 5 — settle the open accuracy questions

- Full 68,183-frame evaluation instead of the 1,200-frame subset.
- pad-value A/B (0.0 vs 0.5) on a test subset.
- Resolution sweep re-run on the new GPU:
  `uv run python scripts/resolution_sweep.py --max-frames 5000`

### Then — Stage 2

`src/sggpipeline/tracking/` (association, identity descriptor, feature pooling)
and `tests/test_tracking.py` already exist in the tree; they were not part of the
Stage 1 work described here. Stage 2 in [plan.md](plan.md#L111) wants conventional
geometry/motion/appearance association compared against the learned identity
descriptor **on fixed detections** — which Step 1 above now provides.

---

## 8. Map of the code

```
src/sggpipeline/
  ag/classes.py          36-class AG vocabulary; multi-prompt expansion, max-pooled
  ag/dataset.py          annotation loader; derives vocabulary from data and warns on mismatch
  ag/extract_frames.py   Charades -> keyframes, resumable, 1-based indexing
  detect/owlv2.py        exportable graph, text-query encoding, retarget_resolution, postprocess
  detect/preprocess.py   stock HF processor (reference only — slow)
  detect/fast_preprocess.py  GPU preprocessing, 18.7x faster, matches reference to 1e-6
  detect/export_onnx.py  ONNX export; static shapes, optional resolution retarget
  detect/quantize.py     fp16 conversion, int8/fp8 PTQ, streaming CalibrationReader
  detect/trt_build.py    strongly-typed engine builder + timing cache
  detect/trt_runner.py   engine execution (torch-backed buffers) + eager reference runner
  bench/latency.py       CUDA-event timing, median/p95, stages kept separate
  bench/streaming.py     pipelined decode->preprocess->engine (UNRUN)
  bench/compare.py       comparison table + caveats
  evaluation/detection_eval.py  COCO mAP with per-class AP
  evaluation/verify.py   engine vs eager reference agreement
  cli.py                 extract-frames | validate-ag | export | prepare | build | bench | verify | evaluate | report
  tracking/              Stage 2 — pre-existing, not covered by this handoff
scripts/
  speed_experiments.py   latency/throughput across engines and batch sizes
  resolution_sweep.py    accuracy/FPS trade-off across input resolutions
```

---

## 9. One free thing to check on the new box

Local peak fp16 matmul measured **46.9 TFLOPS** with 20–27 W drawn against a
120 W cap — consistent with a laptop power-saving profile throttling everything.
On the cloud GPU, run the roofline check first:

```bash
uv run python -c "
import torch, time
a=torch.randn(8192,8192,device='cuda',dtype=torch.float16); b=a.clone()
for _ in range(10): a@b
torch.cuda.synchronize(); t=time.perf_counter()
for _ in range(50): a@b
torch.cuda.synchronize()
print(f'{2*8192**3*50/(time.perf_counter()-t)/1e12:.1f} TFLOPS fp16')"
```

Compare against the card's spec sheet. If it lands far below, fix that before
optimising anything else — it is worth more than every code change in this repo.
