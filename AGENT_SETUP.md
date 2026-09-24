# Agent setup & continue guide (remote GPU box)

You are picking up **Stage 1** of a video scene-graph pipeline on a fresh cloud
GPU instance. You have no history of the earlier work. This file is the complete
brief: setup, verification gates, task order, and the constraints that already
cost days to discover.

Read [HANDOFF.md](HANDOFF.md) for detail and [plan.md](plan.md) for the research
design. Read this file first.

**Goal of Stage 1:** measure how good and how fast a frozen OWLv2 detector is on
Action Genome, compiled to TensorRT. Nothing is trained. The deliverable is a
comparison table of accuracy and per-frame latency across variants.

---

## 0. Ground rules

- **Never train anything in Stage 1.** The detector is frozen. If a task seems to
  need fine-tuning, you have misread the task.
- **Never calibrate a quantized model on the `test` split.** The CLI refuses it,
  writes a manifest of every frame used, and `evaluate` cross-checks it. Do not
  work around the guard.
- **Do not copy `.plan` engine files between machines.** TensorRT engines are
  specific to one GPU, driver and TRT version. Always rebuild.
- **Report measured numbers only.** If a step fails, say so with the error. Do
  not infer a number you did not measure.
- Long jobs: run under `tmux` so a dropped SSH session does not kill them.

---

## 1. Verify the hardware first

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
nproc && free -g | head -2 && df -h /workspace | tail -1
```

Requirements: **compute_cap ≥ 8.9** for fp8 (an A100 at 8.0 cannot do fp8; Ada /
Hopper / Blackwell can). ≥64 GB host RAM. ≥100 GB free disk.

Then check the GPU is not silently throttled — this matters more than any code
change:

```bash
python3 -c "
import torch, time
a=torch.randn(8192,8192,device='cuda',dtype=torch.float16); b=a.clone()
for _ in range(10): a@b
torch.cuda.synchronize(); t=time.perf_counter()
for _ in range(50): a@b
torch.cuda.synchronize()
print(f'{2*8192**3*50/(time.perf_counter()-t)/1e12:.1f} TFLOPS fp16')"
```

Compare against the card's spec sheet. A reference RTX 5090 should be far above
the 46.9 TFLOPS measured on the old laptop GPU. If it lands near spec, proceed.

---

## 2. Environment

**Python must be 3.13.** Not 3.14: `nvidia-modelopt[onnx]` pins
`onnxruntime-gpu <1.25`, which has no cp314 wheels, making int8/fp8 unreachable.
`pyproject.toml` already pins `>=3.13,<3.14`.

```bash
cd /workspace/sggpipeline
uv sync
```

**Then force the GPU build of onnxruntime to win.** `onnxruntime` and
`onnxruntime-gpu` install into the same directory and collide; the CPU build
silently wins, removing the CUDA provider and making calibration unusably slow.

```bash
uv pip uninstall onnxruntime onnxruntime-gpu
uv pip install --reinstall onnxruntime-gpu==1.24.4
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

**Gate:** output must include `CUDAExecutionProvider`. If it shows only CPU, stop
and fix — every later quantization step depends on it.

Sanity-check the rest of the stack:

```bash
uv run python -c "
import torch, tensorrt as trt, modelopt
print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('trt', trt.__version__, '| modelopt', modelopt.__version__)
print('fp16 flag exists:', hasattr(trt.BuilderFlag,'FP16'))  # expect False on TRT 11
"
```

---

## 3. Data

Target layout under `/workspace/sggpipeline/acgdataset/`:

```
acgdataset/
  action_genome_v1.0/   person_bbox.pkl, object_bbox_and_relationship.pkl,
                        frame_list.txt, object_classes.txt, relationship_classes.txt
  Charades_v1_480/      9,848 .mp4
  frames/<video>.mp4/<nnnnnn>.png
```

### 3a. Annotations (285 MB) — must come from the user

Action Genome annotations sit behind a request form, so you cannot download them.
The user uploads this directory. If it is missing, **stop and ask** — nothing
downstream works without it.

### 3b. Charades videos (16 GB) — download here, do not transfer

The user's home uplink is ~3.4 Mbps (~18 h for the dataset). This box has
datacenter bandwidth, so fetch directly:

```bash
cd /workspace/sggpipeline/acgdataset
wget -c https://ai2-public-datasets.s3.amazonaws.com/charades/Charades_v1_480.zip
unzip -q Charades_v1_480.zip && rm Charades_v1_480.zip
ls Charades_v1_480/*.mp4 | wc -l    # expect 9848
```

If that URL 404s, the canonical landing page is
<https://prior.allenai.org/projects/charades> — verify before assuming it moved.

### 3c. Frames — regenerate here, do not transfer

11 GB / 87k small files. Faster to re-extract than to copy, and the extractor is
resumable and parallel.

```bash
uv run sgg extract-frames --ag-root acgdataset --videos-dir acgdataset/Charades_v1_480 \
                          --split test  --workers $(nproc)
uv run sgg extract-frames --ag-root acgdataset --videos-dir acgdataset/Charades_v1_480 \
                          --split train --max-videos 600 --workers $(nproc)
uv run sgg validate-ag --ag-root acgdataset --split test
```

**Gate:** `videos_with_errors` must be 0. Expect ~70,329 test frames (1,814
videos) and ~17,300 train frames (600 videos). `validate-ag` must report 36
classes and `frames_with_out_of_bounds_boxes: 0`.

---

## 4. Rebuild the fp16 baseline — do this before anything else

Everything downstream compares against it, and engines do not transfer.

```bash
uv run sgg export  --variant base_fp16
uv run sgg prepare --variant base_fp16
uv run sgg build   --variant base_fp16
uv run sgg verify  --variant base_fp16 --video acgdataset/Charades_v1_480/001YG.mp4
uv run sgg bench   --variant base_fp16 --video acgdataset/Charades_v1_480/001YG.mp4
```

**Gate:** `verify` must report `mean_logit_cosine ≥ 0.999` and
`mean_detection_match_rate ≥ 0.85`. If it does not, **stop.** Something in the
export or precision path is wrong and every later number is meaningless.

Reference (old RTX 5070 Ti Laptop, expect the 5090 to be faster): engine 19.00 ms
median at 960px, ~50 fps, 58 MB device memory.

---

## 5. Task order

### Task 1 — `large_int8` (the actual Stage 1 deliverable)

This is the one thing Stage 1 still owes. It was never completed because the
local machine ran out of host RAM.

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

**Gates:** the `prepare` summary must show non-zero `quantize_nodes`/
`dequantize_nodes` (zero means it silently produced an unquantized graph). The
`evaluate` result must show `calibration_leakage.clean: true`.

Watch host RAM during `prepare` — `large_fp32.onnx` is 1.2 GB and modelopt keeps
several copies.

### Task 2 — `large_fp16` control

`base_fp16` vs `large_int8` varies **model size and precision at once**, so
neither can be credited for a difference. Build `large_fp16` to separate them.
Add it to `DEFAULT_VARIANTS` in `src/sggpipeline/pipeline.py`.

### Task 3 — decode throughput (cheap, may invalidate other work)

`src/sggpipeline/bench/streaming.py` is written but has never been run. Use
`benchmark_decode_only` and `benchmark_streaming`. If CPU decode cannot sustain
the engine's FPS, the detector is not the bottleneck and resolution reduction
buys nothing.

### Task 4 — fp8 (needs compute_cap ≥ 8.9)

Code is ready: `to_fp8_onnx` in `src/sggpipeline/detect/quantize.py`. The most
promising remaining speed lever — fp8's E4M3 keeps exponent range, so the
outlier activations that hurt int8 on transformer residual paths degrade far
more gracefully. Calibrate exactly as int8 does (train split, streamed).

### Task 5 — settle open accuracy questions

- Full evaluation (68,183 test frames) rather than the 1,200-frame subset.
- **pad-value A/B**: transformers pads with `0.0`; original OWLv2 used `0.5`
  grey. Verified, never A/B'd. Exposed as `pad_value` on `GpuOwlv2Preprocessor`.
  If HF's default is a regression it costs real mAP and will look like
  quantization damage.
- Resolution sweep on this GPU: `uv run python scripts/resolution_sweep.py --max-frames 5000`

---

## 6. Constraints — do not rediscover these

1. **TensorRT 11 is strongly-typed only.** `BuilderFlag.FP16`, `BuilderFlag.INT8`
   and `IInt8EntropyCalibrator2` **do not exist**. Precision is a property of the
   ONNX graph: fp16 by dtype conversion, int8/fp8 by Q/DQ nodes from calibrated
   PTQ. If you find yourself looking for a precision flag, re-read this.
2. **Stale `Cast(to=float32)` after fp16 conversion.** HF's
   `pixel_values.to(weight.dtype)` freezes the dtype at trace time; a
   strongly-typed build then fails with "input and kernel must be of same type".
   Handled by `_retarget_stale_fp32_casts` — it deliberately only retargets casts
   whose consumers hold fp16 weights. Do not make it a blanket rewrite.
3. **Never use `interpolate_pos_encoding` for TensorRT.** It emits a `Range` op
   TRT requires in fp32, colliding with fp16 neighbours. Use
   `retarget_resolution()`, which resamples the position grid into the weights at
   export time (static graph, zero `Range` nodes).
4. **HF caches the patch grid on the model.** `num_patches_height`,
   `num_patches_width` and `box_bias` are set in `__init__`, not read from config
   per call. Updating only `config.image_size` leaves the graph reshaping to the
   old grid. `retarget_resolution` moves all of them together.
5. **modelopt's calibration interface**: needs `get_first()` as well as
   `get_next()`. Passing `calibration_data` as a dict slices **axis 0 of every
   array** as batch, turning the rank-2 `query_embeds` into bogus samples — use
   `calibration_data_reader`.
6. **Always set `high_precision_dtype="fp16"` when quantizing.** Otherwise
   unquantized regions stay fp32, and in a strongly-typed network that is what
   runs — quietly making an "int8" engine mostly fp32.
7. **Calibration must stream.** Materializing 384 frames at 960² fp32 is 3.96 GB
   resident and crashed the previous machine. `CalibrationReader` takes
   `image_paths` + a `loader`; keep it that way.
8. **Calibrate across videos, not the first N frames.** `load(max_frames=N)`
   returns alphabetically-first frames — a handful of scenes. Use
   `_sample_across_videos`.
9. **Sparsity and FlashAttention are settled — do not revisit.**
   `BuilderFlag.SPARSE_WEIGHTS` exists but is inert on dense weights (needs 2:4
   pruning *and* fine-tuning). FlashAttention is already active: the engine uses
   58 MB device memory while a materialized fp16 score matrix at 960px would be
   ~311 MB for one layer, so TRT is necessarily using a fused tiled kernel.

### Verified facts about Action Genome — do not re-derive

- Frame numbers are **1-based** (ffmpeg `image2`).
- Object boxes are `(x, y, w, h)`; person boxes are `xyxy`.
- Boxes are already in **native frame coordinates**; `bbox_size` matches each
  video's own resolution. No rescaling.
- 35 objects + person = 36. The **pickles use slash forms** (`cup/glass/bottle`);
  `object_classes.txt` strips them. `AG_OBJECT_CLASSES` matches the pickles.
- Splits are disjoint: zero video overlap, zero frame overlap.

---

## 7. Interpreting results

- **Absolute mAP ~0.11 is a floor, not a quality statement.** AG annotates only
  interaction-relevant objects, so correctly finding an unlabelled real object
  scores as a false positive. Between-variant comparison is valid; the absolute
  number is not a detector-quality claim. Say this whenever you report mAP.
- **`person` AP measures agreement with a detector**, not with human annotation —
  AG's person boxes are themselves detector output.
- **Watch mAP@50 against mAP@[.5:.95] separately.** On the resolution sweep they
  diverged sharply (768px: −1.3% vs −23%), which showed that lower resolution
  still finds objects but localizes them coarsely. An average would have hidden
  that.

---

## 8. Code map

```
src/sggpipeline/
  ag/classes.py              36-class vocabulary, multi-prompt, max-pooled
  ag/dataset.py              annotation loader, derives vocabulary from data
  ag/extract_frames.py       Charades -> keyframes, resumable, 1-based
  detect/owlv2.py            exportable graph, text queries, retarget_resolution, postprocess
  detect/fast_preprocess.py  GPU preprocessing (18.7x faster, matches HF to 1e-6)
  detect/preprocess.py       stock HF processor — reference only, slow
  detect/export_onnx.py      ONNX export, static shapes
  detect/quantize.py         fp16 convert, int8/fp8 PTQ, streaming CalibrationReader
  detect/trt_build.py        strongly-typed builder + timing cache
  detect/trt_runner.py       engine execution + eager reference runner
  bench/latency.py           CUDA-event timing, stages kept separate
  bench/streaming.py         pipelined decode->preprocess->engine (UNRUN)
  bench/compare.py           comparison table + caveats
  evaluation/detection_eval.py  COCO mAP, per-class AP
  evaluation/verify.py       engine vs eager reference
  cli.py                     extract-frames|validate-ag|export|prepare|build|bench|verify|evaluate|report
  tracking/                  Stage 2 — pre-existing, out of scope here
scripts/speed_experiments.py latency/throughput across engines and batches
scripts/resolution_sweep.py  accuracy/FPS across input resolutions
```

Results land in `artifacts/results/*.json`; `uv run sgg report` renders the
comparison table with its caveats.

---

## 9. When you finish a task

Report: the command run, the measured numbers, whether each gate passed, and
anything that failed with its actual error. Commit with a message describing what
was measured. Do not push engine or ONNX files — `.gitignore` already excludes
`artifacts/`, `acgdataset/` and `.venv/`.
