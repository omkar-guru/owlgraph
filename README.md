# sggpipeline — Stage 1: detector benchmarking

Stage 1 of the plan in [plan.md](plan.md): establish how good, and how fast,
a frozen OWLv2 detector is on Action Genome before any relationship or temporal
machinery is built on top of it.

This stage compares two compiled variants:

| Variant | Checkpoint | Input | Precision |
| --- | --- | --- | --- |
| `base_fp16` | `google/owlv2-base-patch16-ensemble` | 960×960 | fp16 |
| `large_int8` | `google/owlv2-large-patch14-ensemble` | 1008×1008 | int8 (calibrated PTQ) |

Both are compiled to TensorRT and scored with COCO mAP against AG ground truth.

## Environment

Python **3.13** (not 3.14 — `nvidia-modelopt[onnx]` pins `onnxruntime-gpu <1.25`,
which has no cp314 wheels, and int8 is unreachable without it).

```bash
uv sync
```

Verified on: RTX 5070 Ti Laptop (sm_120, 12GB), torch 2.14.0+cu130,
TensorRT 11.3.0.99, modelopt 0.46.1.

> TensorRT 11 builds **strongly-typed** networks only. `BuilderFlag.FP16`,
> `BuilderFlag.INT8` and `IInt8EntropyCalibrator2` no longer exist. Precision is
> a property of the ONNX graph handed to the builder, not a builder flag — fp16
> via dtype conversion, int8 via explicit Q/DQ nodes from calibrated PTQ.

## Data layout

`--ag-root` must point at a directory shaped like:

```
<ag-root>/
  annotations/
    person_bbox.pkl
    object_bbox_and_relationship.pkl
  frames/
    <video_id>.mp4/
      000089.png
      ...
```

Frames are dumped from the Charades videos (AG's `dump_frames.py`). Check the
download before trusting any score from it:

```bash
uv run sgg validate-ag --ag-root /path/to/ag --split test
```

That reports the vocabulary it found, box counts per class, and whether any
boxes fall outside their frame — the usual symptom of a coordinate-space
mismatch.

## Running a variant

```bash
uv run sgg export   --variant base_fp16                      # checkpoint -> fp32 ONNX
uv run sgg prepare  --variant base_fp16                      # -> fp16 ONNX
uv run sgg build    --variant base_fp16                      # -> TensorRT engine
uv run sgg bench    --variant base_fp16 --video clip.mp4      # per-frame latency
uv run sgg evaluate --variant base_fp16 --ag-root /path/to/ag # AG mAP
```

int8 additionally needs calibration frames, taken from the **train** split so the
test split stays untouched:

```bash
uv run sgg prepare --variant large_int8 --ag-root /path/to/ag \
                   --calib-split train --calib-frames 256
```

Results are written to `artifacts/results/*.json`.

## Reading the numbers

**mAP is absolutely understated.** AG only annotates objects involved in an
annotated interaction, so a detector is penalised for correctly finding real
objects AG chose not to label. The comparison *between* variants remains valid —
both are penalised identically — but the absolute value is not a statement about
detection quality.

**The `person` class measures agreement with a detector, not with a human.** AG's
person boxes are themselves detector output. Per-class AP is reported so this
stays visible instead of being averaged away.

**`base_fp16` vs `large_int8` confounds two variables** — model size and
precision. If the goal is to attribute a difference to either one, build
`large_fp16` as a control.

**Latency is reported in three parts**: CPU preprocessing, host-to-device copy,
and engine time, with median and p95. They are kept separate because
preprocessing is shared by every variant and folding it in compresses the
measured gap between them.

## Layout

```
src/sggpipeline/
  ag/          AG vocabulary and ground-truth loader
  detect/      OWLv2 graph, ONNX export, fp16/int8 preparation, TRT build+run
  bench/       latency measurement
  evaluation/  COCO mAP scoring and engine-vs-reference verification
  cli.py       command line entry point
```
