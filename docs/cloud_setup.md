# Cloud VM quick setup for an agent

Run this guide on the Linux NVIDIA GPU VM; start by cloning the repository below. It reflects
the baseline setup inspected on 2026-09-24, with status updated on 2026-10-02. Read `README.md` for the detector workflow,
`docs/history/HANDOFF.md` for historical measurements, and `plan.md` for the proposed research
architecture. Treat source code as authoritative when those documents disagree.

## What you are taking over

The implemented main path benchmarks frozen OWLv2 object detection on Action
Genome: checkpoint → ONNX → precision conversion/calibration → TensorRT →
latency, reference parity, and COCO mAP. `base_fp16` is the first baseline;
`large_int8` is the unfinished comparison. `tracking/` separately implements
fixed-detection association and learned identity experiments with CPU tests.
Relationship prediction is implemented and evaluated on Action Genome and
VG150, including a streaming runtime (see README.md and RESULTS.md). Temporal
Mamba/attention, JEPA, and graph QA remain proposed work. The commands below
reproduce the detector baseline; they do not reproduce the full relationship
experiments or validate real-video tracking.

## 1. Prepare the machine and checkout

Use the existing provisioned GPU VM. The historical handoff budgets **64 GB host
RAM, 24 GB+ VRAM, and 100 GB+ disk** for large-model calibration; these are planning
figures, not a guarantee that every configuration fits. Start with FP16.

Have Git, `uv`, `curl`, `unzip`, and an NVIDIA driver available. Clone the
configured GitHub repository directly onto the VM:

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/omkar-guru/sggpipelinetest.git sggpipeline
cd sggpipeline
```

If the repository requires authentication, use the VM's configured GitHub
credentials, or clone `git@github.com:omkar-guru/sggpipelinetest.git` with an
SSH key authorized for this repository. For an existing checkout, enter its
root, inspect `git status --short` and `git remote -v`, then run
`git pull --ff-only` when the working tree is clean. Preserve local changes;
do not reset them to force an update.

Only pushed commits are available through GitHub. Local code changes and this
guide must be committed and pushed before the VM can retrieve them; the setup
instructions themselves do not publish local changes.

Download the dataset directly on the VM in section 2. `.venv/`, `artifacts/`,
and `acgdataset/` are ignored by Git. Rebuild TensorRT engines and timing caches
on the target machine. The first export also downloads Hugging Face models.
All remaining commands run from the repository root, in the same shell.

```bash
git status --short
git rev-parse HEAD
nvidia-smi
free -h
df -h .
uv sync --locked --python 3.13
```

Python is constrained to **3.13** by `pyproject.toml`. Preserve `uv.lock`.

The lock includes CPU and GPU ONNX Runtime distributions, which share an import
namespace. Apply the handoff's GPU-package repair in this environment:

```bash
uv pip uninstall --python .venv/bin/python onnxruntime onnxruntime-gpu
uv pip install --python .venv/bin/python --reinstall onnxruntime-gpu==1.24.4
```

Use `.venv/bin/python` and `.venv/bin/sgg` below so automatic synchronization does
not undo that repair. If you run `uv sync` again, repeat the repair and checks.

```bash
.venv/bin/python - <<'PY'
import sys
import torch
import tensorrt as trt
import onnxruntime as ort
print('Python:', sys.version)
print('Torch / CUDA:', torch.__version__, torch.version.cuda)
print('TensorRT:', trt.__version__)
print('ORT:', ort.__version__, ort.get_available_providers())
assert torch.cuda.is_available(), 'CUDA is unavailable to PyTorch'
print('GPU:', torch.cuda.get_device_name(0))
x = torch.ones(16, device='cuda')
assert x.sum().item() == 16
assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'Repair ORT GPU installation'
PY
.venv/bin/sgg --help
.venv/bin/python -m unittest discover -s tests -v
```

ORT provider availability is only a preliminary check; a real calibration run
must also load its CUDA libraries successfully. Record actual versions and
resolve driver/library errors before launching long jobs.

## 2. Download and validate data on the VM

Download the annotations from the Google Drive folder linked by the official
[Action Genome repository](https://github.com/JingweiJ/ActionGenome), and the
480p video archive linked by the official
[Charades page](https://prior.allenai.org/projects/charades). No SCP is needed.
The video ZIP endpoint returned HTTP 200 when checked; its compressed size is
about **16.3 GB**. Allow space for both the ZIP and extracted videos, plus frames,
models, and the environment.

```bash
export AG_ROOT="$PWD/acgdataset"
export VIDEOS_DIR="$AG_ROOT/Charades_v1_480"
mkdir -p "$AG_ROOT/downloads" "$AG_ROOT/action_genome_v1.0"

# Isolated download tool; does not change the project's environment or lockfile.
uvx --from gdown gdown --folder --continue \
  'https://drive.google.com/drive/folders/1LGGPK_QgGbh9gH9SDFv_9LIhBliZbZys' \
  -O "$AG_ROOT/action_genome_v1.0/"

# Resume an interrupted download by rerunning this command.
curl --fail --location --retry 5 --continue-at - \
  --output "$AG_ROOT/downloads/Charades_v1_480.zip" \
  'https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip'

# Check archive integrity before extraction; stop if this fails.
unzip -tq "$AG_ROOT/downloads/Charades_v1_480.zip" && \
  unzip -nq "$AG_ROOT/downloads/Charades_v1_480.zip" -d "$AG_ROOT"

test -s "$AG_ROOT/action_genome_v1.0/person_bbox.pkl"
test -s "$AG_ROOT/action_genome_v1.0/object_bbox_and_relationship.pkl"
test -s "$AG_ROOT/action_genome_v1.0/frame_list.txt"
test -d "$VIDEOS_DIR"
```

Stop on a failed download or file check. The expected layout is below; if an
upstream archive adds a nesting level, locate the extracted files and adjust
paths before proceeding. Google Drive may rate-limit downloads: retain completed
files, retry later, and report a persistent quota/access error.
[`gdown` documentation](https://github.com/wkentaro/gdown) describes folder
and resume support. Do not treat an HTML error page as annotation data.

```text
acgdataset/
  action_genome_v1.0/
    person_bbox.pkl
    object_bbox_and_relationship.pkl
    frame_list.txt
  Charades_v1_480/
    <video_id>.mp4
  frames/                       # generated below
    <video_id>.mp4/<nnnnnn>.png
```

An `annotations/` directory is also accepted instead of `action_genome_v1.0/`.
Generate only the needed annotated frames on the VM:

```bash
export AG_ROOT="$PWD/acgdataset"
export VIDEOS_DIR="$AG_ROOT/Charades_v1_480"
.venv/bin/sgg extract-frames --ag-root "$AG_ROOT" --videos-dir "$VIDEOS_DIR" --split test --workers 8
.venv/bin/sgg extract-frames --ag-root "$AG_ROOT" --videos-dir "$VIDEOS_DIR" --split train --max-videos 600 --workers 8
.venv/bin/sgg validate-ag --ag-root "$AG_ROOT" --split test
```

Extraction is resumable. Require zero extraction errors and inspect
`frames_missing_image` and box-bound validation before scoring. Evaluation
silently filters frames without images, so an incomplete dataset is not a
full-split result. Keep default 1-based frame indexing. Object boxes are XYWH;
person boxes are XYXY; the loader handles conversion.

## 3. Rebuild and check the FP16 baseline

Set `CLIP` to an existing, nonempty Charades video before running:

```bash
export CLIP="$VIDEOS_DIR/001YG.mp4"
test -s "$CLIP"
.venv/bin/sgg export --variant base_fp16
.venv/bin/sgg prepare --variant base_fp16
.venv/bin/sgg build --variant base_fp16
.venv/bin/sgg verify --variant base_fp16 --video "$CLIP"
.venv/bin/sgg bench --variant base_fp16 --video "$CLIP"
.venv/bin/sgg evaluate --variant base_fp16 --ag-root "$AG_ROOT" --split test --max-frames 100
.venv/bin/sgg report
```

Inspect `artifacts/results/verify_base_fp16.json`: the historical FP16 acceptance
criterion is `mean_logit_cosine >= 0.999`. Stop and investigate a failure before
comparing performance. The 100-frame evaluation is a smoke test, not a
representative score (`--max-frames` takes the first frames). For full evaluation:

```bash
.venv/bin/sgg evaluate --variant base_fp16 --ag-root "$AG_ROOT" --split test
```

Results are under `artifacts/results/`; repeated runs overwrite each variant's
files. Archive smoke and full results separately, together with commit, GPU,
driver, package versions, commands, and evaluated frame count. `bench` measures
preprocessing and inference stages; it does not establish video decode throughput.
AG labels only interaction-relevant objects, which limits interpretation of
absolute detector mAP. Re-measure on this GPU; laptop timings are historical.

## 4. Continue only after baseline success

**Fix INT8 calibration memory use first.** `CalibrationReader` supports streaming
via `image_paths` and `loader`, but `cli.py::_prepare_int8` still loads every image,
preprocesses the entire batch, and passes an array. Wire this call site to the
streaming interface before a full calibration run. Keep per-frame input shape
`(1, 3, S, S)`, the fixed query embeddings, cross-video sampling, and the exact
calibration manifest. Verify the change on a small calibration run first.

After that fix, the intended commands are:

```bash
.venv/bin/sgg export --variant large_int8
.venv/bin/sgg prepare --variant large_int8 --ag-root "$AG_ROOT" --calib-split train --eval-split test --calib-frames 256
.venv/bin/sgg build --variant large_int8
.venv/bin/sgg bench --variant large_int8 --video "$CLIP"
.venv/bin/sgg evaluate --variant large_int8 --ag-root "$AG_ROOT" --split test
.venv/bin/sgg report
```

Watch host RAM and VRAM during calibration. Require a saved calibration manifest
and `calibration_leakage.checked: true`, `calibration_leakage.clean: true` in the
INT8 evaluation report. Never calibrate on test frames.

Other current boundaries:

- `base_fp8` is registered, but `sgg prepare` rejects FP8. A Python quantization
  helper exists; CLI integration and target GPU compatibility need work.
- `large_fp16` is not registered. Add this control before attributing a
  base-FP16 versus large-INT8 difference to quantization alone.
- TensorRT precision comes from ONNX types/Q-DQ nodes. Preserve the existing
  strongly typed build and conversion fixes in `detect/`.
- Tracking entry point: `.venv/bin/python -m sggpipeline.tracking --help`.
  Training/evaluation require `DetectionCache` NPZ files with frozen detections,
  features, timestamps, and trusted instance correspondences. There is no
  end-to-end detector-to-cache CLI. AG category labels are not instance IDs.
- `docs/history/HANDOFF.md` contains stale claims: the repository now has commits,
  `acgdataset/` is already ignored, and calibration is not streamed by the CLI.
  Its `/tmp/fp8_calib.py` reference is not a repository-provided script.

Setup is complete when environment checks and tracking tests pass, test data
validates, and the rebuilt FP16 engine passes parity and produces benchmark and
smoke-evaluation JSON. Report any failed step with the exact command and error.
