# Step 2: tracking and identity

The implementation lives in `src/sggpipeline/tracking/`, independently of the
Stage 1 detector. Synthetic CPU tests cover association, identity training and
the command-line workflow. The streaming detector now supplies aligned frozen features. Real-data identity
training and evaluation still require trusted instance correspondences.

## What is implemented

- Causal tracking using box overlap, timestamp-based constant-velocity motion,
  appearance similarity and global assignment.
- Track expiry and conservative rejection of ambiguous matches. Rejected old
  identities are retired so downstream memory can be cleared.
- A small identity adapter and cross-frame contrastive training. Training batches
  contain different physical instances of the same category.
- Fixed-detection comparison of frozen appearance versus learned identity using
  identical tracker settings. Semantic labels and scores remain detector outputs.
- Pickle-free feature caches, checkpoint provenance and video-split checks.

## Stage 1 handoff

`AssociationTracker.update(detections, features, timestamp)` accepts the existing
`sggpipeline.detect.owlv2.Detections` object and a finite, nonzero `(N,D)` appearance
array aligned exactly with its detection rows. Boxes are native-image xyxy pixels;
timestamps are strictly increasing seconds. Call it on empty frames too, with
features shaped `(0,D)`. Use consistent image coordinates within each video.

```python
from sggpipeline.tracking import AssociationTracker

tracker = AssociationTracker()
result = tracker.update(detections, features, timestamp_seconds)
# result.track_ids aligns with detections.boxes
# result.is_new identifies newly assigned identities
# result.uncertain marks observations whose ambiguous association was rejected
# result.expired_ids includes expired and conservatively retired identities
```

Use `(video_id, track_id)` as the global identity key. Reset the tracker between
videos. Clear downstream pair memory for retired IDs. A missing observation does
not establish that an interaction ended, and correct identity alone does not
retire an outdated relationship.

Build the feature-exporting engines with `scripts/build_engines.py --features`.
`StreamingDetector` returns `FrameResult.detections` and row-aligned
`FrameResult.features` (CPU NumPy), ready for `tracker.update(...)`; the same
features remain on the GPU for the relationship head. These are descriptors of
the patches that produced each detection, requiring no second encoder.
`tracking.features.pool_box_features(...)` remains available for experiments
using a shared visual feature map, with the OWLv2 padding convention. Include
feature extraction, transfer, and association in end-to-end tracking timing.

Hold proposal filtering, duplicate suppression and detection budgets fixed across
controls. Multiple detections of one object must not become multiple trusted
observations of that instance in one frame.

## Cache format

Write one `tracking.cache.DetectionCache` per video using `.save(path)`.
The compressed NPZ contains:

| Field | Shape / meaning |
| --- | --- |
| `metadata` | JSON scalar, written by `.save()` |
| `timestamps` | `(F,)` seconds, including empty frames |
| `offsets` | `(F+1,)` integers; frame f occupies rows `[offsets[f]:offsets[f+1]]` |
| `boxes` | `(N,4)` finite, positive-area native xyxy boxes |
| `scores`, `objectness` | `(N,)` probabilities |
| `labels` | `(N,)` nonnegative integer semantic labels |
| `features` | `(N,D)` frozen appearance features in detection order |
| `instance_ids` | `(N,)` integer physical-instance IDs scoped to the video; `-1` unknown |

Metadata requires `schema_version: 1`, a stable `video_id`, `split` (`train`,
`val`, or `test`), `feature_source`, and `identity_source` (`none`, `human`, or
`reviewed_pseudo`). Make `feature_source` identify the checkpoint, resolution,
feature layer, pooling and vocabulary convention so incompatible caches are
not combined. Record additional extraction settings in metadata as needed.

Identity annotations must correspond to physical objects across frames, with at
most one known detection per instance per frame. Category labels and unreviewed
tracker IDs are not trusted identity supervision. Mark unknown and duplicate
proposals `-1`. A cache declaring `identity_source: none` must have all IDs unknown.
The current AG loader does not supply these correspondences automatically.

## Run

Commands use CPU and add no dependencies to the existing environment. Limiting
CPU threads keeps the small training job from competing heavily with Stage 1.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m sggpipeline.tracking train \
  --caches artifacts/tracking/train/*.npz \
  --output artifacts/tracking/identity.pt --steps 500

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m sggpipeline.tracking evaluate \
  --caches artifacts/tracking/val/*.npz \
  --checkpoint artifacts/tracking/identity.pt \
  --output artifacts/results/tracking_val.json
```

Omit `--checkpoint` to evaluate the frozen-appearance baseline alone. Both
evaluation modes require trusted correspondences. Training needs at least two
same-category instances with two labelled frames each. Video IDs must be stable
across splits: evaluation rejects videos recorded in checkpoint training metadata.
Validation and test caches cannot be pooled in one report.

`evaluate --config path.json` accepts fields of `TrackerConfig`, applied unchanged
to both variants. Threshold defaults are experimental. Choose them on validation
videos and freeze them before test evaluation. Pseudo-label-based measurements
must be reported as such and are not substitutes for independent human labels.

Reports include ID switches, cross-instance transfers, fragmentation, per-video
predicted IDs and median/p95 CPU frame timing. These are association measurements
conditional on labelled detection rows, not full MOT benchmark metrics. Unknown
rows still participate in tracking but do not contribute ground-truth metrics.
Timing covers descriptor transformation and tracking, including first-frame
overhead; it excludes detection and cache I/O and is not a deployment speed claim.
Real occlusion/swap subsets, downstream relation accuracy and end-to-end timings
remain needed to assess the research claim.

## Verify

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s tests -p 'test_tracking.py' -v
```

The 15 synthetic tests cover assignment against brute force, same-class swaps,
motion, missing observations, expiry, ambiguity retirement, category changes,
pooling coordinates, contrastive gradients, cache validation, fixed inputs,
training updates, metrics and the checkpoint/evaluation workflow. Passing these
tests establishes implementation behavior, not real-video tracking quality.

## BoT-SORT using OWL features directly

`OWLBoTSORT` uses the installed Ultralytics BoT-SORT implementation with
`with_reid=True` and `model="auto"`: the supplied appearance features pass
through without loading a YOLO detector or another ReID encoder. This adapter
has been tested with Ultralytics 8.4.165 and LAP 0.5.13. LAP is declared in the
optional `tracking` extra; install it with `uv sync --extra tracking`.

```python
from sggpipeline.tracking import OWLBoTSORT

tracker = OWLBoTSORT()
# detections: Stage 1 Detections (boxes, scores, labels, objectness)
# owl_features: one visual descriptor per detection, shape (N,D)
features = owl_features.detach().float().cpu().numpy()  # if a Torch tensor
result = tracker.update(detections, features, timestamp_seconds)
assigned = result.track_ids >= 0
```

Any consistent visual feature dimension is accepted. The adapter normalizes
features and uses them for BoT-SORT appearance association alongside geometry,
Kalman motion prediction and high/low-confidence matching. Supply object-aligned
visual features, not text query embeddings or category IDs. Direct features need
no identity training to run; the learned identity head remains an optional
experiment. Detector features are not guaranteed to separate identical objects.
Their appearance thresholds need validation on OWL outputs.

OWL-ViT and OWLv2 are different checkpoints/architectures; the tracker accepts
per-object features from either. The existing pooling helper assumes the
OWLv2 preprocessing used in this repository. For another detector/preprocessor,
preserve its own token-to-image coordinate mapping. The feature-exporting streaming engines already provide aligned descriptors;
see the Stage 1 handoff above.

To evaluate existing regularly sampled feature caches:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m sggpipeline.tracking evaluate \
  --tracker botsort --caches artifacts/tracking/val/*.npz \
  --output artifacts/results/botsort_owl_features.json
```

Add `--checkpoint artifacts/tracking/identity.pt` to compare the original OWL
features with learned descriptors using the same BoT-SORT settings. With
`--tracker botsort`, the optional JSON `--config` uses `BoTSORTConfig` fields.
The original custom tracker remains available as `--tracker baseline`.

Behavior to account for:

- Inputs require detection scores as well as boxes and aligned features.
  Retain low-confidence candidates; discarded detections cannot aid recovery.
  OWL score calibration differs from other detectors, so tune thresholds.
- This features-only mode disables camera-motion compensation and requires no
  image input. It does not make a camera-motion-correction claim.
- The upstream motion model uses fixed update steps. The adapter checks regular
  timestamp spacing (0.1% relative tolerance). Call on empty frames too.
  Irregularly spaced AG keyframes cannot be replayed directly; use a uniformly
  sampled video/cache or the timestamp-aware baseline.
- `track_buffer` is measured in updates. At 10 updates/second, 30 updates
  correspond to approximately 3 seconds. Tune for the chosen sampling rate.
- Output IDs align with input detections. A value of `-1` means filtered or
  unconfirmed; do not use it as an object identity or write it into pair memory.
  Detector semantics stay unchanged. Association itself is not class-gated.
- BoT-SORT does not implement the custom baseline's ambiguity rejection.
  Its result's `uncertain` flags are placeholders, not uncertainty estimates;
  comparison reports explicitly mark the estimates unavailable.
- Expired IDs include retired tracks; the adapter prunes upstream removed-state
  entries immediately to prevent their accidental reactivation next frame.
- Reports include assignment coverage and unassigned counts alongside switches.
  A tracker dropping every detection must not look good merely because it has
  zero ID switches. These still are conditional association diagnostics,
  not full MOT benchmark scores.

Run both the baseline tests and the eight BoT-SORT integration tests:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s tests -p 'test_*.py' -v
```

The BoT-SORT tests use the real installed implementation. They verify that changing
only supplied features changes association, no separate ReID model is constructed,
low-confidence matching preserves detection indices, tentative observations remain
unassigned, expired identities stay retired, and separate tracker instances keep
independent ID allocation.

Implementation reference: [Ultralytics BoT-SORT API](https://docs.ultralytics.com/reference/trackers/bot_sort/).
