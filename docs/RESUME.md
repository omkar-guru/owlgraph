# Resume entry

**Fast Open-Vocabulary Scene Graph Pipeline — Independent Project (In Progress)**

Python · PyTorch · TensorRT · CUDA · Transformers

- Built a streaming OWLv2 detection and relationship-prediction pipeline (TensorRT, CUDA graphs); benchmarked **4.9 ms/frame on an RTX 5090 with GPU-resident inputs**.
- Accelerated the detector **~1.5× (204 → 316 FPS sustained)** with selective, previous-frame-guided token merging, retaining **98.6% of baseline detection mAP across 68,183 Action Genome test frames** with detection-aware protection.
- Designed a directed relationship head (pair routing + text-embedding classifiers) costing **0.29 ms/frame**, which keeps **~98% of its predicate accuracy on object classes never seen in training**; evaluated against SG-ViT on VG150 with a ported reference evaluator, and integrated appearance-based tracking (BoT-SORT).

## Interview context

Timing excludes video decoding and tracking. Detection mAP was 0.1064 versus
0.1079 unmerged; 98.6% is relative retention, not absolute detection accuracy.
The 204 -> 316 FPS figures are the unmerged and merged engines without box
protection; protection adds 0.23 ms/frame and is what gives the 98.6%. The
~98% held-out-object figure is predicate AP 0.328 vs 0.333 with ground-truth
boxes, four held-out classes, single seed. VG150 mean recall is below SG-ViT's
(different training: SG-ViT fine-tunes its whole model; this head runs on a
frozen detector).
Tracking has synthetic and integration tests, but real-video identity evaluation
is pending. Temporal attention/Mamba, JEPA, and graph QA are future work.
Open-vocabulary object transfer has supporting experiments; broad unseen-action
recognition does not. See [the experiment record](../RESULTS.md) for protocols
and limitations. Add your actual dates and repository URL when using this entry.
