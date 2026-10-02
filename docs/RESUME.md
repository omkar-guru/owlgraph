# Resume entry

**Fast Open-Vocabulary Scene Graph Pipeline — Independent Project (In Progress)**

Python · PyTorch · TensorRT · CUDA · Transformers

- Built a streaming OWLv2 detection and relationship-prediction pipeline; benchmarked **4.9 ms/frame on an RTX 5090 with GPU-resident inputs**.
- Implemented selective token merging and detection-aware protection, retaining **98.6% of baseline detection mAP across 68,183 Action Genome test frames**.
- Developed directed relationship heads and evaluated open-vocabulary generalization on Action Genome and VG150; implemented appearance-based tracking and BoT-SORT integration.

## Interview context

Timing excludes video decoding and tracking. Detection mAP was 0.1064 versus
0.1079 unmerged; 98.6% is relative retention, not absolute detection accuracy.
Tracking has synthetic and integration tests, but real-video identity evaluation
is pending. Temporal attention/Mamba, JEPA, and graph QA are future work.
Open-vocabulary object transfer has supporting experiments; broad unseen-action
recognition does not. See [the experiment record](../RESULTS.md) for protocols
and limitations. Add your actual dates and repository URL when using this entry.
