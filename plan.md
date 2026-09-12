Fast open-vocabulary
video scene graphs

SG-ViT-style relationships • Mamba temporal memory • JEPA training
Identity, semantics and evidence-based answering

Architecture summary | 8 September 2026 | Proposed experiments, not validated results

**Recommended starting point:** keep the pretrained OWLv2 detector, build a lightweight relationship baseline, then test temporal attention versus Mamba and predictive training versus no predictive training.

The goal is a fast system that detects objects, follows the same instances through video, recognises their interactions and answers questions using a timestamped graph. Inference cost is the primary efficiency budget; inherited pretraining and additional training compute must still be reported.

## Four variants of one shared pipeline

| Variant | Temporal processing | JEPA objective | Question being tested |
| --- | --- | --- | --- |
| A | Causal attention | Off | Establish a temporal baseline |
| B | Same attention | On | Better features at unchanged inference cost? |
| C | Causal Mamba | Off | Better temporal accuracy–cost trade-off? |
| D | Same Mamba | On | Do both changes help together? |

**Naming:** A–D here label the temporal experiments. All four retain the earlier detector-first “Option A” strategy; this document does not reopen the original detector-pretraining pathways.

## What each component contributes

* **SG-ViT-style head:** cheaply extracts directed subject–object relationship features from shared visual tokens.
* **Identity head:** helps associate observations of the same physical object across time.
* **Semantic head:** matches object features to category and attribute descriptions.
* **Mamba:** a candidate temporal memory architecture. **JEPA:** an additional predictive learning objective; it can train attention or Mamba.

The baseline is inspired by SG-ViT, not an exact reproduction: it reuses an object-detector checkpoint and adds video memory and separate head roles. [1, 2]

# Shared perception: relationships, identity and semantics

Begin with google/owlv2-base-patch16-ensemble. Reuse its detector features, boxes and text-aligned object scores. Add small branches rather than another full visual encoder for each task. [2]

![Shared detector features branch to semantic matching, instance identity association, and directed object-pair features.](data:image/png;base64...)

Figure 1. Proposed head roles. The identity descriptor assists association; the tracker assigns persistent IDs.

**SG-ViT contribution.** Its directed subject and object projections produce pair scores; hard top-k selection limits which pair embeddings are classified. The original embedding combines the two role features, and predicate classification compares it with text embeddings. Its selector uses the model’s later classification confidence as a self-supervised target. [1]

**Our adaptation.** Preserve this inexpensive pair-feature idea, retain OWLv2’s existing object outputs, and add box geometry plus temporal context. An initial budget of 32 objects and 64–128 ordered pairs is a proposed setting to sweep, not an SG-ViT paper setting. Pair selection recall must be measured: downstream models cannot classify pairs that never reach them.

**Identity:** train a descriptor to separate different instances of the same class and match the same instance across time. Combine it with geometry and motion during association. Stable detector ranking is not a track identity.

**Semantics:** initially reuse OWLv2’s text-aligned classification head. An extra adapter is justified only if it improves fine-grained category or attribute matching. A dot product is a compatibility score, not proof of correct open-vocabulary recognition.

“Mug versus teacup” is a semantic distinction. “This teacup versus another identical teacup” is an identity distinction. Train and evaluate them separately.

# Temporal processing: attention or Mamba

For each selected ordered pair, assemble subject and object features, relative box geometry, track IDs used for indexing, elapsed time and an observation-validity mask. The temporal network weights are shared across pairs; its stored history or state is separate for each pair.

![Two panels compare causal attention over stored pair observations with a Mamba state updated from its prior state and the current observation.](data:image/png;base64...)

Figure 2. A/B read a bounded observation history; C/D recurrently update a compact state. Both output a feature for the same relationship head.

**A / B: causal attention.** Each current pair can consult recent observations. This is a strong, simple baseline. A fixed sliding window already bounds stored history; compare against this baseline rather than only against full-history attention.

**C / D: causal Mamba.** An input-dependent state update carries information forward. State size per active pair stays fixed as a video grows, although total state still grows with active pair count. Mamba’s sequence-efficiency results do not guarantee lower end-to-end latency in this pipeline. [3]

* Use timestamps and missing-observation masks; do not interpret a missed detection as a confirmed relationship ending.
* Reset or conservatively reinitialise state after uncertain identity reassignment. Expire inactive states using a validated policy.
* Keep a relation-change mechanism: correct identity does not prevent “holding” from persisting after release.

Causal processing uses only observations available at the current time. Bidirectional video backbones require later frames and need a separate offline or delayed-streaming evaluation. [10]

# Predictive training: JEPA variants B and D

JEPA learns by predicting feature vectors rather than reconstructing pixels. Here it is an auxiliary training task on the temporal representation, not a replacement for object detection, predicate supervision or language alignment. Object-level masked prediction is already explored by Causal-JEPA. [4]

![A student temporal encoder receives past observations and trains on relationship labels plus a future-feature prediction loss against a detached frozen target.](data:image/png;base64...)

Figure 3. Proposed causal predictive objective. Purple dashed arrows denote auxiliary training paths; future targets never enter the student’s past-only input.

**Training example.** Observe a person lifting a cup up to time t. Predict a later person–cup feature from this history. Compare it with a frozen extractor’s feature of the actual later observations. The future is a training target, not an inference-time input. Prefer targets that contain interaction information; predicting an almost constant object-category vector may teach little about actions.

Start with cached features and a frozen target extractor. Keep target vectors detached from gradient updates and preserve track correspondence. A different video teacher is a later experiment.

**Combined loss:** relationship supervision + pair-selection supervision + optional identity/semantic losses + weighted JEPA prediction loss. Enable detector box/class losses only when those components are being updated. Weak labels and teacher outputs remain imperfect; absent annotations are not automatically reliable negatives.

The auxiliary predictor and target branch can be removed at inference only because this design trains the deployed temporal encoder through the prediction loss. A separate predictor trained on entirely frozen deployed features would not improve those features by itself.

Record observed events separately from forecasts. Predicting that drinking is likely is not evidence that it happened.

# Combined variant D and the answering system

![Shared OWLv2 features feed semantic, identity and pair branches. Track association and pair features feed a Mamba relation module, which writes an event graph. An auxiliary JEPA loss attaches during training only.](data:image/png;base64...)

Figure 4. Combined experimental design: SG-ViT-style pair features, identity-indexed Mamba memory, semantic labels and an auxiliary JEPA objective.

The graph stores subject and object IDs, predicate, start/end times, confidence and evidence references. Merge repeated observations only while the relationship remains consistent; preserve transitions and uncertainty. The SSM’s hidden state supports current prediction, while the explicit event record supports historical retrieval.

![Graph events are retrieved for a question and passed to a small text reader. An optional saved-clip verification path can supply missing visual evidence.](data:image/png;base64...)

Figure 5. Answering uses retrieved events. Revisiting a clip is an optional, separately measured fallback.

Start with a fixed small text reader, such as Qwen3-1.7B in non-thinking mode. Give it compact event descriptions and ask for a short answer plus supporting event IDs; software resolves those IDs to timestamps. Short clips can initially supply all events, avoiding retrieval as a confounding factor. [8]

“Holding cup near mouth” alone does not establish “drinking.” If the graph lacks sufficient action evidence, the reader should abstain or request the relevant clip. That fallback adds inference cost. Exact lookups and counting can instead use graph queries when the question has been interpreted reliably.

# Training plan and implementation boundaries

Do not train a foundation model from scratch. Reuse the detector checkpoint, build a working graph predictor, and keep experiments small enough to isolate why a change helps.

## Stage 1 — Establish the detector-first baseline

Freeze OWLv2 and cache features, boxes and text embeddings. Train the pair head and router on the training split. Validate object, pair and relationship recall before adding memory. Freezing saves backward-pass cost, not deployment forward-pass cost.

## Stage 2 — Establish tracks and head controls

Compare conventional association using geometry, motion and appearance against the learned identity descriptor, with fixed detections. Use trusted instance correspondences or labelled pseudo-tracks; category labels cannot supervise instance identity. Reuse existing object semantics before adding an adapter.

## Stage 3 — Run the temporal comparison

Train A and C on the same observations, pair budgets and relationship supervision. Keep spatial pair construction and causal inputs identical. Measure latency and memory; a simple GRU is an optional recurrent control.

## Stage 4 — Add predictive training

Add the same JEPA task to create B and D. Keep teacher targets, data and schedules comparable; report extra videos and teacher compute. Keep deployed networks identical within A/B and C/D, removing auxiliary branches for timing.

## Stage 5 — Adapt or distil only if justified

If frozen features limit quality, unfreeze selected detector blocks at a lower learning rate. SGClip can provide imperfect relationship targets; its pair-region inputs require adaptation. Distil into the shared-token head for a speed experiment. A V-JEPA teacher is optional and need not run at inference. [5, 6, 7]

Keep the answering model fixed when comparing graph variants. Fine-tune it separately on predicted, imperfect training graphs if necessary. Evaluate held-out videos; never train any component on graphs or answers from the evaluation videos.

Loss weights, budgets and refresh thresholds remain experimental choices; no real-time speed is claimed.

# Experiments that can support a research claim

The primary comparison is useful output at an inference budget. A complete-system gain is meaningful, but individual claims need controls that isolate the component responsible.

| Claim | Required comparison | Useful measurements |
| --- | --- | --- |
| JEPA improves features | A vs B; C vs D | Relation/action accuracy; grounded QA; unchanged deployed path |
| Mamba improves efficiency | A vs C; B vs D | Temporal and total latency; GPU memory versus video length |
| Identity improves memory | Fixed detections and update policy; change association features | ID switches; cross-instance memory transfer; relation accuracy |
| Semantics adds value | Existing text head vs extra semantic adapter | Held-out category/attribute matching; query and SGG accuracy |

## Use three complementary evaluation layers

* **Graph quality:** Action Genome for the initial video SGG evaluation; report task setting, graph constraints, recall and mean recall. Keep matching rules and candidate budgets explicit.
* **Temporal correctness:** measure false persistence after release, event-change delay, occlusion recovery and same-class object swaps. Add a suitable tracked-instance subset if the standard benchmark cannot support these tests.
* **Application usefulness:** NExT-QA/NExT-GQA for answers and supporting time intervals; add streaming QA when the memory system is ready. Match the final output format across systems. [9]

Time detection, pair routing, relationship inference, association, graph maintenance, retrieval, reader and fallback. Report median and p95 response latency, ingestion throughput, peak GPU memory and stored memory per video hour. Test both one question and repeated questions per video. Cached VLMs are necessary efficiency baselines. [11]

For open-vocabulary claims, explicitly hold out relevant categories, predicates or compositions during task-specific training and report seen/unseen results separately. Pretraining exposure should be disclosed; a dot-product interface alone does not demonstrate generalisation.

Identity can prevent memory from being assigned to the wrong cup. A separate update/reset rule must retire an outdated relationship about the correct cup. Test these failure modes separately.

Interpretation: “better graph recall” is not automatically “better answers.” Include a flat timestamped event-list control using the same events and reader to test whether explicit graph structure adds value.

# Evidence, limits and reading map

The integrated system in this guide is a proposal. Existing papers support individual mechanisms in different settings; they do not validate their combination or establish a speed advantage for this project. “SG-ViT-style” describes reused ideas, not a reproduced checkpoint or training recipe.

**Relevant precedents:** DeWorldSG tests V-JEPA-based relationship refinement on 3DSSG/ReplicaSSG; Causal-JEPA tests object-level prediction on CLEVRER/PushT. STORM-Net already combines Mamba with dynamic scene graphs on Action Genome/RESCUE. Its reported timing is not evidence of universal superiority over attention. [4, 5, 12]

**Working vocabulary:** an embedding is a feature vector; a predicate is a named relationship; a track is an associated object over time; an SSM state is a compact learned memory; an ablation changes one component to test its contribution. “Causal” here means past-only input unless a stronger definition is explicitly justified.

## Primary sources

[1] [SG-ViT](https://arxiv.org/abs/2403.14270) — Architecture, relationship selection and joint training. Sections 3.1–3.3.

[2] [OWLv2-B/16 checkpoint](https://huggingface.co/google/owlv2-base-patch16-ensemble) — Pretrained detector starting point.

[3] [Mamba-2](https://arxiv.org/abs/2405.21060) — Selective state-space processing and efficient sequence computation.

[4] [Causal-JEPA](https://arxiv.org/html/2602.11389v2) — Object-level latent masking and predictive learning; CLEVRER and PushT.

[5] [DeWorldSG](https://arxiv.org/html/2607.00889v1) — Frozen V-JEPA 2 pair-clip features for 3D relationship refinement.

[6] [V-JEPA 2.1](https://arxiv.org/html/2603.14482v2) — Dense video features and distilled smaller models.

[7] [ESCA / SGClip](https://arxiv.org/html/2510.15963v1) — Weakly supervised, text-aligned relationship knowledge.

[8] [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) — Candidate text reader; non-thinking mode.

[9] [NExT-GQA](https://arxiv.org/html/2309.01327v2) — Answer accuracy and temporal evidence grounding.

[10] [VideoMamba](https://arxiv.org/html/2403.06977v2) — Video state-space modelling; bidirectional processing is not live causal inference.

[11] [ReKV](https://arxiv.org/abs/2503.00540) — Cached video question answering and streaming efficiency evaluation.

[12] [STORM-Net](https://doi.org/10.1109/ACCESS.2026.3707528) — Mamba and summary tokens for dynamic scene graphs.
