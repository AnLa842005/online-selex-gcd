# CUB reproduction audit — 2026-10-31

## Verdict: PASS WITH QUALIFICATION

Numerical reproduction is established for the paper's primary All-v2 metric. After 200 epochs on CUB-200-2011 with DINOv1 ViT-B/16 and seed 1, the final checkpoint scored **73.95%** versus **73.60%** in the paper (+0.35 percentage points). Exact split-level reproduction remains qualified. The Old/New allocation differs in a direction consistent with the upstream data-order sensitivity warning. Establishing causality requires an isolation experiment.

The authoritative baseline is **73.95 / 71.45 / 75.21** (All/Old/New, percent). The source is the user-supplied extraction from the completed RunPod standalone Final Reports evaluation, with `EVALUATION_COMPLETE` observed. The raw logs and checkpoints were not independently inspected here.

| Standalone SS-K-Means Train ACC Unlabelled_v2 | All (%) | Old (%) | New (%) | Difference from paper (All / Old / New, pp) |
|---|---:|---:|---:|---|
| Final, model.pt | 73.95 | 71.45 | 75.21 | +0.35 / -3.85 / +2.41 |
| Best, model_best.pt | 70.93 | 73.72 | 69.54 | -2.67 / -1.58 / -3.26 |
| Paper | 73.60 | 75.30 | 72.80 | — |

The final checkpoint outperforms the best checkpoint on standalone All-v2. Best-checkpoint selection maximizes disjoint Test Old-v1, so it need not maximize standalone Train Unlabelled-v2. The standalone scores use extracted features and semi-supervised K-Means with labelled training anchors. The earlier **61.5881 / 47.5650 / 68.6019** is an in-training TensorBoard Train ACC Unlabelled-v2 event. The disjoint Test ACC-v2 result of **56.0062 / 56.6574 / 55.3608** evaluates the held-out test population using unconstrained K-Means. Only the standalone v2 rows above are used for the paper comparison.

## Ranked investigations

| Status | Investigation | Evidence or next step |
|---|---|---|
| Resolved | Protocol mismatch | Compare the standalone Train Unlabelled-v2 report with the paper. |
| Resolved | Checkpoint evaluation | Both checkpoints have completed standalone reports. The upstream README evaluates both and does not identify which produced its table. |
| Non-blocking follow-up | Data-order sensitivity | Compare original and fixed shuffled order on cached features, preserving labels, masks and IDs and reversing the permutation for scoring. [Upstream order warning](https://github.com/SarahRastegar/SelEx#readme) |
| Reproducibility note | Run provenance and config | Archive actual arguments, environment, pretrained and checkpoint hashes, and imported K-Means source revision when available. Repository defaults alone cannot certify the server configuration. |
| Optional robustness work | Additional training seeds | Estimate run-to-run variation. Baseline acceptance does not require these runs. |

## Full diff classification

Earlier source-parity audit diff: `git diff c3d3b42a0aa7d7c9ea748a8305a6890a34100f73 91baedd` (42 files, 923 insertions, 42 deletions, including removed tracked bytecode).

Potential numerical effects, including conditional effects:
- config.py:8,20,26: environment-based data, split, pretrained and output paths. Portability; different resolved assets can change results.
- reproduction.py:6 and training.py:752: teacher/backbone normalization, module prefix stripping and head filtering; additionally accepts plain backbone weights. Portability, but checkpoint identity must be verified.
- extract_features.py:168: local ViT replaces live torch.hub implementation. Removes network dependency; strict state load checks shapes, not forward numerical equivalence. Compare fixed-image features if protocol-correct scores remain low.
- faster_mix_k_means_pytorch.py:5,176,248: public joblib import, input-derived device, explicit int64. Portability; Linux CUDA/int64 baseline expected equivalent, CPU/Windows now supported. No clustering objective change.
- training.py:238: resume restores state and copies best artifacts; changes trajectory only when enabled. :287 batch limiter changes number of updates only when nonzero (launcher leaves zero). :397 finite checks stop invalid runs, do not alter finite loss. :545,636 device routing replaces cuda default.
- training.py:463: additional state serialization; no intended RNG consumption. :847 final reports now use checked subprocesses, correct argument names, explicit best/final booleans and run-specific feature cache. extract_features.py:201 separates cache directories. These change evaluation orchestration and prevent stale/wrong-cache results; they are not loss changes.
- general_utils.py:140: portable log naming. All other additions are launch/setup/smoke/verification/tests/docs/requirements or ignored/removed bytecode. Server launcher controls effective config; smoke script is not the full-training entrypoint. requirements-reproduction.txt pins editable kmeans source plus numba.

Data source, split pickle, transforms, loss block, metric implementation unchanged. Six regression tests include normalized AST/source parity, but do not prove identical GPU results or equivalence of remote assets.

## CUB pipeline

- Official bundled SSB pickle: data/get_datasets.py:179–191: 100 known; novel concatenated Hard + Medium + Easy. No sorting. Target original 1..200 becomes 0..199 at cub.py:98; then known IDs map to 0..99, novel to 100..199 by list order (get_datasets.py:74–81).
- Metadata joins preserve image-table ordering; official train/test flag filters at cub.py:55–61. Subsampling uses boolean mask, so set iteration at :179 does NOT itself reorder resulting dataset.
- Labelled selection: known classes then global 50% instance choice, not per-class 50% (cub.py:168; data_utils.py:4–8). Hard-coded NumPy seed 0; get_datasets does not pass training seed to builder. Unlabelled is complement. Validation split is calculated but unused; its RNG consumption still occurs (:173–187).
- Counts from prior LOCAL_VALIDATION.md: 1498 labelled + 4496 unlabelled = 5994 train; 5794 official test. Not newly verified against RunPod. At batch 128: train 46 batches (5888 draws, 106 discarded), merge/prototype 47, unlabelled eval 36, disjoint test 46.
- train.py:808–829: WeightedRandomSampler draws 5994 with replacement; labelled weight 1, unlabelled 1498/5994. Expected labelled fraction ~57.14%, not exact half. All loaders shuffle=False; training drops last, others do not. Four workers, no custom generator/worker_init_fn, default nonpersistent workers.
- Train transforms (augmentations/__init__.py:7–32): bicubic resize shorter side 256, random crop 224, random horizontal flip, default ColorJitter (no effective jitter), tensor/ImageNet normalization; two views. Test: resize 256 + center crop 224. Prototype extraction uses first augmented TRAIN view (:545), not deterministic test transform.
- Training seeds NumPy and torch at :731–732, no explicit Python random seed or deterministic CUDA flags. Dataset selection resets NumPy to 0. Worker seeding delegated to installed PyTorch; no independent generator state. Final k_means.py:139 explicitly seed_torch(0), not seed 1. These are upstream behaviors, not fork regressions.

## Configuration comparison

Expected here means pinned upstream CUB/DINOv1 Python defaults plus README CUB smoothing instruction, not independently recovered paper run args. Root contrastive_train.sh targets CIFAR10 and is not a CUB reference.

| Parameter | Launcher / expected |
|---|---|
| Epochs / batch / seed | 200 / 128 / 1 |
| Optimizer | SGD lr 0.1, momentum 0.9, wd 1e-4 |
| Schedule / warmup | CosineAnnealingLR T_max=200 eta_min=0.0001 / no warmup schedule |
| Fine tuning | blocks >=10; backbone otherwise frozen |
| USE / SSE | 0.65 * USE + 0.35 * accumulated SSE / 2 (:385); smoothing 1.0 |
| Views / temperature | 2 / 1.0 |
| Hierarchy | zero_one, cluster_momentum=1, euclidean, balanced |
| HSSK | K=200, tol=1e-4, max_iter=10, n_init=1, pairwise batch 1024, cap ceil(5994/200)=30 |
| Higher levels | old/new counts 50,25,12,6,3,1 each; sklearn KMeans(random_state=0) on labelled prototypes; HSSK n_init=1 (:602–617) |
| Frequency | prototypes each epoch; disjoint Test each epoch; standalone Train kmeans every 200 epochs |
| Final SS-KMeans | max_iter=10, n_init=10; seed 0 (k_means.py:122–139) |
| Selection | max disjoint Test Old **v1**, first eval_func; strict greater-than, not loss or v2 |

Script prints static launcher configuration and differences; optional --actual-json compares an exported actual Namespace JSON. It cannot certify an unavailable RunPod run. Paths and shell environment must be checked separately.

## Checkpoints and evaluation

model.pt is overwritten after each epoch, so epoch 199 if completion confirmed (:444). model_best.pt maximizes old_acc_test (:449), which is returned for FIRST eval function, v1 by default. Misleading print says New, actual condition is Old. Projection head saved separately; label_head_best is a duplicate of projection head (:457–458), no independent label classifier. No final label head saved.

Extraction loads backbone strictly (:210–211), projection head with strict=False (:224–230), so complete projection loading is not asserted. extract_features_dino computes backbone features; head is not used in clustering. Missing head files nevertheless cause loading failure. This is not evidence that missing label-head parameters reduce ACC.

Resume restores model/head, optimizer, scheduler, epoch+1, Python/NumPy/torch CPU/all CUDA RNG and history. Save follows scheduler step. No worker-internal state saved; default workers restart. Prototype state is recomputed; interval=1 and cluster_momentum=1 are expected. Non-default interval rejects incompatible boundary. CPU unit test verifies partial state roundtrip, not full GPU/worker equivalence. Copying into same directory may raise SameFileError; server launcher creates a new run directory.

v1 matches Hungarian separately in Old and New then sample-weights All (cluster_and_log_utils.py:6). v2 matches once globally then computes Old/New from that assignment (:30). All is sample-weighted, not arithmetic mean of class accuracies. Values are fractions internally, percentages in supplied comparison.

The disjoint test loader includes ALL 200 classes (despite variable name test_loader_labelled). Paper-comparable final SS-KMeans uses labelled anchors plus unlabelled TRAIN examples and scores only unlabelled examples (k_means.py:69–104); held-out test features are extracted but not used by final fit. Thus “correct unlabelled test set” must distinguish transductive training pool from official held-out test.

Test ACC_v1_Old stopping at 121: no source branch supports selective stop; v1 and v2 logged sequentially every test via add_scalars (:105–118), with child event writers. Missing files, wrong run directory, resume across runs, event-reader purge/downsampling, or incomplete copied logs are hypotheses. Read every recursive event file using EventAccumulator(size_guidance={'scalars':0}, purge_orphaned_data=False), group by run/path/tag, inspect max step and wall time, duplicates and corruption, compare train.log. Reservoir sampling alone is not proof of missing training. Without raw events root cause is unresolved; last_step=199 of v2 does not establish all streams are intact.

## Environment evidence

requirements.txt already pins torch 2.0.0, sklearn 1.3.0, timm 0.9.10, torchvision 0.15.1. Therefore these versions are not evidence of fork drift. environment.yml conflicts (Python3.7/torch1.8.1/CUDA10.2/sklearn0.21.3/timm0.4.12); no proof which produced the paper table. Python3.10, cu117 and A40 have no measured significant effect here; do not rank them as causes. kmeans-pytorch is explicitly required by README and imported for periodic evaluation; custom HSSK is separate. Pinned editable revision f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805 must be checked against actual imported server source, not only pip version 0.3.0. Backend substitution has plausible impact but is not established on this run.

## Repeat the standalone evaluation

Follow the [one-command checkpoint workflow](BASELINE_SETUP.md#standalone-checkpoint-evaluation). The default wrapper preflight checks checkpoints, directories and metadata. Set `SELEX_VERIFY_ALL_IMAGES=1` to check every listed image before extraction. Use the final row of `summary.txt` for the baseline. The [tracked record](results/cub_dinov1_seed1.json) includes both v1 and v2 for each checkpoint.
