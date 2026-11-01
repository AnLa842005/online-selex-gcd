# Phase 3A: clean frozen-DINO CUB infrastructure

Protocol: **online-selex-cub-frozen-dino-v0**. Real CUB feature extraction and
online evaluation: **NOT RUN**. Phase 3A validates infrastructure using fake
metadata, fake feature caches and a tiny mocked encoder only. No real DINO
inference, training, feature extraction job, download or GPU workload was run.

The reproduced full SelEx checkpoint saw unlabelled novel CUB training images.
It is not clean Online GCD initialization. Its 73.95 / 71.45 / 75.21 All/Old/New-v2
result stays an offline upper-bound reference or future explicitly named
warm-start ablation. Do not compare these raw scores with synthetic Phase 2 or
this clean online task as if their training exposure and evaluation populations
were identical. "Clean" here excludes CUB fine-tuning/full-SelEx exposure; it is
not a claim that generic ImageNet pretraining had no semantically related birds.

## Fixed population definitions

Use the existing `data/ssb_splits/cub_osr_splits.pkl` through the Phase 1 reader:
100 known and 100 novel classes, original zero-based CUB IDs, with novel order
`Hard + Medium + Easy`. No new class partition or label remapping is created.
The pinned SSB SHA256 is
`48bdcaa05603d932037738d3df776ab500945e9b7aeff85f4668305bf71bd1e7`.

| Population | Exact definition | Label access |
|---|---|---|
| Labelled base | Official train rows in known classes, then the existing global 50% labelled subset | Known labels available for calibration and prototype initialization |
| Online stream | Every remaining official train row: unlabelled known plus all novel train images, each once | True labels only in evaluator manifest |
| Test probes | Official test rows; known always eligible, novel only from its assigned introduction session | Evaluator only; no threshold, prototype or model updates |

Metadata is read from `images.txt`, `image_class_labels.txt` and
`train_test_split.txt` in the supplied **CUB_200_2011 directory itself**, not its
parent. Preserve `images.txt` row order, join on official image ID, subtract one
from official class labels and filter train rows exactly as the existing loader.
On the resulting known-train list, use
`RandomState(0).choice(range(n), replace=False, size=int(0.5*n))`, then retain the
chosen rows in original order. This is identical to `subsample_instances` plus
the existing boolean-mask subsampling. It is **not per-class 50%**, and does not
depend on the online seed. Tests execute the upstream selection function against
the new local-RNG implementation. No original sampling or dataset code is edited.
The upstream unused train/validation split does not change this labelled pool.

For orientation, the baseline documentation reports approximately 5,994 train,
5,794 test, 1,498 labelled base and 4,496 stream observations. These numbers are
not acceptance constants: derive exact counts from the supplied metadata and
print them when building manifests. Fail on mismatched ID sets, duplicate IDs or
paths, invalid flags/classes, incomplete class coverage or inadequate known
stream coverage. Base/stream/probe must be mutually disjoint and cover every
metadata row exactly once. No test row enters base or stream.

The default seed is 1. Reuse Phase 1's five-session schedule, 20 new classes per
session, per-class round-robin allocation over eligible sessions and local seeded
shuffling. Known observations occur throughout. Seen novel classes recur via
different images; never duplicate original samples or expose a class early.

## Manifest and learner boundary

The manifest command publishes five checksummed JSON files together:

| File | Audience and content |
|---|---|
| `partition.json` | Privileged builder/extractor/evaluator: source metadata including class-bearing paths, original class sets, schedule, complete feature row ordering and provenance |
| `base.json` | Authorized base projection: sample ID, feature row, known class ID |
| `stream_learner.json` | Orchestrator source for current-session projections: only sample ID, session ID, feature row in each record |
| `stream_evaluator.json` | Stream records plus hidden class, known status and introduction session |
| `probe_evaluator.json` | Test records with hidden class, official split and eligibility session |

Every view records protocol, seed, SSB hash, labelled-subset hash, record ordering,
partition reference and its own checksum. SHA256 covers canonical JSON excluding
the checksum field, with sorted keys and compact separators. Partition identity
also covers all source records, raw metadata file hashes, configuration and
class schedule. Loading recomputes the deterministic partition and all views,
rejecting inconsistent row indices, labels, schedule or projections. Production
CLIs also require the exact existing SSB lists, not just matching class counts.

Sample IDs hash the dataset name and official image ID; feature rows are ordered
lexicographically by these opaque IDs rather than by class. No class-bearing
path or original class-major image index enters the learner records. These IDs
are stable identifiers, not a secrecy mechanism against access to source metadata.

Do not hand the full learner manifest, full cache or any evaluator manifest to
the bank: they include future observations. The trusted runner calls
`learner_session(bundle, t)` and passes only each current sample ID, session ID
and feature vector to `PrototypeBank.observe`. Only the initial known-labelled
groups are separately authorized. Novel labels, schedules, total novel count,
evaluation mappings and class-bearing paths never enter the learner API/state.
The boundary is a tested API convention within a trusted process, not an OS sandbox.

## Frozen encoder and feature contract

- DINOv1 ViT-B/16, `models.vision_transformer.vit_base(patch_size=16)`.
- Only official ImageNet-pretrained **backbone-only** weights with SHA256
  `bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e`,
  already recorded in `BASELINE_SETUP.md` and `scripts/download_assets.py`.
- The checksum is checked before deserialization/model construction. A renamed
  SelEx `model.pt`, projection head, teacher wrapper or CUB fine-tuned checkpoint
  is rejected even if its tensors have compatible shapes. No hash override is
  exposed by config or CLI; another checkpoint requires a new audited protocol.
- Load tensors with `torch.load(..., map_location='cpu', weights_only=True)` and
  strict state-dict matching. No torch.hub, network or asset repair is used.
- Frozen parameters, eval mode and `torch.inference_mode()`; no optimizer,
  augmentation, mixed precision, projection head or gradient computation.
- Convert images to RGB. Reuse the existing ImageNet **evaluation** transform:
  PIL bicubic resize of the shorter side to 256 (`224 / 0.875`), center crop
  224x224, ToTensor scaling to [0,1], mean [0.485,0.456,0.406] and standard
  deviation [0.229,0.224,0.225]. No random crop or flip. Pixel-parity is tested
  against the existing transform with a generated image.
- `model(images)` returns the final LayerNorm output at token 0 (CLS), dimension
  **768**; no intermediate layer, patch pooling or head output is substituted.
- Normalize each CLS vector to unit L2 norm and store float32. Zero or nonfinite
  vectors are errors, never repaired.

Extraction uses sequential non-shuffled batches, no dropped tail, fixed torch
seed, deterministic algorithms, cuDNN benchmark off and TF32 off. Set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization. Unsupported
deterministic operations fail rather than silently changing settings. Record
device and library versions; bitwise equality across GPU models/libraries is
not promised. The metadata/image dataset is separate from the original CUB
loader, so it has no automatic download hooks or attribute-file requirement.

## Cache format, integrity and provenance

A completed cache directory contains:

- `features.npy`: normalized float32 matrix `[all_metadata_rows, 768]`, written
  sequentially into a staging memmap without retaining every batch in RAM.
- `sample_ids.json`: label-free row-to-ID mapping.
- `cache.json`: checksummed completion/provenance record, including feature-file
  SHA256, sample-order hash, partition/metadata/SSB/labelled-subset identities,
  checkpoint hash and role, architecture, token, dimension, preprocessing and
  its hash, normalization, checkpoint path, extraction config, timestamp UTC,
  code commit, device and Python/Torch/torchvision/Pillow/NumPy versions.

There are no true labels or class-named feature directories in the cache. The
extractor's provenance/path record stays with the trusted orchestrator, not the
learner. Cache reads use NumPy with `allow_pickle=False`, validate every contract
hash, exact ID order, dtype, shape, finite values and unit norms. The CPU runner
loads a read-only array (about 35 MiB for full CUB), avoiding Windows mmap locks
through evaluator references and error tracebacks.

Publication happens only after all batches and checks complete, by renaming a
staged directory. Existing output directories are never overwritten. **No partial
resume** is implemented. A failed extraction publishes no complete cache; restart
into a fresh output directory after fixing the error. `--reuse` validates the
pinned checkpoint, current metadata and completed cache before reuse, without
opening images or constructing a model. It does not claim to revalidate every
image's pixels: metadata hashes do not detect changed image bytes at unchanged
paths. Archive the official dataset's integrity record separately.

Hashes guard accidental stale/mismatched assets, not a malicious party who can
rewrite both artifacts and hashes. Internal test caches are explicitly marked
`fixture` and rejected by production CLIs; no CLI fixture bypass exists.

## Known-only threshold calibration

For each known class, normalize labelled base features. Construct normalized
class means. For each base sample x in class c, compute:

1. Positive score: cosine to the normalized sum of the **other** base samples in
   c (leave one out). Do not score against a prototype containing x.
2. Negative score: maximum cosine to a different known-class mean. Those means
   contain no x and no stream/probe samples.

Require at least two known classes and at least two base samples per class;
reject inadequate classes or cancelling zero means/LOO vectors explicitly.
Maximize `0.5 * (positive acceptance rate + negative rejection rate)`, with
acceptance `score >= threshold` and rejection `score < threshold`. Balance is
between positive and negative examples, not between class frequencies.
Candidates are -1, 1, all observed scores, and their next representable float64
values toward +infinity within [-1,1], covering every distinct decision boundary.
Break objective ties by choosing the **largest threshold**.

The calibration JSON records method/version, candidates, accepted/rejected
counts, objective, selected threshold, sample/class counts, normalized base
feature hash and base-manifest/cache identities. The callable accepts only
labelled base groups. The runner calibrates before constructing the evaluator;
no novel/test labels influence threshold selection. Loading a saved calibration
recomputes and compares the known-only objective and identity record, rejecting
stale or edited thresholds. No speed benefit from bypassing validation is claimed.

EMA momentum is fixed at **0.9**. Momentum selection is a future ablation, not a
test/probe-tuned parameter. Prototype growth is allowed and logged; replay budget
is zero and no merge/split/delete or prototype cap is implemented.

## CPU evaluation and resource records

Reuse Phase 2's prototype bank and evaluator unchanged. Stream scores use recorded
arrival-time assignments before EMA updates; do not re-predict previous stream
observations. Report local and cumulative Hungarian All/Old/New, novel NMI/ARI,
fragmentation/duplicate incidences, active/known/novel-origin bank counts,
cumulative fixed mapping, eligible held-out probe accuracy, signed/clipped
forgetting and identity churn. The evaluator verifies exact learner state
equality before/after probe prediction. Future novel probes never enter inference
or metrics. All empty metric subsets retain Phase 2's explicit null/count rules.

Result JSON distinguishes `experiment: frozen_DINO_clean_online` from
`fixture_only`; Phase 2 uses its separate synthetic result schema. Record config,
manifest/cache identities, extraction provenance, calibration, code commit,
versions, base probe report, five session reports and final summary. Learner
state remains Phase 2's label-safe JSON and may be saved separately.

Each session logs processing/evaluation seconds, prototype vector payload bytes,
serialized learner state size and peak tracemalloc bytes. The latter includes
evaluator allocations and excludes untraced native/RSS memory; it is not total
machine memory or a strict bank budget. Also record feature payload size and total
runtime. The CPU runner limits existing BLAS pools to one thread to avoid small
matrix thread-launch overhead; it restores prior thread settings afterward.
For determinism comparisons exclude top-level/session `performance` fields and
compare metrics, calibration and learner state in the same software environment.

## Exact lab runbook (Linux shell)

Use the existing reproduction Python environment; NumPy, SciPy, scikit-learn,
PyYAML, Torch, torchvision and Pillow are already dependencies. No installation,
asset download or repair is performed by these commands. Supply existing assets:

```bash
cd /path/to/online-selex-gcd
git switch research/online-selex
git pull --ff-only origin research/online-selex

export CUB_ROOT=/path/to/CUB_200_2011
export DINO_PRETRAINED_PATH=/path/to/dino_vitbase16_pretrain.pth
export ONLINE_SELEX_OUTPUT_ROOT=/path/to/new/online-selex-clean-v0
export ONLINE_SELEX_GPU=cuda:0
mkdir -p "$ONLINE_SELEX_OUTPUT_ROOT"

# 1. Metadata-only partition preparation; does not read image contents.
python scripts/build_cub_online_manifests.py \
  --config configs/online_selex/cub_frozen_dino_v0.yaml \
  --cub-root "$CUB_ROOT" \
  --output "$ONLINE_SELEX_OUTPUT_ROOT/manifests"

# 2. Lab GPU extraction. Run only when explicitly ready to use these real assets.
python scripts/extract_clean_dino_cub_features.py \
  --config configs/online_selex/cub_frozen_dino_v0.yaml \
  --cub-root "$CUB_ROOT" --checkpoint "$DINO_PRETRAINED_PATH" \
  --manifests "$ONLINE_SELEX_OUTPUT_ROOT/manifests" \
  --output "$ONLINE_SELEX_OUTPUT_ROOT/features" \
  --device "$ONLINE_SELEX_GPU" --batch-size 64 --workers 4

# 3. CPU baseline: known-only calibration, chronological stream, held-out probes.
CUDA_VISIBLE_DEVICES='' python scripts/run_cub_frozen_dino_baseline.py \
  --config configs/online_selex/cub_frozen_dino_v0.yaml \
  --manifests "$ONLINE_SELEX_OUTPUT_ROOT/manifests" \
  --cache "$ONLINE_SELEX_OUTPUT_ROOT/features" \
  --output-json "$ONLINE_SELEX_OUTPUT_ROOT/result.json" \
  --save-calibration "$ONLINE_SELEX_OUTPUT_ROOT/calibration.json" \
  --save-state "$ONLINE_SELEX_OUTPUT_ROOT/learner_state.json"
```

For extraction reuse, repeat step 2 with `--reuse`. For a rerun with a matching
calibration, add `--calibration-json "$ONLINE_SELEX_OUTPUT_ROOT/calibration.json"`
to step 3 and use new output filenames (omit `--save-calibration` or use a new
path). This is a fresh chronological run, not a resumed evaluator.

Use output paths outside Git or under existing ignored `outputs/`. Never commit
manifests with paths/labels, feature matrices, caches, calibration results,
learner state, run results, checkpoints or temporary staging directories.

## Phase 3A validation and Phase 3B work

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p 'test_online_cub_*.py' -v
python -m unittest discover -s tests -p 'test_online_clean_cache.py' -v
python scripts/build_cub_online_manifests.py --help
python scripts/extract_clean_dino_cub_features.py --help
python scripts/run_cub_frozen_dino_baseline.py --help
python scripts/compare_reproduction_config.py
git diff --check
```

Fixtures exercise metadata partitioning, exact upstream labelled selection,
SSB reuse, manifest tampering, cache integrity, preprocessing parity, frozen
mock inference, known-only calibration, probe isolation and repeatable fake-cache
end-to-end execution. They require no real dataset/checkpoint/network/CUDA.
Full test discovery also runs existing CPU reproduction regression tests.

Phase 3B must run the lab commands on authorized real assets, inspect derived
counts and extraction provenance, archive commit/checkpoint/dataset identities,
then report actual metrics and resource usage with **clean online exposure**.
Check threshold calibration diagnostics, prototype proliferation, fragmentation
and forgetting before designing further ablations. Broader seeds, threshold
robustness, momentum ablations and explicit bank/replay budgets remain future
experiments. Real CUB performance remains **NOT RUN**, not inferred from fixtures.
