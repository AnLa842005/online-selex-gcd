# Official offline SelEx CUB baseline

Upstream: https://github.com/SarahRastegar/SelEx

Pinned upstream commit: `c3d3b42a0aa7d7c9ea748a8305a6890a34100f73`.
The original MIT LICENSE, paper citation, and authors' README are retained.
This branch implements offline SelEx only. No Hyperbolic/HypCD or Online GCD.

## Architecture and exact upstream settings

The real training entry point is
`python -m methods.contrastive_training.contrastive_training`.
The README's bare `contrastive_training.py` command does not refer to a root file.
`dino_v1=True` selects DINOv1 ViT-B/16, 768-dimensional features, 224px images.
The projection head has 3 layers and 65,536 outputs. Blocks 10 and 11 are trainable.
The DINOv2 path exists upstream but is inactive and is not used here.

HSSK: `extract_labeled_protos` and
`methods/clustering/faster_mix_k_means_pytorch.py`; hierarchical counts are
200, 100, 50, 24, 12, 6, 2. USE: `info_nce_logits` plus `LabelSmoothingLoss`.
SSE: labelled `SupConLoss` plus the hierarchy-weighted contrastive terms.
The unchanged code combines them as `0.65 * USE + 0.35 * SSE / 2`.

The server script explicitly uses the CUB parser defaults: batch 128, 200 epochs,
seed 1, SGD lr 0.1, momentum 0.9, weight decay 1e-4, cosine schedule with
minimum lr 1e-4, temperature 1, two views, supervised weight 0.35,
unsupervised smoothing 1.0, Euclidean distance, `zero_one` strategy,
cluster momentum 1, balanced clustering, SSB enabled, labelled proportion 0.5,
prototype extraction every epoch, training KMeans report interval 200.
The supplied shell examples target CIFAR10 and have differing weight decay and
block settings; they are not CUB configs. CUB parser defaults plus the README's
CUB smoothing instruction are the source of this configuration.

## Known upstream limitation

The [upstream README](https://github.com/SarahRastegar/SelEx#note) states that the
current code is sensitive to dataset order bias (issue #11), with fixed code and
updated results pending. This branch preserves that algorithm and ordering.
A successful smoke test establishes executable behavior, not paper reproduction.

## Environment

Local uses a separate repository `.venv`, Python 3.11.9, with the exact official
`requirements.txt` versions. Python 3.11 was already installed and has compatible
wheels for these pins; Conda/Mamba were unavailable. The parent Python 3.12 venv
was not modified. The legacy upstream `environment.yml` is a Linux Python 3.7
environment export with conflicting old packages; use the README's pip workflow.

`requirements-reproduction.txt` installs numba (needed by the kmeans source's
SoftDTW import) and editable kmeans from the exact repository named upstream,
commit `f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805`.
Install it AFTER the official requirements, so the PyPI build is replaced.
Local CUDA wheels keep torch 2.0.0 and torchvision 0.15.1, with CUDA 11.8.
`requirements-lock-windows.txt` records the validated local transitive package
versions without private paths; it is a Windows snapshot, not a Linux lock.
The driver-reported CUDA maximum is not the torch runtime version.

## Data, weights, and paths

Set shell environment variables as shown in `.env.example`; `.env` is not
automatically loaded. Paths default to siblings of this repository and are
independent of the current working directory.

```text
SELEX_DATA_ROOT/
  CUB_200_2011/
    classes.txt
    images.txt
    image_class_labels.txt
    train_test_split.txt
    attributes/image_attribute_labels.txt
    images/001.Black_footed_Albatross/...
SELEX_PRETRAINED_PATH = /path/to/dino_vitbase16_pretrain.pth
SELEX_OUTPUT_ROOT = /path/to/runs/selex
```

`SELEX_DATA_ROOT` is the PARENT of `CUB_200_2011`. The official loader also reads
the attribute annotations, even though its `__getitem__` returns image/label/ID.
The official CUB archive MD5 is `97eceeb196236b17998738112f37df78`.
Data source: https://data.caltech.edu/records/65de6-vp158.
Weights source: https://github.com/facebookresearch/dino#pretrained-models,
ViT-B/16 backbone-only, SHA256
`bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e`. The loader also accepts the full teacher checkpoint,
strips only backbone/module prefixes and excludes the projection head, then
strictly loads backbone parameters. No random pretrained substitute is used.

The tracked official `data/ssb_splits/cub_osr_splits.pkl` supplies 100 old and 100
novel classes. The loader fixes sample subsampling to seed 0, independently of
training seed 1. It uses CUB's official 5,994 training and 5,794 test images;
50% of examples in old classes are labelled: **1,498 labelled / 4,496 unlabelled / 5,794 test**.
The SSB file SHA256 is `48bdcaa05603d932037738d3df776ab500945e9b7aeff85f4668305bf71bd1e7`. `verify_cub.py` prints exact counts,
split checksum, and verifies all 11,788 images. No replacement class split exists.

## Windows smoke

```powershell
# Run in the repository. The wrapper uses .venv without activating it.
.\.venv\Scripts\python.exe scripts/verify_environment.py
.\.venv\Scripts\python.exe scripts/verify_cub.py
powershell -ExecutionPolicy Bypass -File scripts/smoke_cub_local.ps1 -Device cpu -BatchSize 2
# This 4 GB laptop required CPU fallback. A larger GPU can use -Device cuda:0.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

See [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md) for the actual validated environment,
loss values, runtime, and GPU memory failures.

The smoke runner retains all 200 SSB classes, taking 100 labelled and 300
unlabelled training images, and 200 held-out images for evaluation. Original IDs
are recorded, then densely indexed only within this debug subset. It invokes the
actual upstream HSSK hierarchy and `train` function, with the full backbone and
65,536-output head. Tiny mixed batches include labelled old-class and unlabelled novel-class examples
to exercise SSE with a useful mix even at batch size 2. Only one
optimizer batch and one epoch execute. `--max-train-batches` on the normal
training entry point defaults to 0 (unlimited). Smoke prints USE/SSE/total,
rejects nonfinite total, performs evaluation, and saves and reloads model/head.
Every run has a UUID output directory. The report records runtime and peak CUDA
memory. Smoke metrics are never baseline research results.

## Linux GPU server handoff

Run on the server itself, in a shell with Conda initialized. The repository is public and cloning does not require authentication. Replace only the three storage
paths with actual server locations.

```bash
git clone --branch baseline/reproduce-selex https://github.com/AnLa842005/online-selex-gcd.git
cd online-selex-gcd
git remote add upstream https://github.com/SarahRastegar/SelEx.git
conda create -n selex python=3.11 pip -y
conda activate selex
python -m pip install -r requirements.txt
python -m pip install -r requirements-reproduction.txt
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.0.0+cu118 torchvision==0.15.1+cu118
export SELEX_DATA_ROOT="$HOME/datasets"
export SELEX_PRETRAINED_PATH="$HOME/pretrained/dino_vitbase16_pretrain.pth"
export SELEX_OUTPUT_ROOT="$HOME/runs/selex"
export SELEX_GPU=0
python scripts/download_assets.py --data-root "$SELEX_DATA_ROOT" --pretrained-root "$(dirname "$SELEX_PRETRAINED_PATH")"
python scripts/verify_environment.py
python scripts/verify_cub.py
CUDA_VISIBLE_DEVICES="$SELEX_GPU" python -u scripts/smoke_cub.py --device cuda:0 --batch-size 4
python -m unittest discover -s tests -v
bash scripts/train_cub_server.sh
```

This full-training script has not been run on the laptop. It expects an NVIDIA
GPU and sufficient memory for batch 128 with the original architecture. An A6000
48 GB is the intended server target; actual peak usage and full runtime remain
unmeasured. Plan at least 32 GB host RAM and 20 GB free storage; repeated runs,
resume states, and extracted features need more. No full-training completion or
accuracy is claimed before the server run.

The output directory is created with `mktemp` and never reuses a previous run.
It contains `train.log`, TensorBoard events, plots, `model.pt`, best model,
projection-head weights, and `training_state.pt`. Final reports evaluate both
best and last checkpoints using extracted features and upstream SS-KMeans.
Feature directories are distinct for each checkpoint/run.

## Resume

```bash
export SELEX_RESUME=/absolute/path/to/previous/checkpoints/training_state.pt
bash scripts/train_cub_server.sh
unset SELEX_RESUME
```

Resume writes into a NEW run, restores model/head, optimizer, scheduler, epoch,
best-selection statistics, histories, and Python/NumPy/torch CUDA RNG states.
Keep the original best model/head beside the state file; they are copied into the
new run. The supported full config extracts prototypes every epoch. Use the
same data, environment, seed and hyperparameters. Resume is supported only at
epoch boundaries, not mid-epoch; GPU kernels and worker augmentation ordering
mean bit-for-bit equivalence across machines is not guaranteed. An upstream
weights-only checkpoint can warm-start, but cannot restore optimizer progress.

## Evaluation and paper target

Both v1/v2 implementations are unchanged. v1 matches old/new subsets separately;
v2 computes one Hungarian assignment over all samples then reports All/Old/New.
The first function (`v1`) controls returned values and best-checkpoint selection;
the best checkpoint is selected by highest OLD accuracy on the disjoint test set,
not novel accuracy despite the upstream message wording. Final research reporting
uses `SS-K-Means Train ACC Unlabelled_v2` in the best/last final report sections,
as directed by the authors' README. All/Old/New are clustering accuracies on all
unlabelled examples, old-class examples and novel-class examples respectively.

[ECCV 2024 paper, Table 2](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08926.pdf)
reports CUB DINOv1 **73.6 / 75.3 / 72.8** (All/Known/Novel, percent). These are
reference targets under the paper/README's unlabelled v2 protocol, not results
obtained here. Do not compare this DINOv1 run with the DINOv2 row. The upstream
order-bias warning applies to attempts to reproduce these numbers.

### Standalone checkpoint evaluation

With the existing assets and evaluation environment on Linux, export the four required paths and run the wrapper from the repository root.

```bash
export MODEL=/absolute/run/log/RUN_ID/checkpoints/model.pt
export SELEX_DATA_ROOT=/absolute/datasets
export SELEX_PRETRAINED_PATH=/absolute/pretrained/dino_vitbase16_pretrain.pth
export SELEX_OUTPUT_ROOT=/absolute/new-evaluation-output
bash scripts/evaluate_cub_checkpoints.sh
```

Replace the example paths (including `RUN_ID`) before running. **SELEX_OUTPUT_ROOT must be a new absolute path** to prevent stale feature reuse. Set `SELEX_PYTHON` to select a different interpreter if needed. The fast preflight checks both backbones (`model.pt`, `model_best.pt`), their companion heads (`model_proj_head.pt`, `model_proj_head_best.pt`), DINO weights, the CUB_200_2011 root and images directories, and four metadata files (`images.txt`, `image_class_labels.txt`, `train_test_split.txt`, `attributes/image_attribute_labels.txt`). Only `SELEX_VERIFY_ALL_IMAGES=1` adds a preflight stat of each image in `images.txt`. The loader retains its own integrity checks during evaluation. The wrapper disables dataset download and archive repair.

Under `SELEX_OUTPUT_ROOT`, `extract-{final,best}.log` and `evaluation-{final,best}.log` capture each stage. Compare the final row in `summary.txt` (standalone SS-K-Means Train ACC Unlabelled_v2, percent) with the paper. The seed-1 final checkpoint reproduced **73.95 / 71.45 / 75.21** for All/Old/New, versus **73.60 / 75.30 / 72.80** in the paper. The best checkpoint scored **70.93 / 73.72 / 69.54**. Files `results-{final,best}.txt` preserve the first evaluation function (v1) for provenance.

## Differences from upstream

No algorithmic changes. HSSK assignments, hierarchy, USE, SSE, total-loss formula,
SSB split, sampling in full training, transforms, and metric definitions remain
upstream. Changes are:

1. Environment-based CUB/pretrained/output paths and repository-relative SSB path.
2. Backbone-only/full-teacher checkpoint format support with strict backbone load.
3. `joblib` public import replacing removed sklearn private import; tensor device
   inference replacing hard-coded CUDA in HSSK for CPU smoke support; explicit
   int64 cluster indices to match Linux NumPy behavior on Windows.
4. Windows-safe experiment directory separator and portable module invocation.
5. Checked final-evaluation subprocesses using the active interpreter, explicit
   dataset/experiment/checkpoint identifiers, matching extracted-feature paths,
   and the repository's identical ViT architecture instead of a torch.hub fetch.
6. Debug batch limit (default off), device option (default CUDA), nonfinite-loss
   guard, and debug-only component logging.
7. Additional epoch-boundary resume state; original model/head files remain.
8. Real-data verification/smoke scripts, server launcher, download helper, tests,
   environment snapshot, ignore rules, and this documentation.
9. Stop tracking upstream bytecode artifacts in the new commit. Upstream history
   is retained without rewriting; original attribution and LICENSE are untouched.

Data, weights, checkpoints, logs, venv, editable dependency checkout and `.env`
are excluded from new commits. The `data/` CODE package and official SSB split
files must remain tracked; ignoring all of `data/` would break the baseline.
