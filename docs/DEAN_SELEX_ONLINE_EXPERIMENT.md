# DEAN online experiment infrastructure

Branch: `experiment/dean-selex-online`. Created from clean `research/dean-selex`
at `fdb4611`, which contains the DEAN integration and is 12 commits ahead of
`baseline/reproduce-selex`. No baseline branch, datasets, checkpoints, or prior
run artifacts were modified. No training, commit, or push was performed.

## Audit at pinned upstream

Official source remains external at `../OCGCD-dean-audit`; verified SHA:
`28b9384cd7ea9b123b03d045bac2e92d8d0b2870`. Existing verification rejects a
different SHA, origin, file tree, blob hashes, or dirty checkout.

`Ours_train_online.py` boundaries at this exact revision:

- Base epoch loop: line 258, best model/head save: line 342.
- Base-best reload: lines 345–350; initial base evaluation: lines 358–377.
- Online session starts: line 389; shuffled stream DataLoader: line 395.
- Online batch updates: line 738; original loss: line 767,
  `ce_loss + args.energy_hp * energy_loss`.

Upstream has no external base-resume/skip flag. Its "Resuming" message reloads
the base-best file generated during that same invocation. The checkpoint has
only `model_state_dict` and `header_state_dict`; no optimizer or RNG state.
Existing wrapper checks the full DINO teacher checkpoint (a distinct asset),
binds one dataset path literal through AST, and isolates runs per seed. Before
this change, only `execution.json` was finalized; aggregate `provenance.json`
kept `NOT RUN`. Existing CPU tests cover frozen contracts, teacher loading,
checkpoint compatibility, and one selected original online update.

User-supplied completed seed-1 reference, not rerun or independently measured
here: All=0.547808, Old/Base=0.631773, New=0.219492, pre-online base=0.8223.
The locked official config and `OFFICIAL_ARGS` are unchanged: lr=1e-4,
base_epochs=30, inc_epochs=15, batch=64, base classes=160, one online session,
seed=1, LoRA layers 7–11, EC/VFA enabled, n_replay=5.

## Implementation map

- `dean_integration/provenance.py`: aggregate/per-seed start/end UTC times,
  monotonic durations, exit code, terminal COMPLETE/FAILED, exception details,
  output paths, and generated checkpoint hashes. Subprocess failures stop the
  seed sweep; interrupts finalize as FAILED with code 130.
- `scripts/run_dean.py`: wrapper flags, command propagation, asset hashes,
  strict online-only preflight, separate experiment namespace, fresh outputs.
  Official full runs retain `official/cub_v0`; experiments use
  `experiments/cub_online_v1` (strict/SelEx full) or
  `experiments/cub_paired_v1` (paired). Existing run directories are refused.
- `dean_integration/base_checkpoint.py`: tensor-only CPU deserialization;
  exact model/head keys, shapes, dtypes, and finite values checked before any
  weights are copied. No key rewriting, non-strict load, or base fallback.
  CPU reference architecture mirrors upstream's post-DINO replacements.
- `dean_integration/runtime.py` and `external.py`: checked AST hooks only in
  memory. Online-only empties the base loop, replaces the base reload with
  strict validation, chooses the explicit base path, and restores boundary RNG.
  Optional SelEx wraps exactly the online loss expression. Source drift fails.
- `dean_integration/selex_online.py`: deterministic nearest-prototype parent
  hierarchy, synthetic USE soft parent-target CE, synthetic SSE parent cosine
  alignment; inputs are current head prototypes and DEAN pseudo targets only.
  These are interface fixtures, **not paper-faithful SelEx losses**. No claim of
  scientific SelEx equivalence. Hierarchy rebuilds after head expansion.
- `experiment_config.py` / `configs/dean/selex_online_synthetic_v1.yaml`:
  separate nonnegative finite USE/SSE weights, default 0.1 each. The feature
  flag `--use_selex_online` defaults off. When off, original online loss AST
  remains unchanged; the augmentation API returns the same loss object without
  inspecting inputs, doing arithmetic, or consuming RNG.
- `tests/test_dean_experiment.py`: CPU lifecycle, checkpoint rejection,
  command/CLI guards, RNG/stream equivalence, disabled-loss equality and exact
  gradients, USE/SSE forward/backward, AST source-drift tests.
- `paired_stream.py` / `tests/test_dean_paired_stream.py`: deterministic paired
  manifests, fixed-ID batch sampler, independent worker generator, reload and
  compatibility validation, paired CLI/AST/provenance contracts.

## Exact online-only and the historical checkpoint limitation

Full wrapper runs now save `base_boundary_rng.json` beside the base checkpoint
immediately before the base-best reload. This records Python, NumPy, CPU Torch,
and all CUDA RNG states without consuming random draws; it binds the state to
the checkpoint SHA256, seed, upstream/config hashes, official arguments, DINO
hash, and dataset metadata hashes. It captures the **end of all base epochs**,
not the epoch where the best weights were saved.

Online-only restores this state at the same boundary, before init evaluation,
PET/head initialization and stream iterator creation. It leaves the upstream
shuffle/DataLoader/worker behavior intact. Under the same software, device
configuration and dataset bytes, this preserves the full-run RNG progression
and stream order. CPU tests exercise matching shuffled batches after synthetic
base RNG consumption and intervening eval/model initialization. They do not
establish full CUDA training/metric equivalence.

The already completed `cub_VIT_best_step_0.pth` alone cannot reconstruct the
historical post-base RNG state. This implementation refuses exact online-only
without a matching sidecar; it never fabricates one from the initial seed and
never silently retrains. No sidecar was manufactured for the historical run.
If the original process did not save that state elsewhere, exact replay of its
stream remains unavailable. The explicitly requested paired protocol below
allows new controlled comparisons without reconstructing that historical RNG.

Example (Linux official environment; paths are placeholders; not executed):

```bash
python scripts/run_dean.py --action verify \
  --upstream /path/OCGCD --dataset-parent /path/data \
  --checkpoint /path/dino_full.pth --output /path/fresh-output \
  --seed 1 --online-only \
  --base-checkpoint /path/cub_VIT_best_step_0.pth \
  --boundary-rng /path/base_boundary_rng.json
```

Use `--action train` only for an authorized training invocation. Add
`--use_selex_online --selex-config configs/dean/selex_online_synthetic_v1.yaml`
to exercise experimental wiring. The DINO checkpoint remains required for the
unchanged upstream initialization/preflight. Plan/verify do not train.

## Paired online protocol v1

The three experiment categories must be reported separately:

| Category | Initialization and stream | Result status |
|---|---|---|
| A. Historical full-run DEAN baseline | Original seed-1 full base/online run | User-reported All=0.547808, Old=0.631773, New=0.219492; base before online=0.8223 |
| B. Paired online-only DEAN control | Existing validated base checkpoint, reset online RNG, paired manifest, SelEx off | **NOT RUN; must be rerun with the new deterministic stream** |
| C. Paired online-only DEAN + SelEx treatment | Exactly B's checkpoint, official parameters, online seed and manifest; SelEx on | **NOT RUN; current USE/SSE are synthetic interface fixtures** |

Compare C against B to isolate the treatment under this paired protocol. B is
not a replay of A, and A's score must not be assigned to B. No 30-epoch base
retraining is needed to create B/C or their manifests.

Mode selection is explicit:

- No online mode flag: existing official full-run algorithm, loss and shuffle.
- `--online-only`: strict replay requiring `--boundary-rng`; missing or
  incompatible sidecar fails, with no paired-mode fallback.
- `--paired-online`: requires `--seed`, `--base-checkpoint`, `--online_seed`;
  rejects `--boundary-rng` and is mutually exclusive with `--online-only`.
  `--online_seed` and `--stream-manifest` are rejected outside paired mode.

At the audited boundary immediately before the base-best reload, paired mode
prepares/validates the complete incoming stream manifest and resets Python,
NumPy, Torch CPU and all CUDA RNG states. The base loop is empty and the exact
base checkpoint loader is reused. This happens before init evaluation, PET/head
construction and online iteration. The ordinary `--seed 1` still controls the
official dataset/clustering protocol; `--online_seed 17` independently selects
the new online initialization/global RNG and stream order. Seeds are integers
in `[0, 2**32)`.

Each session's sample permutation uses a private CPU Torch generator seeded
with `online_seed + session_id`. A separate DataLoader generator seeded with
`online_seed + session_id + 2**32` controls worker initialization. The loader
uses fixed manifest batches as its batch sampler, so training RNG consumption
cannot change incoming batch membership/order. Original transforms, inner
pseudo-labelled update loaders, EC/VFA, clustering, evaluation, LoRA layers,
and official arguments remain unchanged. The manifest covers incoming samples,
not treatment-dependent pseudo-labels or training partitions within a batch.

`seed1/stream_manifest.json` is canonical JSON containing protocol version,
online/offline seeds, base/DINO hashes, upstream/config hashes, official argv,
dataset metadata hashes, session dataset-ID order, exact batches of sample IDs,
and batch size including the final short batch. Its SHA256 and online seed are
stored in aggregate `provenance.json` and per-seed `execution.json`. The manifest
describes the scheduled full stream; FAILED status does not imply it was fully
consumed. A successful paired worker without a valid manifest fails finalization.

Omit `--stream-manifest` to generate; specify it to read an existing file. Each
arm saves its own canonical copy in its fresh output directory. Reload validates
schema, seed/checkpoint/protocol/metadata binding, batch boundaries and exact
one-time sample coverage; when upstream datasets are built, their session
sample IDs and order must also match. Verify checks manifest schema/context;
the worker performs the actual dataset-ID comparison before online training.
Missing, malformed, incompatible or duplicate-ID manifests fail without
regenerating a replacement. Existing manifest/output artifacts are not overwritten.

## Exact future paired CLI commands (not executed)

Run from the repository root in the official Linux/CUDA environment. Replace
the four asset paths below with the existing files on that host; do not create
a new base checkpoint. Pick a fresh output root. All hyperparameters come from
the unchanged locked official config.

```bash
set -euo pipefail
UPSTREAM=/path/to/OCGCD
DATA_PARENT=/path/to/datasets
DINO=/path/to/dino_vitbase16_pretrain_full_checkpoint.pth
BASE=/path/to/cub_VIT_best_step_0.pth
PAIR_ROOT=/path/to/runs/dean_paired_seed1_online17_v1

COMMON=(--config configs/dean/cub_official_v0.yaml
  --upstream "$UPSTREAM" --dataset-parent "$DATA_PARENT"
  --checkpoint "$DINO" --base-checkpoint "$BASE"
  --seed 1 --paired-online --online_seed 17)

# B: validate and then run the new paired DEAN control, SelEx disabled.
python scripts/run_dean.py --action verify "${COMMON[@]}" --output "$PAIR_ROOT/control"
python scripts/run_dean.py --action train  "${COMMON[@]}" --output "$PAIR_ROOT/control"

MANIFEST="$PAIR_ROOT/control/experiments/cub_paired_v1/seed1/stream_manifest.json"

# C: validate and then run treatment on B's exact manifest.
python scripts/run_dean.py --action verify "${COMMON[@]}" \
  --output "$PAIR_ROOT/treatment" --stream-manifest "$MANIFEST" \
  --use_selex_online --selex-config configs/dean/selex_online_synthetic_v1.yaml
python scripts/run_dean.py --action train "${COMMON[@]}" \
  --output "$PAIR_ROOT/treatment" --stream-manifest "$MANIFEST" \
  --use_selex_online --selex-config configs/dean/selex_online_synthetic_v1.yaml

# Both hashes must match. Also compare stream_context in both provenance files.
sha256sum "$MANIFEST" "$PAIR_ROOT/treatment/experiments/cub_paired_v1/seed1/stream_manifest.json"
```

For another paired-control repeat, use a fresh output root and pass the same
`--stream-manifest "$MANIFEST"` with SelEx left disabled. Do not substitute an
old RNG sidecar or omit `--paired-online` to get around a validation failure.

Strict replay remains available only when the original matching sidecar exists:

```bash
python scripts/run_dean.py --action verify \
  --upstream "$UPSTREAM" --dataset-parent "$DATA_PARENT" --checkpoint "$DINO" \
  --base-checkpoint "$BASE" --seed 1 --online-only \
  --boundary-rng /path/to/base_boundary_rng.json --output /path/to/fresh-strict-replay
```

The existing seed-1 run has no such sidecar. No retraining was done to make one.

## Paired-mode validation

Final full-suite command on the existing local environment:

```powershell
$env:DEAN_UPSTREAM_CHECKOUT = (Resolve-Path ../OCGCD-dean-audit).Path
.venv/Scripts/python.exe -m unittest discover -s tests -v
# Ran 232 tests in 145.251s — OK
```

The new paired test module independently passed 13 tests. The full suite covers
the final source, including both control/treatment CLI context and manifest
reuse, identical incoming batches despite different global RNG consumption,
two-worker DataLoader order, repeated manifest bytes/hashes, online-seed changes,
Python/NumPy/CPU/CUDA seed reset, malformed/incompatible manifest rejection,
no overwrite/fallback, missing-manifest provenance failure, official full-run
and disabled-loss contracts, strict sidecar requirements, and pinned external
AST compilation for both paired arms. Existing frozen baseline tests passed.

A real `--action plan --paired-online --seed 1 --online_seed 17` against the
pinned checkout confirmed mode `paired`, SelEx off by default and unchanged
official arguments. It used placeholder checkpoint paths and did not load data
or train. `git diff --check` passed and upstream remained clean. No GPU training,
base retraining, dependency installation, commit or push was performed.

Files changed for the paired extension (earlier uncommitted work is preserved):

- `scripts/run_dean.py`
- `dean_integration/external.py`
- `dean_integration/runtime.py`
- `dean_integration/paired_stream.py` (new)
- `dean_integration/provenance.py`
- `tests/test_dean_paired_stream.py` (new)
- `docs/DEAN_SELEX_ONLINE_EXPERIMENT.md`

## Earlier validation (before paired mode)

Measured locally with `DEAN_UPSTREAM_CHECKOUT` set to the pinned external
checkout:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
# Ran 218 tests in 142.954s — OK

.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_dean*.py' -v
# Final targeted rerun: Ran 52 tests in 22.594s — OK
```

The targeted rerun includes the final added configuration validation test and
all final source changes; unrelated code was not changed after the full suite.
Both runs enabled external CPU fixtures, including original online loss and
tiny ViT/PET update tests. `git diff --check` passed, upstream stayed clean, and
the online-only plan retained official upstream arguments with SelEx disabled.

Initial checks with the system Python passed the dependency-light subset but
could not collect the whole suite because scipy/sklearn/threadpoolctl were
missing; external imports also lacked timm. The existing repository `.venv`
already supplied those dependencies and Torch 2.0.0+cu118. No packages were
installed. Its environment has no pytest, so final validation used unittest.
GPU training and the historical seed-1 result were not rerun.
