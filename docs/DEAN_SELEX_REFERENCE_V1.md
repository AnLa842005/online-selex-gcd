# SelEx adapted to DEAN online discovery: reference v1

This implementation follows the reviewed design in
`DEAN_SELEX_REAL_INTEGRATION_PLAN.md`. The design-only commit is
`d2fd3f20dece59b4487be0bcda42c764c24d8f2b` with message
`docs(dean): finalize real SelEx online integration design`. Implementation release
is gated by the final pre-commit review below. No GPU training has been performed.

This is **SelEx adapted to DEAN online discovery**, not a literal reproduction
of offline SelEx. The old `dean_integration/selex_online.py` and
`selex_online_synthetic_v1.yaml` remain **synthetic test scaffolding**, selected
only by the existing `synthetic` implementation identity. They are not the real
loss and have not been silently replaced.

## Selection and exact bypass

Use both flags to select the real implementation:

```text
--use_selex_online --selex-implementation reference_v1
--selex-config configs/dean/selex_online_reference_v1.yaml
```

The default implementation identity remains synthetic for compatibility with
previous experiment provenance. A real config passed to the synthetic loader
is rejected. A missing enable flag, or real gamma=0, takes the exact original
DEAN path: no real AST hooks, projector, auxiliary image reads, RNG scope,
prototypes, optimizer groups or extra arithmetic. Config parsing and wrapper
provenance still describe the requested setup. Strict replay and paired online
mode retain their existing explicit selection and requirements.

The primary config uses gamma=0.1, output_dim=65536, hidden_dim=2048,
bottleneck_dim=256, SSE mixture=0.35, CUB smoothing=1, USE temperature=1,
SSE/base temperature=0.07, hierarchy_decay=0.5, prototype momentum=0.9,
capacity=512, projector lr=1e-4, epsilon=1e-8. Small test dimensions are supplied
explicitly via a separate config/dataclass; production dimensions are never
automatically reduced.

## Source correspondence and deliberate changes from c3d3b42

Reference SHA: `c3d3b42a0aa7d7c9ea748a8305a6890a34100f73` in
`SarahRastegar/SelEx`. Exact permalinks/equations are in the design audit.

| Reference behavior | v1 implementation and reason |
|---|---|
| DINOHead three-layer MLP, normalized bottleneck, weight-normalized last layer with frozen gain | Same architecture, initialization order, normalization axis and trainability; output configurable, default 65536 |
| Finest semi-supervised K-means over all offline training features | No competing leaf discovery: use authoritative DEAN pseudo leaf IDs |
| Higher-level BSSK with raw dataset-wide features | Recursively merge nearest EMA prototype groups separately within base/novel sides; halve group counts; no raw history bank or future dataset scan |
| Labels returned in reordered known/unlabelled partitions, then consumed as loader order | Explicit sample-ID/class-ID mapping throughout; DEAN IDs remain unchanged; all parent IDs are globally disjoint negative integers and never recycle across rebuilds |
| Global feature norm in optional cosine branch; unnormalized batch geometry versus normalized clustering geometry | Euclidean-only v1, feature-axis normalized 768-D geometry and EMA prototypes; all projected outputs normalized along their feature axis |
| Broadcast confusion division by a vector of row sums; no denominator guard; no self-removal renormalization | Detached targets, remove self, stable per-row normalization with epsilon-clamped denominator |
| Potential zero target mass / no positives / empty means and divisions | Invalid term/anchor contributes differentiable finite zero with counters; no one-hot fallback; structural/nonfinite input errors remain hard errors |
| Singleton class still has the other view as a positive | Reviewed v1 requires at least two DISTINCT real samples in a semantic group; singleton SSE anchors and unsupported USE level groups are skipped, while valid terms remain |
| Original distance exponentials may underflow when self is the row maximum | Stable non-self logsumexp, retaining unsquared Euclidean distances and temperature scaling |
| Full-dataset hierarchy at epoch start | Bounded normalized EMA class prototypes, refreshed after discovery once per incoming stream batch; fixed assignments/centers/radii during all 15 inner epochs |
| Two augmented views per offline optimization step | Two fresh independent auxiliary real-image views per online optimization update, from current raw files only |
| GT supervised term from the labelled dataset subset | GT term accepts only explicit `LabeledReplay`; pseudo labels contribute only hierarchy grouping; wrapper defaults to no replay |
| SGD with full-block fine-tuning | DEAN LoRA/AdamW/CE/weighted EC preserved; persistent projector AdamW group immediately before final expandable classifier |
| SelEx alone is the training objective | Add gamma times the reference internal USE/SSE mixture to the unchanged DEAN objective |

SSE uses the source's coordinate-prefix geometry: normalize the FULL projected
vector, then slice prefixes without re-normalizing each prefix. K-means source
bugs are not retained as a production mode. Test-only original AST extraction
is used for mathematical comparisons on supported tensors.
Skipping a singleton SSE group means its own anchors contribute zero; its
features may still serve as negatives for another supported group's anchors.

## Lifecycle and data access

1. After the upstream student is created, initialize the auxiliary projector
   inside a private Python/NumPy/Torch CPU/used-CUDA RNG scope.
2. Let DEAN finish EC splitting, VFA/AP/GMM discovery and leaf-label offsets.
   VFA feature replicas are never treated as real images or contrastive views.
3. Read only filepath/image data for the CURRENT merged pseudo-labelled batch,
   keyed by `uq_idxs`. The raw adapter does not invoke the dataset's GT-returning
   `__getitem__`, read `sample.target`, or consume `gt_y` diagnostics.
4. Under no_grad/eval, extract current normalized 768-D student pre-classifier
   features using deterministic eval transforms. Restore all original module
   training flags. Update class EMA prototypes, evict least recently updated
   inactive leaves with deterministic ties, and rebuild the hierarchy once.
   Capacity cannot silently exclude any current class.
5. For each optimization update, retain DEAN's original CE/EC computation.
   Read current real images, generate TWO new independently augmented views in
   the private RNG scope, forward them through the same live student and head,
   and compute USE/SSE. The hierarchy stays fixed. No historical raw images or
   feature samples are cached; the current provider is replaced on next batch.
6. Optional explicit available labelled replay is independently augmented and
   forwarded for the GT SSE term ONLY. It does not alter current prototypes,
   pseudo-label hierarchy or USE targets. Replay IDs must be disjoint from the
   current stream. The CLI intentionally does not auto-create a labelled replay
   buffer or reread base training images; the integration API can receive an
   explicit `LabeledReplay` from a separately authorized provider.
7. One backward updates LoRA, projector and DEAN classifier as appropriate;
   frozen teacher/backbone/embedding parameters remain frozen. Projector state
   survives the ORIGINAL DEAN classifier expansion/state-copy logic.

The private RNG maintains its own advancing states but restores the caller's
Python, NumPy, CPU and used CUDA states even on exceptions. It does not seed or
modify the paired stream's dedicated generator. It is not a full-training resume
checkpoint; strict replay still needs its original base-boundary sidecar.

## Loss and telemetry

```text
USE = soft-target CE on normalized Euclidean two-view contrastive logits
H = sum_i hierarchy_decay^(i+1) * SupCon(projected_prefix_i, hierarchy_labels_i)
GT = SupCon(full_projected_explicit_replay, true_labels)  # zero without replay
SSE = (GT + H)/2
SelEx = (1-sse_weight)*USE + sse_weight*SSE
Final = CE + EC_weighted + gamma*SelEx
EC_weighted = args.energy_hp * upstream_energy_loss
```

`selex_reference_metrics.jsonl` records session/config/source/RNG metadata,
per-batch support and hierarchy depth, optimizer groups and trainable counts,
then CE, EC, USE, SSE, GT SSE, hierarchy SSE, SelEx total and final total for
EVERY update. Skip counters distinguish empty groups, semantic singleton
groups, no-positive anchors, zero-target rows, missing hierarchy, zero prefix
dimensions, invalid pseudo labels, and prototype evictions. Replay sample
counts are explicit. `selex_reference_summary.json` contains config, source,
RNG seed, last batch/update, cumulative counters, bounded prototype counts,
projector trainable count and optimizer groups. Aggregate provenance copies
the summary into the seed execution record and hashes it. A successful real
worker without research telemetry fails finalization.

This change does not repair DEAN's original EC mask selection or log-ratio
domain. Nonfinite official losses in active real mode fail clearly; they are
not silently clamped or attributed to a skipped SelEx term.

## Resource implications of the reviewed two-view policy

Production head trainable count is 23,073,024; frozen weight-norm gains add
65,536 parameters. With AdamW FP32 weight, gradient and two moments, head state
is about 352.32 MiB, excluding effective weight-norm matrices and activations.

The approved output_dim=768 ablation STILL uses the three-layer head. It has
6,492,416 trainable parameters, not zero. The earlier audit's direct-embedding
zero-parameter alternative is not the approved v1 ablation.

Two FRESH auxiliary views cannot reuse DEAN's original fixed image tensor
across all 15 updates. Therefore each update has one original DEAN image forward
plus a batched two-view auxiliary forward/backward: approximately THREE image
exposures per original image/update, not the earlier candidate's two. This is
an analytic compute accounting, not a measured runtime multiplier. An additional
detached feature extraction occurs once per stream batch. The upstream optimizer
state deepcopy during classifier expansion now also copies projector Adam state;
its transient memory must be included in a future GPU profile. No GPU memory,
throughput, convergence or accuracy result has been measured in this task.

## Future CLI (not executed)

Use the existing official Linux environment. Asset paths below are placeholders;
the base checkpoint is the already completed one. No base retraining is needed.
Use separate fresh outputs for control/treatment.

```bash
set -euo pipefail
COMMON=(--upstream /path/to/OCGCD --dataset-parent /path/to/datasets
  --checkpoint /path/to/dino_vitbase16_pretrain_full_checkpoint.pth
  --base-checkpoint /path/to/cub_VIT_best_step_0.pth
  --seed 1 --paired-online --online_seed 17)

# New paired DEAN control; no SelEx flag.
python scripts/run_dean.py --action train "${COMMON[@]}" --output /path/to/fresh-control
MANIFEST=/path/to/fresh-control/experiments/cub_paired_v1/seed1/stream_manifest.json

# SelEx adapted to DEAN online discovery, on the exact control stream.
python scripts/run_dean.py --action verify "${COMMON[@]}" \
  --output /path/to/fresh-treatment --stream-manifest "$MANIFEST" \
  --use_selex_online --selex-implementation reference_v1 \
  --selex-config configs/dean/selex_online_reference_v1.yaml
python scripts/run_dean.py --action train "${COMMON[@]}" \
  --output /path/to/fresh-treatment --stream-manifest "$MANIFEST" \
  --use_selex_online --selex-implementation reference_v1 \
  --selex-config configs/dean/selex_online_reference_v1.yaml
```

The historical full-run metrics cannot be assigned to the new paired control.
These training commands have NOT been run. Routine unit tests use small head
dimensions; a separate one-off production-dimension CPU probe is recorded below.

## Delivered diff and local validation

The implementation covers the following 16 files:

| File(s) | Change |
|---|---|
| `configs/dean/selex_online_reference_v1.yaml` | Reviewed production defaults and configurable output/internal weights |
| `dean_integration/selex_config.py` | Strict real config validation and source revision identity |
| `dean_integration/selex_rng.py` | Advancing private host/used-CUDA RNG scopes with exception-safe restoration |
| `dean_integration/selex_projector.py` | Three-layer reference DINOHead |
| `dean_integration/selex_hierarchy.py` | Bounded EMA, deterministic prototype-only higher groups, unique IDs and fixed snapshots |
| `dean_integration/selex_reference.py` | Corrected USE/SupCon/SSE, detached targets and skip counters |
| `dean_integration/selex_real_online.py` | Current-image/replay boundaries, fresh per-update views, group assertions and telemetry |
| `dean_integration/experiment_config.py` | Explicit synthetic/reference config dispatch |
| `dean_integration/runtime.py`, `dean_integration/external.py` | Fail-closed in-memory session/batch/optimizer/loss hooks; exact inactive bypass |
| `scripts/run_dean.py` | CLI selection, child command propagation and real metadata; lazy active engine construction |
| `dean_integration/provenance.py` | Research summary/config/hash capture, successful-update validation, failed-worker code preservation |
| `tests/test_dean_selex_reference.py` | Original AST head/loss/gradient parity on non-degenerate small tensors, corrected targets and degeneracy |
| `tests/test_dean_selex_hierarchy.py` | Bounded EMA, identity, sparse/reordered IDs, all-known/all-novel/empty/tied support |
| `tests/test_dean_selex_real_online.py` | Fresh views, RNG and CUDA-state mock, true replay separation, live gradients, real worker CLI, telemetry and pinned DEAN expansion |
| `docs/DEAN_SELEX_REFERENCE_V1.md` | Usage, all intentional departures, file map, resource accounting and measured validation |

Initial implementation suite, before the final pre-commit audit:

```powershell
$env:DEAN_UPSTREAM_CHECKOUT = (Resolve-Path ../OCGCD-dean-audit).Path
.venv/Scripts/python.exe -m unittest discover -s tests -v
# Ran 267 tests in 224.590s — OK
```

There are 35 new real-SelEx tests; all passed independently as well. The suite
includes all previous 232 tests, exact inactive/gamma-zero AST behavior,
original reference scalar/gradient comparisons, 15 consecutive fresh-view
updates with fixed hierarchy, real LoRA-style/projector gradients, replay-only
GT supervision, source DEAN optimizer expansion and a subsequent successful
AdamW step, research provenance and partial-log failure handling. CUDA state
isolation is unit-tested with mocked device states; actual GPU execution has
not been validated.

`git diff --check`, new-file trailing whitespace checks and Python 3.9 syntax
checks passed. A real plan command using the production YAML confirmed paired
mode, `reference_v1`, gamma=0.1 and output_dim=65536 without creating the head or
loading data. No dataset/checkpoint/baseline artifacts were modified; upstream
stayed clean. This initial validation preceded the pre-commit review below.

## Final pre-commit audit (2026-11-05)

All 16 files were read before edits and checked against the reviewed design,
SelEx source c3d3b42, infrastructure d1c7a030 and design d2fd3f20. The audit found
real gaps in mixed-precision handling: CPU cdist did not support FP16/BF16, and
FP16 epsilon/reductions could underflow/overflow. The implementation now keeps
normalization, geometry, contrastive distances and logsumexp in FP32 for low-
precision inputs, while preserving differentiable casts and ordinary AMP
linear layers. EMA state is always detached CPU FP32, with autocast disabled.
Numerical config values must be representable in FP32; temperatures/epsilon
cannot underflow to zero. Zero terms use scalar selection instead of an unsafe
large reduction. FP32 reference initialization/forward/gradient parity remains
covered by the unchanged reference fixtures.

The review also removed avoidable projection-sized packing copies: the live
`[2B,D]` tensor is exposed as a `[B,2,D]` view, and view-major conversion/prefixes
share storage where layout permits. Prefix width always follows the reference
`D//2^(level+1)`, independently of configurable hierarchy loss weights. A
nonpositive width is skipped and counted BEFORE any slice/kernel is created.

| Gate | Verified behavior/evidence |
|---|---|
| 1. No 65536x65536 matrix | Static axes review, batch-axis cdist/identity regression and actual production-shape dispatch trace; loss pairwise arrays are 2B x 2B. Prototype coarsening separately uses bounded K x K distances with K<=capacity, never projection dimensions |
| 2. Detached 768-D EMA/hierarchy | no_grad update/rebuild; CPU FP32 normalization; fixed snapshots and AMP state regression |
| 3. Projection bound | Production probe emitted [4,65536] for B=2; [B,2,D] interfaces are storage views, not feature-by-feature matrices |
| 4. Safe prefixes | Storage-alias test, positive-width assertions and independent weight/prefix regression |
| 5-6. Optimizer expansion/order | Original pinned DEAN expansion AST, persistent projector moment buffers, classifier final, successful next AdamW update |
| 7. RNG isolation | Fresh per-update views; independent Python/NumPy/CPU/used-CUDA states; CUDA state mock; paired manifest hash unchanged |
| 8. Exact bypass | Disabled/gamma-zero avoids real construction/hooks and consumes no SelEx RNG; pinned bytecode/CLI regressions |
| 9. No raw history | Only current-batch image provider plus bounded class prototypes/counts; old providers are replaced |
| 10. Degenerate terms | Differentiable finite zero and named counters; overflow-safe zero regression; valid terms continue |
| 11-12. True vs pseudo supervision | Explicit labelled replay is the sole GT SSE input; pseudo labels only select hierarchy groups in SelEx. Original DEAN pseudo-label CE remains unchanged |
| 13. Logging | Per-update CE, weighted EC, USE, SSE, SelEx/final totals, skip counters; optimizer groups/parameter counts in telemetry and hashed provenance |
| 14. Validation | Invalid dimensions, temperatures, weights, epsilon and unrepresentable FP32 settings rejected |
| 15. Python/AMP | All 14 changed Python files compiled by actual CPython 3.9.13; CPU BF16 autocast integration and FP16/BF16 kernel backward tests; CUDA AMP execution still requires the future GPU smoke |

Six targeted regression tests were added only for gaps identified during this
review. The real-SelEx targeted suite now has 41 passing tests.

### One-off production projector measurement

One production head was instantiated on CPU using the unchanged production YAML,
Torch 2.0.0+cu118, Python 3.11.9 and two CPU threads. B=2 real-image feature rows
with two views produced `[4,768] -> [4,65536]`. This measured the head and actual
USE/SSE kernels, not a complete ViT or image-loader training run. A dispatch
guard observed 638 tensor operations, rejected feature-by-feature allocations
and traced all outputs. Loss distance matrices were 4x4; the largest 2-D tensor
was the expected final weight/gradient `[65536,256]`. CUDA was not initialized.

| Quantity | Actual/estimated value |
|---|---|
| Trainable parameters | 23,073,024 |
| Frozen gain parameters | 65,536 |
| Total registered parameters | 23,138,560 |
| FP32 parameter bytes (all parameters) | 92,554,240 bytes = 88.266602 MiB |
| Trainable parameter/gradient bytes (each) | 92,292,096 bytes = 88.016602 MiB |
| Estimated AdamW moments | 184,584,192 bytes = 176.033203 MiB |
| Actual AdamW state after one step | 184,584,220 bytes (moments plus seven scalar step tensors) |
| Head construction | 0.339406 s |
| Forward + USE/SSE + backward | 0.264668 s |
| AdamW step | 0.287001 s |
| Process runtime including imports | 5.254287 s |
| Peak process working set | 939,487,232 bytes = 895.964844 MiB |
| Peak Windows process commit | 1,647,095,808 bytes = 1570.792969 MiB |
| Peak working-set increase over pre-head baseline | 564.242188 MiB |

The scalar loss and every trainable gradient were finite. The production head
was not instantiated in the regular unit suite. Scratch probe code/results are
local ignored artifacts, not datasets/checkpoints or committed binary files.

CPython 3.9.13 was obtained as the official Python.org embeddable distribution
in a fresh temporary directory, without changing the existing test environment.
Its published archive checksum was checked; archive SHA256 was
`938a1f3b80d580320836260612084d74ce094a261e36f9ff3ac7b9463df5f5e4`.
Actual py_compile compiled every changed Python file, not just a grammar check.
The full Torch test suite still runs in the existing Python 3.11.9 environment;
this does not claim a full dependency/runtime test on Python 3.9.

CPU peak RSS is not GPU VRAM. GPU risks remain the ~352.32 MiB FP32 head +
gradient + moments, additional materialized weight-normalization buffers,
FP32 128x65536 projections at batch 64 (32 MiB each), cdist/backward temporaries,
two fresh auxiliary-view graphs alongside the original DEAN graph, and the
upstream optimizer-state deepcopy during classifier expansion. There is no
measured batch-64 GPU peak or throughput claim. Before GPU smoke, the real Linux
CUDA environment and actual base/DINO/dataset assets must pass preflight, then
bounded forward/backward and expansion memory must be profiled. A missing
historical RNG sidecar is not a blocker for the explicit paired protocol.

Final release gates passed with pinned external CPU fixtures enabled:

```powershell
$env:DEAN_UPSTREAM_CHECKOUT = (Resolve-Path ../OCGCD-dean-audit).Path
.venv/Scripts/python.exe -m unittest discover -s tests -v
# Ran 273 tests in 218.540s — OK
```

The targeted real-SelEx suite passed 41 tests in 4.421 seconds. Actual CPython
3.9.13 py_compile passed all 14 changed Python files. Both ordinary and staged
`git diff --check` passed. The reviewed design file remained unchanged after
its separate d2fd3f20 commit; the implementation stayed within the 16-file scope
listed above. No dataset/checkpoint/baseline/upstream files were modified.
The implementation release commit and design commit are pushed together only
after these gates, without merge, PR creation or GPU training.
