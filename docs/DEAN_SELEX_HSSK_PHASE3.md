# Phase 3: Online HSSK + USE/SSE, bounded smoke only

Status: **smoke_ready, full_train_blocked** (2026-11-06).
No GPU run has been performed. Pin the reviewed Phase-3 commit SHA from the
commit report before preflight or bounded smoke.
The [design contract](DEAN_SELEX_DESIGN_CONTRACT.md) and
[architecture realignment](DEAN_SELEX_ARCHITECTURE_REALIGNMENT.md) remain authoritative.
Phase 2 is committed/pushed at `67eed0ede1cc26dd6bd7e4bc024c0b6187c72892`;
the protected baseline branch remains at `d7ba08745a8043a87d3126da8366a4720d81951f`.

`hssk_online` is the primary provider, called **Online HSSK + USE/SSE**.
`dean_prototypes` is only the explicitly selected **SelEx-inspired prototype
hierarchy** ablation. Neither insufficient support nor a smoke gate selects it
as fallback. Projector and loss kernels are shared and unchanged.

## Observation and hierarchy flow

1. DEAN discovery, energy/AP/GMM/VFA and classifier expansion finish normally.
   HSSK receives current image IDs and deterministic eval embeddings, not DEAN
   labels, classifier IDs/weights or discovery masks. The current-image adapter
   reads filepaths only and never calls the GT-returning dataset item interface.
2. Prepare verified replay once, in complete two-reference class groups, using
   the existing canonical round-robin cursor. Selected IDs/hash stay fixed for
   all 15 optimization epochs. Re-extract current and replay features under
   eval/no_grad with autocast disabled, then restore every module training flag.
3. Run the unchanged Phase-2 leaf fitter against a proposed copy of HSSK memory.
   This preserves its exact K policy, real-row support gates and seed-only memory.
4. Higher levels receive ONLY child centers, real support weights, GT-derived
   anchor-group identities, HSSK IDs and lineage. Group anchored child centers
   into ceil(A/2) weighted hypergroups; these hyperlabels stay fixed in the
   subsequent semi-supervised fit. Target free-parent count is ceil(F/2).
   Two-phase BSSK balances CHILD row counts, while centroid means/objective
   geometry use REAL-image support weights. Free children may join anchored
   parents; they never acquire GT supervision from that assignment.
5. Reject an unsupported proposed level and stop at the last valid depth:
   require strict group-count reduction, two distinct children per free parent,
   at least two real images per parent, finite geometry, and positive prefix
   width D//2^(level+1). In particular odd F may make ceil-halving infeasible;
   do not fabricate a one-child free parent to continue the tree.
6. Compose sample-to-leaf-to-ancestor maps, allocate disjoint decreasing negative
   parent IDs from the proposed HSSK allocator, and publish the snapshot/memory
   transaction only after validation. Leaf IDs can retain their persistent track
   identity; new parents never collide with leaf tracks or earlier parent IDs.
   Radii are epsilon-clamped support-weighted child-to-parent distances.
7. The snapshot copies detached geometry into canonical immutable storage. Hashing
   canonicalizes JSON keys BEFORE hashing, including nested integer-keyed child
   maps, so checkpoint state round-trips preserve the exact snapshot hash.

No raw feature rows are re-read by higher levels. The verified fixtures produce
four levels with group counts [8,4,2,1] without replay, and the production-dimension
probe produces five levels [32,16,8,4,2] with 16 anchored classes.

## Reuse, skip, and state boundaries

Insufficient fresh leaf support attempts transfer of the previous successful
HSSK tree only. Check model lineage, dataset, feature/normalization version,
config/schema, seed and seed protocol; source age must be exactly one incoming
batch and the preceding decision cannot be REUSE. Every current feature must
be within Euclidean distance 0.5 of its nearest prior leaf, every selected GT
class must retain a compatible anchored track, and real leaf/parent support
must remain valid. Current IDs get NEW assignments to the OLD geometry/tree.
No second view or memory center supplies support.

Reuse keeps centers/radii/tree and memory/allocator unchanged, recording source
batch, age, previous snapshot and reuse hash. It never replaces the successful
source snapshot or chains into another reuse. Failed compatibility/support takes
`reuse_then_skip` (whole auxiliary batch differentiable zero, counters) or
`reuse_then_error` (raise before optimization). Structural errors remain hard.
With a valid hierarchy, missing verified GT omits only GT and invalid individual
terms retain the existing finite-zero/counter behavior.

Because the locked K enumeration includes a supported single free leaf for
ordinary unanchored N>=2, successful fresh fits are common. Reuse-branch tests
inject a leaf support failure without inventing an extra confidence gate or
changing the locked K policy.

Provider state serializes independent memory, the previous immutable snapshot,
identity, counters, last batch and status. Restore validates config, hashes,
snapshot/memory/source-batch consistency and provider identity. This is NOT full
trainer resume. Model/optimizer/stream/augmentation continuation remains Phase 4;
full training is blocked until that readiness is reviewed.

## Live views and loss

Current samples receive two fresh augmented views per update, projected separately
from the selected replay views. Lookup keys are stable sample IDs; both views
inherit the same hierarchy assignment. Replay has its own augmentation RNG and
only verified reference labels reach GT SupCon. Caller-supplied replay labels,
DEAN pseudo labels and current diagnostic gt_y cannot become GT inputs.

```text
H = sum_l hierarchy_decay^(l+1) * SupCon(projected_prefix_l, HSSK_labels_l)
GT = SupCon(full_projected_verified_replay, true_labels)
SSE = (GT + H)/2
SelEx = (1-sse_weight)*USE + sse_weight*SSE
L_online = CE + EC_weighted + gamma*SelEx
```

The 65536-D three-layer DINOHead and corrected kernels are unchanged. Full-vector
normalization precedes positive-width prefix VIEWS; prefixes are not renormalized.
Pairwise losses remain sample-by-sample. Gradients reach the live LoRA/projector;
no graph crosses hierarchy fitting. The projector AdamW group remains immediately
before the final expandable classifier, retaining its optimizer state.

HSSK fitting has a private deterministic seed protocol. Replay selection is
cursor-based and draws no RNG. Eval extraction, projector initialization, current
augmentation and replay augmentation have isolated private domains. Caller
Python/NumPy/Torch CPU/used-CUDA RNG and the paired stream generator are restored.
Disabled/gamma-zero bypasses provider/projector/replay/image/RNG construction.

Metrics log CE, weighted EC, USE, SSE/GT/hierarchy, SelEx and final total separately.
The reviewed RunPod smoke protocol caps the action at **two optimizer updates**
(default one), without changing configured inc_epochs=15 or any model/loss
hyperparameter. `smoke_diagnostics.jsonl` additionally records commit/source/
config/asset identities, exact selected replay IDs/hash, synchronized hierarchy
and full-update wall times, finite/nonzero projector/LoRA gradient checks after
each optimizer step, CUDA peak allocated/reserved and completion/error exit code.
The in-memory metrics are copied to the final summary. CUDA OOM is recorded and
re-raised; dimensions, batch and replay are never reduced automatically.
Summary/aggregate provenance includes provider, status, K/levels/support, memory
and snapshot hashes, replay selection hash/counters and bounded completion.
A successful worker without the exact post-step smoke completion/update count
or hierarchy provenance is rejected during finalization.

## Verified eligibility and future RunPod preflight

The wrapper reconstructs only the original train_labelled split from the three
CUB metadata tables, pinned upstream SHA/tree hash, prop_samples=0.8 and the
internal NumPy seed 0 reset PER CLASS. Preserve original training-row uq_idxs,
then canonicalize output records. Use isolated RandomState so generation does
not perturb DEAN/stream RNG. The enclosing manifest records source, parameters,
metadata hashes, dataset/split fingerprints and the exact eligible ID/path/GT
triples. Existing manifests must match both their trusted canonical SHA and a
fresh reconstruction against the actual dataset/source. Wrong source/split/
dataset/hash fails. Output creation is exclusive; no overwrite of old manifests.

The real CLI preflight verifies the pinned checkout and complete official CUB
inventory. Library tests use tiny explicit fixtures; they are never emitted as
experiment manifests. No real CUB directory is available at the checked local
workspace path, so no real experiment eligibility manifest was generated here.

Example Linux/RunPod paths below must be replaced with actual validated assets.
These commands are provided for later review, NOT executed in this step:

Before using the smoke command, pin the exact Phase-3 commit SHA from the review
report and require a clean checkout. Verify the pinned external SHA, then run
`scripts/run_dean.py --action verify` with the same dataset, DINO/base checkpoints,
`--paired-online --seed 1 --online_seed 1` and an isolated preflight output path.
Do NOT enable SelEx in that asset-only verification action: active HSSK full-run
actions intentionally remain blocked. This verification checks Linux/CUDA,
complete real CUB files, DINO teacher keys and the finite compatible base model.
Record `nvidia-smi` output separately. Then reconstruct the eligibility manifest
below, re-read it through `verified_manifest` with its canonical SHA, assert the
config remains 65536-D with 32 selected replay references, and check that
`ensure_provider_ready(config)` raises `FullTrainingBlocked`, while its explicit
bounded-smoke form is permitted. `SmokeBudget(3)` must raise. None of these
preflights retrains base or uses a one-batch accuracy result.

```bash
python scripts/build_dean_replay_manifest.py \
  --upstream /workspace/OCGCD-dean \
  --dataset-root /workspace/datasets/CUB_200_2011 \
  --output /workspace/cub_seed1_eligibility.json
# Record canonical_sha256 printed by preflight.

python scripts/run_dean.py --action online-smoke \
  --upstream /workspace/OCGCD-dean \
  --dataset-parent /workspace/datasets \
  --checkpoint /workspace/checkpoints/dino_vitbase16_pretrain.pth \
  --base-checkpoint /workspace/checkpoints/cub_VIT_best_step_0.pth \
  --paired-online --seed 1 --online_seed 1 \
  --use_selex_online --selex-implementation reference_v1 \
  --selex-config configs/dean/selex_online_hssk_v1.yaml \
  --replay-manifest /workspace/cub_seed1_eligibility.json \
  --replay-manifest-sha256 CANONICAL_SHA256_FROM_PREFLIGHT \
  --smoke-updates 2 --output /workspace/runs/hssk_phase3_smoke_fresh
```

If no manifest path is supplied, the explicit bounded action reconstructs and
stores one under its new isolated run directory. It never retrains base or
infers eligibility from a label range. It verifies actual upstream train_labelled
IDs against the manifest before reading selected replay images.
Expected worker diagnostics are under
`OUTPUT/experiments/cub_hssk_smoke_v1/seed1/`: `smoke_diagnostics.jsonl`,
`selex_reference_metrics.jsonl`, `selex_reference_summary.json`, `execution.json`,
`console.log`, replay eligibility and stream manifests. Aggregate provenance is
in that namespace's `provenance.json`. Recommend an external 30-minute timeout
with a 60-second termination grace and capture the launcher's exit code too.
On CUDA OOM, preserve the error/peaks, stop, and review; never reduce 65536-D,
current batch 64, or replay size to claim production success. No full stream,
Phase 4 work, merge, or GPU execution is authorized by this local preparation.

## Smoke/full-training gate and source adaptations

Only `online-smoke` / its internal `smoke-worker` permits active HSSK execution.
It requires explicit paired-online mode, one seed and validated checkpoints,
one incoming batch and 1..2 optimizer updates (default 1). The CLI budget enforces
these smoke bounds; the CPU hierarchy-lifecycle test still verifies a frozen
snapshot through all 15 configured epochs. A fail-closed AST hook stops immediately AFTER the final
allowed optimizer step, before inference on the full test set. Initial full-base
GPU evaluation is suppressed ONLY for this smoke; base accuracy is None and
logged as not measured, never a fabricated numeric baseline result. The 30 base
epochs are skipped via the existing paired mode. All other official arguments
remain unchanged, including configured inc_epochs=15 and VFA n_replay=5.

This bounded diagnostic is not a paired accuracy result: suppressing the base
evaluation also changes its DataLoader RNG consumption. The exact stream
manifest stays isolated; full paired runs remain blocked. The AST changes are
in-memory only; the external checkout is untouched. No full-trainer resume or
scientific production result is implied by smoke completion.

Four existing scaffold-gate tests retain their role but now assert actual CPU
hierarchy construction / smoke_ready / Phase-4 full-training rejection rather
than the obsolete algorithm-incomplete condition. No numeric legacy test or
loss kernel is weakened or removed.

| SelEx c3d3b42 reference | Adaptation | Reason | Scientific effect | Test / ablation |
|---|---|---|---|---|
| Higher levels refit full sample rows after known prototype grouping | Weighted BSSK over child centers only; GT-derived fixed hyperlabels, nested ancestry | Locked online contract and bounded history | Strict nesting replaces raw-row coarse reassignment | Center-only spy, nested mappings, support/width tests; future offline-coarsening comparison |
| Balanced capacity counts dataset samples | Coarse balance counts distinct child rows; real support weights centroid means and radii | Child centers are fitting rows; weights preserve real mass without duplicating samples | Different coarse partitions from uniform-prototype/whole-dataset fitting | Child/support constraints; future uniform-vs-support-weight ablation |
| Offline periodic whole hierarchy | Once-per-batch frozen snapshot, compatible previous-tree transfer or explicit skip/error | Online lifecycle, no historical raw bank | Different refresh/coverage and possible skipped batches | 15-update freeze, reuse age/distance/GT/context/no-chain and no-memory-mutation tests |
| Original implicit loader/class index alignment | Sample-keyed immutable snapshots and shared negative ID allocator | Correct alignment and stable checkpoint identity | Corrects mistaken positives/ID collisions | Sparse IDs/two-view identity, negative IDs, hash/state round-trip |
| Offline SGD, labelled subset and SelEx-only objective | Existing DEAN CE/weighted EC +gamma*shared USE/SSE, verified bounded GT replay, persistent AdamW projector | Preserve DEAN discovery and tested optimization | This is an online adaptation, not offline reproduction | Formula/GT poison/gradient/optimizer/RNG/legacy parity tests |

Phase-2 departures (local K, numerical guards, centroid bookkeeping, seed-only
memory and radii) remain documented in [the Phase-2 report](DEAN_SELEX_HSSK_LEAF_PHASE2.md).
No new loss equation or intentional bug-reproduction mode is introduced.

## CPU production-dimension probe

One CPU thread, 64 synthetic current images +32 verified-fixture replay images,
identity/rank-2-LoRA backbone proxy (NOT real DINO), production 768-2048-2048-256-
65536 projector. The synthetic DEAN-side term is CE plus 0.01*CE as weighted EC;
this probe does not claim reproduction of upstream energy discovery/loss.
The regular suite uses small projection dimensions.

| Measurement | Observed value |
|---|---|
| Levels / group counts | 5 / [32,16,8,4,2] |
| Projector trainable parameters | 23,073,024 |
| Head construction | 0.153 s |
| Leaf + higher hierarchy | 3.288 s |
| Live forward + loss + backward | 8.603 s |
| Total after imports | 12.323 s |
| Peak process RSS | 984,244,224 bytes = 938.648 MiB |
| Peak Windows process commit | 1,706,078,208 bytes |
| Guarded tensor operations | 88,103 |
| Largest 2-D tensor | [65536,256] weight/gradient |
| Largest projected batch | [128,65536]; replay [64,65536] separately |
| Largest contrastive pairwise matrix | [128,128] |
| LoRA-B gradient absolute sum | 0.026202254, finite/nonzero |

CE=3.4670894, weighted EC=0.0346709, USE=4.9427161, GT SSE=1.0988226,
hierarchy SSE=4.2323060, SSE=2.6655643, SelEx=4.1457129, final=3.9163315.
All trainable projector gradients were finite; hierarchy snapshot hash stayed
fixed. Dispatch guards rejected feature-square requests and traced all outputs;
no 65536x65536 allocation occurred. Local probe artifacts are ignored `.tmp`
files. No AdamW step/moments are included in this production-dimension memory
probe; the small-dimension worker fixture does perform optimizer steps.

RSS includes imports and CPU allocator caches, not GPU VRAM. Real DINO graphs,
AdamW moments (~176 MiB for the projector), effective weight-norm buffers and
classifier-expansion state copies remain GPU memory risks to profile later.
No GPU memory/throughput claim is made.

## Final verification and exact scope

The complete CPU suite passed **354/354 tests**, no skips, in **157.457 seconds**:
333 retained tests (including the four advanced scaffold-gate assertions) and
21 new Phase-3 tests. The final pre-commit protocol/telemetry revision passed the
same **354/354 tests**, no skips, in **156.931 seconds**. Actual **CPython 3.9.13** compiled all **16 changed Python
files**. Runtime tests used the existing Python 3.11 environment; compilation
does not claim a dependency-bearing runtime test on Python 3.9. Ordinary
`git diff --check` and separate whitespace checks of new files passed.

The implementation adds/changes exactly these 19 files:

- `dean_integration/selex_hssk_hierarchy.py`
- `dean_integration/selex_hssk_provider.py`
- `dean_integration/selex_hssk_runtime.py`
- `dean_integration/selex_replay_manifest.py`
- `dean_integration/selex_smoke_gate.py`
- `dean_integration/selex_hierarchy_provider.py`
- `dean_integration/selex_real_online.py`
- `dean_integration/selex_config.py`
- `dean_integration/runtime.py`
- `dean_integration/external.py`
- `dean_integration/provenance.py`
- `scripts/build_dean_replay_manifest.py`
- `scripts/run_dean.py`
- `tests/test_dean_hssk_phase3.py`
- `tests/test_dean_hierarchy_provider.py`
- `tests/test_dean_hssk_online.py`
- `docs/DEAN_SELEX_HSSK_PHASE3.md`
- `docs/DEAN_SELEX_REFERENCE_V1.md`
- `README.md`

The two locked design documents, Phase-2 numerical leaf fitter, projector,
USE/SSE kernels, original EMA, frozen reference YAML, datasets, checkpoints and
external upstream source remain unchanged. No architecture/loss decision was
changed. Full-trainer checkpoint/readiness and fair replay-matched production
experiments remain Phase 4 work. This records the pre-commit checks for the single
reviewed Phase-3 commit. No later commit, Phase-4 work or local GPU run is part of
this preparation; no merge/GPU execution has occurred.
