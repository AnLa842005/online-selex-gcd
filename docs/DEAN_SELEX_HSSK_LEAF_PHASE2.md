# Phase 2: standalone HSSK leaf fitting and independent memory

Status: leaf-only CPU numerical engine, **not a full hierarchy**. The production
provider remains `scaffold/incomplete` and raises `HSSKImplementationIncomplete`
before training. No higher levels, center coarsening, whole-tree reuse,
USE/SSE production wiring, GPU training or full-trainer resume is implemented.

The authoritative architecture remains
[the design contract](DEAN_SELEX_DESIGN_CONTRACT.md) and
[the realignment report](DEAN_SELEX_ARCHITECTURE_REALIGNMENT.md).
No locked algorithmic decision is changed by Phase 2.

Phase 1 was committed/pushed as
`7be10fd6d82591cc2f356016d17a3ac932e7917a` after validating the exact nine-file
scope and unchanged sources from its 301-test run. The protected DEAN branch
remains at `d7ba08745a8043a87d3126da8366a4720d81951f`.
Phase 2 implementation is intentionally uncommitted.

## API and observation boundary

`HSSKLeafFitter.fit(current_features, sample_ids, *, replay, memory, seed, batch_id)`
accepts only real current embeddings and stable IDs, ephemeral verified replay
features, a pre-batch `HSSKPrototypeMemory` and private seed/batch metadata.
There is no catch-all kwargs path for DEAN predictions, classifiers, energy
masks, AP/GMM labels, VFA features, current diagnostic GT or the prototype EMA.

The future integration boundary must produce deterministic eval/no_grad DINO
embeddings from current images and eligible replay. A numerical feature-only API
cannot prove how a caller ran a model; it has no model/image/dataset handle.
Here the engine detaches, copies to CPU FP32, validates finite `[N,768]` tensors,
normalizes along the feature dimension, and canonicalizes real rows by sample ID.
Global duplicate IDs, including current/replay overlap, are errors. Zero feature
vectors remain finite with the normalization epsilon; they do not create extra
distinct geometry or additional real support.

`VerifiedReplayFeatures.from_selection` requires the Phase-1 replay manager's
frozen selection. It revalidates manifest membership, exact IDs, and two distinct
references per selected GT class; it obtains GT from verified entries rather
than a caller-provided label tensor. It holds only this batch's detached features.
Those features never enter serialized replay or prototype-memory state.
The actual seed-1 eligibility manifest still needs independent preflight
verification; synthetic test manifests are not evidence for the real CUB split.

## Actual leaf algorithm

Let N count distinct real current+selected-replay images; A count selected
verified replay classes with exactly two references each; m=2. The code uses:

```text
K_free in [0, min(16, floor(N/2)-A)]
K_free=0 allowed only if A>0
K = A + K_free
max(A,1) <= K <= min(A+16, floor(N/2))

score = mean_real_rows(||normalized_feature - assigned_final_center||^2)
        + 0.02*K_free
```

1. GT-class means initialize A anchor centers. Hard replay assignments remain
   fixed throughout both phases. Current rows are always unlabelled. Equal
   centers never collapse two verified classes into one identity.
2. For each feasible K, one private CPU `torch.Generator` drives k-means++.
   The seed is a stable digest of the leaf algorithm domain/version, supplied
   private seed, incoming batch, K and config identity. It consumes no global
   Python, NumPy, Torch CPU/CUDA or paired-stream generator state. Candidates
   are current rows and at most 128 geometrically relevant memory centers;
   chosen seed locations within epsilon are excluded from subsequent draws.
3. Balanced fitting runs at most 10 iterations. Replay anchor centers stay at
   their initial means in this phase. Capacity is `ceil(N/K)`, at least the
   fixed replay count. Within an overfull cluster, keep closest current rows
   (stable sample-ID ties); reassign excess rows in sample-ID order to the nearest
   remaining-capacity cluster, with local seed-order ties. Never move GT rows.
4. Refinement runs at most 10 unconstrained iterations, still fixing replay
   assignments. Every center, including an anchor center, becomes the mean of
   its assigned REAL rows. Centroids are not renormalized into spherical K-means.
   Both phases use squared sum-of-center-shifts <1e-4 as the stopping test.
5. An empty free center can be reseeded only from a highest-residual real current
   point with a feasible donor. Two distinct current rows from that donor provide
   initial support; the donor must retain two real rows and, if free, two current
   rows. The seed itself is a real current feature; subsequent centroid updates
   use assigned real rows. Otherwise reject the candidate. This operation is
   local to the candidate and never reads or reallocates persistent track IDs.
6. Recheck every leaf's two-real-image support after refinement, plus two-current
   support for free leaves. Score only accepted final labels/means. Select by
   score, smaller K, then stable seed order. There is no oracle novel count.
7. If no candidate survives, return typed `INSUFFICIENT_SUPPORT` with reasons and
   unchanged memory hashes. This is not tree reuse, a valid full hierarchy or
   permission to run a prototype fallback. Production dispatch remains blocked.

`LeafFitResult` is an immutable canonical value with copy-out tensors/maps. It
contains status, N/A/K/K_free, sample-to-negative-leaf IDs (current and replay),
centers, real/current counts, anchored/free and GT provenance metadata, score,
phase convergence/iteration/seed diagnostics, rejected-K reasons, counters,
memory-before/after hashes, and deterministic fit hash. Timings do not enter
deterministic hashes. It deliberately has no ancestor tree.

Leaf radii are the mean real-row distance to the final center, epsilon-clamped
with a counter. The original source's half-intercenter initialization/frozen
radius is not silently claimed here. No USE/SSE interpretation of these leaf
radii is enabled until the reviewed higher-hierarchy integration exists.

## Memory ownership and transaction

`HSSKPrototypeMemory` owns at most 512 detached normalized 768-D track centers.
Tracks contain cumulative observed support (not historical distinct-ID counts),
current radius, creation/last-active batch, update count, verified-anchor metadata,
manifest/dataset/split provenance fingerprints and one bounded lineage hash.
There are no raw images, paths, sample IDs, sample-feature histories or graphs.

Memory supplies only geometry-ranked seeds; it never contributes fitting rows,
N, capacity, support, score, GT or SSE examples. No memory label becomes a hard
anchor in the fitter. Current selected replay alone creates hard anchors.

After selecting a valid candidate, create a proposed memory copy. Match current
leaf centers against old normalized centers by globally sorted geometric edges,
then track ID and local seed ID. Matching is one-to-one; conflicting verified GT
identities do not exchange anchored tracks. This constraint affects persistent
track identity, not clustering assignments or K. Free current leaves have no
inherited GT anchor constraint from memory.

Matched tracks store `normalize(0.9*old_center + 0.1*new_real_mean)`. New tracks
store normalized real means and receive monotonically decreasing negative IDs.
Evict least-recent inactive tracks with stable-ID ties; an active leaf cannot be
evicted. Only after result and proposed state validation publish the copy to
the caller's memory. Failed/rejected fits and duplicate-success batch errors
leave centers, ages, counts and ID allocator unchanged. A successful batch
cannot update memory twice.

Memory state has its own provider/schema/config/hash validation, disjoint from
the legacy EMA. Restoring it reproduces the next leaf fit/hash/allocator when
given the same inputs, seed and batch. This is **not** full trainer resume and
does not serialize models, optimizers, stream/RNG domains or frozen trees.

## Reference departure register for Phase-2 details

Reference is SelEx `c3d3b42a0aa7d7c9ea748a8305a6890a34100f73`.
BSSK refers to `methods/clustering/faster_mix_k_means_pytorch.py:174-303`;
trainer refers to `methods/contrastive_training/contrastive_training.py`.
These rows specialize the online departures already authorized by the contract;
they do not modify the locked architecture or introduce a literal-bug mode.

| Reference behavior | Adapted behavior | Reason | Scientific impact | Test / ablation |
|---|---|---|---|---|
| Whole-dataset known K, trainer 535-555 | Enumerated supported K and final real-row error +0.02 per free cluster | Bounded online support without oracle counts | Recovered cluster count can differ | Exhaustive N/A bounds; support properties; penalty/cap sensitivity in later research |
| BSSK known anchors and capacity-repair loops, 174-268 | Fixed true replay anchors; deterministic explicit capacity repair; real high-residual empty repair and final support gate | Fix empty/ordering behavior at batch 64 | Different partitions when repair/support gates bind | Original AST nondegenerate fixture; anchor, overflow/reseed, singleton and final-support tests |
| BSSK tracks best inertia before centroid update, 232-303; strict improvement can preserve stale balanced centers on early refinement convergence | Return the final refined labels/means, compute their true real-row inertia, then check support | Contract explicitly requires refined real means and final support/score | Result can differ from literal stale output when stopping early | Final anchor-mean test; source parity fixture avoids stale-bookkeeping defect; later fixed-K sensitivity study |
| Standard offline k-means++ / implicit RNG | Private versioned generator, canonical real IDs and memory seed proposals | Isolate stream and DEAN randomness | Different initialization distribution with memory enabled | Same-seed/hash and global-RNG invariants; memory-off/on ablation |
| No online persistent seed tracks | Independent bounded HSSK EMA; seed-only; one-to-one deterministic matching with GT provenance consistency | Temporal evidence and stable IDs without raw history | Initialization bias and possible representation drift | Matching/momentum, eviction, capacity, resume and no-history tests; memory-free ablation |
| Initial radius is half nearest-center distance, trainer 254-263; source default m=1 freezes it | Store epsilon-clamped mean distance over assigned real rows for leaf summaries | Contract uses real support at leaves; no unsupported fake neighbors | Radius scale differs and would alter later USE geometry | Radius/support invariants; future radius ablation required before attributing USE gains |
| CUDA-specific BSSK and unguarded data cases | Detached normalized CPU FP32 fit from FP16/BF16/FP32 input, typed insufficient result, structural errors hard | CPU verification and finite online behavior | Online fit intentionally differs from literal error-prone branches | Mixed-dtype, bad-input and no-mutation tests; no GPU accuracy/performance claim |

## CPU probe and verification

Probe uses synthetic normalized-direction features plus private-generator noise,
64 current images and 32 verified-fixture replay references (16 classes), seed
17 and one Torch CPU thread. Cold and bounded-memory cases run in one isolated
Windows process, in that order. No DINO/projector forward, loss, backpropagation,
data loading or GPU work is included. Windows process peak RSS includes Torch
imports and is cumulative; it is not an allocation-only or GPU VRAM estimate.

| Case | Real N / selected K | Memory tracks / selected seeds | Fit wall / CPU seconds | Process peak RSS | Increase over prior process peak |
|---|---|---|---|---|---|
| Cold | 96 / 32 | 32 after fit / 0 | 3.272 / 2.906 | 345.445 MiB | 8.738 MiB |
| Preloaded full bounded memory | 96 / 32 | 512 / 128 | 10.632 / 9.875 | 398.824 MiB | 22.648 MiB |

Both results were SUCCESS with 16 anchored and 16 free leaves. Serialized memory
was 368,678 / 2,412,039 bytes respectively, containing bounded summaries only.
Fit timings include validation, hashing and atomic memory publication; they are
single observations, not a throughput guarantee. Local ignored probe artifacts
are `dean_hssk_phase2_probe.tmp` and `dean_hssk_phase2_cpu64.tmp`.

The targeted leaf/memory suite passed 32 tests in 9.494 seconds. It includes
1,921 N/A bound cases (N=0..128, A=0..min(16,floor(N/2))), 21 fitted support
combinations, a batch-64 case, original-source fixture, state corruption and
metadata-only serialization inspection. Final gates passed: **333/333 CPU tests**
(301 existing +32 new), no skips, in **247.278 seconds**; both Phase-2 Python
files compiled with actual **CPython 3.9.13**; `git diff --check` plus independent
whitespace checks of new files passed. The dependency-bearing CPU suite ran in
the existing Python 3.11 environment, not the dependency-free Python 3.9 compiler.

```powershell
$env:DEAN_UPSTREAM_CHECKOUT = (Resolve-Path ../OCGCD-dean-audit).Path
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Existing projector, loss kernels, original EMA, production provider/runner,
official configs and Phase-1 implementation remain unchanged. Phase 2 adds only
the leaf module, its tests and this explanation. Stop for review before Phase 3;
no Phase-2 commit/push, GPU training or merge is performed.
