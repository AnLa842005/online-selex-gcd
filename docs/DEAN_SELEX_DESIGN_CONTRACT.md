# Online HSSK + USE/SSE design contract

Status: **LOCKED v1 design; implementation NOT authorized in this step**.
This contract records the reviewed architecture and supersedes conflicting
proposals in the first architecture realignment report and older prototype-v1
documents. It does not claim that HSSK or resumable HSSK state already exists.
Branch: `experiment/dean-selex-hssk-online`; code baseline: `d7ba087`.
Reference: SelEx `c3d3b42a0aa7d7c9ea748a8305a6890a34100f73`.

## 1. Immutable architecture and provider identity

```text
DINO/LoRA embedding --detached deterministic eval features--> Online HSSK hierarchy
DINO/LoRA embedding --two fresh live views--> projector --> USE + SSE
                                                    ^ fixed HSSK assignments
DEAN CE + weighted EC + gamma * SelEx <---------------+
```

The hierarchy branch supplies detached targets; differentiable live views reach
LoRA and the shared projector. HSSK itself is not in the autograd graph.

1. `hssk_online` / **Online HSSK + USE/SSE** is the primary hierarchy provider
   and the intended default of the NEW active research configuration. It is not
   conditional on a future default-switch decision. GPU execution remains gated.
2. `dean_prototypes` / **SelEx-inspired prototype hierarchy** is an explicitly
   selected ablation, not the primary/default and never an automatic fallback.
   Its existing tested arithmetic and RNG progression must remain identical.
3. HSSK cannot write DEAN pseudo-labels, classifier IDs/weights, energy-discovery
   state, AP/GMM outputs or VFA state. DEAN discovery may run first; its labels
   do not determine HSSK K, feature filtering, assignments or memory updates.
4. HSSK owns a bounded, detached 768-D prototype memory with independent HSSK
   identities. It must not read/share/update `dean_prototypes`' `EMAHierarchy`.
5. GT-supervised SSE accepts only provenance-verified **ground-truth** labels
   from eligible labelled replay. DEAN/HSSK assignments are hierarchy targets,
   never GT. Current-stream `gt_y` diagnostics are unavailable to this path.
6. All provider assignments are returned by stable `sample_id`. Both views of
   the same sample use the same assignment at every level; row order is not ID.
7. Exactly one build/reuse/skip decision occurs before the incoming batch's
   15 optimization epochs. Its snapshot, replay selection, centers/radii,
   mappings and hierarchy hash are frozen throughout those epochs.
8. Hierarchy features use deterministic eval transforms and `model.eval()` under
   `torch.no_grad()`, in FP32 with feature-axis normalization and epsilon guards.
   All model training flags are restored. Random USE/SSE views cannot affect
   hierarchy features, fitting seeds or assignments at the same model state.
9. Higher levels coarsen HSSK leaf centers, not current raw rows again and not
   DEAN pseudo-anchors. Each sample inherits its leaf's ancestor at each level.
10. Degenerate support uses only explicit HSSK reuse, skip or error policies.
    There is no switch to the prototype ablation or fabricated supervision.
11. Every departure from the pinned SelEx reference requires a registered row
    with **reference behavior, adapted behavior, reason, scientific impact,
    and test/ablation**. No known-bug reproduction mode is added to production.

The provider returns a typed `BUILD`, `REUSE`, `SKIP` or error result, provider
identity, immutable sample-keyed snapshot if available, support diagnostics,
feature/config versions and state hash. A skipped result is not an empty valid
hierarchy silently passed to another provider.

## 2. Stable observations, replay and memory

The real support pool is the current batch plus a bounded selected set of true
labelled replay images. Memory is a third input source, but **not a collection
of extra fitting rows**. Distinct-image support counts exclude both additional
views and memory centers. Images are identified by dataset fingerprint, train
split and `uq_idx`/canonical path; train/test IDs are not globally interchangeable.

Replay is separate from official DEAN `--n_replay 5`, which generates Gaussian
VFA features (audited source evidence in the realignment report). The new manager
stores at most 320 eligible base-labelled references, at most 2/class, with a
replay batch budget of 32 references. Selection is by **class-complete groups**:
each selected class supplies exactly 2 distinct verified references, so at most
16 classes are anchored per incoming batch. A class with fewer than 2 eligible
distinct references in the buffer cannot be selected for GT supervision or as
a GT anchor. Never use a current sample, a second view, a duplicate reference
or a memory center to complete its pair. If fewer than 16 complete classes are
available, select all available complete groups without padding or repetition;
the actual selected reference count is even and may be less than 32.

Canonicalize class IDs and sample IDs before admission AND selection; deduplicate
by verified sample identity before counting support. Admission/eviction is
deterministic and class-balanced: evict from the most overrepresented class,
then oldest admission, then class ID/sample ID. Process admissions in canonical
order, not DataLoader arrival order. No pixels, PIL objects or historical
per-image features survive the batch.

Select complete classes by deterministic round-robin over canonical class IDs,
starting at a saved class cursor. Scan at most one class cycle, skip ineligible
classes, select at most 16 distinct classes, and take each selected class's two
references in canonical sample-ID order. Advance the cursor to the canonical
successor of the last inspected class exactly once per incoming batch; if the
buffer is empty, preserve the cursor. Represent the cursor as a class-ID ordering
key, not a loader position; if that class was evicted, start at the first class
ID greater than or equal to the key, wrapping if needed. Selection never depends
on DataLoader iteration order and draws no random numbers.

Freeze the selected IDs for all 15 epochs. Checkpoint the pre/post-selection
class cursor, selected class/sample IDs and selection hash; mid-batch resume
restores the frozen selection without selecting again or advancing the cursor.
Next-batch resume starts from the saved post-selection cursor. Define the hash
as SHA-256 of versioned canonical serialization containing the eligibility
manifest hash, incoming batch ID, pre/post cursors and selected class IDs with
their sorted sample-ID pairs. Persist its serialization version and recompute
the hash on restore. These are fixed contract budgets, not optional values from
older prompts or claims of optimal scientific settings.

Eligible references must come from the original authorized `train_labelled`
split, not all base-class images. Validate split membership, label map, dataset
and manifest hashes before admission. Missing requested/corrupt replay is an
error; replay explicitly disabled/unavailable records `sse.gt.unavailable` with
reason and omits only GT if hierarchy fitting remains supported. Do not infer
eligibility from `gt_y`, `.ys` after relabelling, or numerical label range.
Do not retrain the base model to reconstruct a manifest. Original split metadata
must be established and hashed before a supervised experiment can be called valid.

HSSK memory holds at most 512 detached centers, support counts, radius/age,
anchor provenance, stable HSSK track IDs and hierarchy lineage. Select at most
128 geometrically relevant memory seeds with deterministic ties. It is not keyed
by DEAN labels. In v1 memory is **seed-only** (anchor weight zero after seeding):
no memory pseudo-row, GT label, positive pair, cluster-size support, SSE target
or additive inertia contribution. A weighted-prior-anchor variant requires a
separate config, departure entry and ablation; it is not an implicit v1 feature.
Current GT replay alone supplies hard label anchors.

Fit reads pre-batch HSSK memory. On a successful fresh fit, match new leaf centers
to prior HSSK tracks by geometry with deterministic ties, update weighted means
with configured momentum 0.9 once under no_grad, then evict least-recent inactive
tracks with stable-ID ties. Allocate monotonic negative IDs for new leaf/ancestor
groups, disjoint from nonnegative DEAN IDs. A fit may not exceed capacity or
silently remove an active leaf. Reused/skipped batches do not update memory,
ages or ID allocators as if a new fit succeeded; log their observation separately.
No raw historical sample bank is retained.

## 3. Locked fitting and coarsening policy

1. After DEAN discovery/VFA and before optimization, read current images by ID
   through the filepath-only adapter. HSSK uses **all valid current image rows**,
   independent of DEAN label validity/known/seen/unseen masks. Nonfinite features,
   duplicate IDs and structural mismatches are hard errors. Canonicalize fitting
   order by sample ID and map outputs back explicitly.
2. Select eligible replay once and recompute its deterministic eval embeddings
   with the same current backbone snapshot. GT-class means initialize anchored
   centers; replay assignments stay fixed. Current points remain unlabelled.
3. Add free seeds using private deterministic k-means++ with memory candidates
   and current geometry. Deduplicate seed locations; distinct GT classes are
   never merged merely because initial centers coincide.
4. Choose K only from feasible support. Let N be distinct CURRENT+REPLAY real
   image count, A the number of selected replay classes each providing exactly
   2 distinct verified references, and m=2 the minimum real samples per leaf.
   Thus `A <= 16` and selected replay count is `2*A <= 32`. A class with only
   one reference contributes neither an anchor nor a replay row to N. Current
   samples, views and memory cannot complete a replay pair or increase A.
   Require `K <= min(16+A, floor(N/m))`, `K >= max(A,1)`.
   Enumerate free counts from 0 to min(16, floor(N/m)-A); zero is allowed only
   with anchors. Every retained leaf, including an anchored leaf, must have at
   least m distinct real images after refinement. Each free leaf also requires
   at least m current images. No phantom 160 known centers or oracle novel K.
   If GT anchors make these constraints infeasible, take the support policy in
   section 4; do not merge true labels or count memory as missing observations.
5. Run at most 10 balanced semi-supervised iterations followed by at most 10
   unconstrained refinement iterations, tolerance 1e-4, one deterministic
   initialization per K. Labelled row assignments remain fixed in both phases.
   In the balanced phase GT centers stay at their replay means; final refinement
   updates centers from assigned REAL rows, as the reference does. Capacity is
   ceil(N/K), adjusted to fixed anchor counts. Deterministic excess reassignment
   must respect labels. An empty free center may be reseeded from a real current
   high-residual point only if support remains feasible; otherwise reject K.
6. Among feasible candidates minimize mean squared REAL-row fitting distance
   plus `0.02 * K_free`, with smaller K then stable seed order breaking ties.
   This versioned online K-selection heuristic is an explicit scientific
   adaptation, not the original reference's known-K algorithm. A candidate is
   invalid if min-support fails AFTER final refinement; final unconstrained
   refinement does not waive support checks. No feasible K triggers section 4.
7. Build higher levels **on leaf centers**. Use leaf real-support counts as
   weights, not duplicated raw rows. At each level halve the number of anchored
   groups and free groups separately (ceil division, minimum one per present
   side), mapping anchored groups into deterministic GT-derived hypergroups.
   Run balanced semi-supervised coarsening of child centers, with anchored child
   hyperlabels fixed; free children may join an anchored or free parent. Require
   at least two distinct children per free parent and at least m real-image
   support per parent; stop when a feasible strict reduction is unavailable.
   If ceil-halving cannot strictly reduce a side, stop at the available depth.
   Known hypergroups may contain multiple true classes; those hyperlabels remain
   hierarchy-only. Never feed coarse labels to GT-SSE.
8. Recursively compose leaf-to-parent mappings to sample IDs, giving a strictly
   nested tree. No HSSK-pseudo-labelled current points become higher-level GT
   anchors. Stop before any projection-prefix width would become zero.
   Means of normalized rows are Euclidean centroids; do not silently renormalize
   every Lloyd centroid into a different spherical clustering algorithm.
9. Freeze the accepted snapshot. Radii use current/replay real support at leaves
   and count-weighted child geometry at coarser levels, with epsilon guards and
   explicit unsupported-geometry counters. Memory is only normalized when stored
   as a future seed. All targets/geometry are detached.

## 4. Insufficient support, reuse and counters

`insufficient_support_policy: reuse_then_skip` is the primary setting;
`reuse_then_error` is the explicit strict alternative. There is no `dean` option.
Malformed input, nonfinite tensors, corrupt state or invalid configuration are
always errors, not insufficient support.

Reuse means transfer the previous **HSSK centers and ancestor tree** to new IDs,
not pretend old sample assignments cover new samples. Reuse is allowed only if:

- there is a prior successful HSSK snapshot from this uninterrupted model lineage,
  same feature dimension/normalization/config/schema/dataset and seed protocol;
- it is at most one incoming batch old, never chains reused snapshots indefinitely;
- every selected GT class is represented by its verified anchored track, and
  current GT anchor rows can be assigned without conflicting class identity;
- deterministic nearest-leaf transfer covers every current sample and each
  normalized feature is within configured Euclidean distance 0.5 of its leaf;
- resulting leaf support meets the same m=2 real-image rule, including replay;
  relevant parent/radius/positive-support checks pass. A singleton cannot gain
  support from a memory center or its second view.

Reuse keeps prior tree/centers fixed, computes sample assignments/validity once,
and marks source batch and age. Incompatible model/schema/config/label maps or
insufficient transfer support prohibit reuse. Geometry threshold is a documented
heuristic to ablate, not a guarantee against representation drift.

If reuse fails, skip the **entire SelEx auxiliary contribution for this batch**
with differentiable finite zero, preserving the DEAN loss and recording exact
reason; strict policy raises before optimization. With an accepted hierarchy,
individual invalid USE/SSE terms still skip individually and missing GT omits
only GT. These are different counters/events, not interchangeable fallbacks.
Record `hssk.build`, `hssk.reuse`, `hssk.reuse_rejected.<reason>`,
`hssk.insufficient_support.<reason>`, `selex.batch_skipped`,
`sse.gt.unavailable.<reason>`, per-term singleton/empty/no-positive counters,
provider identity, source batch, support/K and snapshot/memory hashes. Emit
zero-valued component logs for a batch skip and indicate why GT was also skipped.

For every incoming-batch selection, log these exact replay telemetry keys:

- `replay.class_ineligible.insufficient_distinct_samples`: counter of buffered
  classes with fewer than 2 distinct eligible references, assessed once per batch
  over the canonical buffer inventory, not once per loader row or epoch.
- `replay.class_selected`: selected complete class count A (at most 16).
- `replay.reference_selected`: distinct selected reference count 2*A (at most 32).
- `replay.selection_hash`: SHA-256 identity of the frozen canonical selection;
  this is a string metadata field, not an arithmetic counter.

Log per-batch counts and cumulative numeric counters with checkpointed update
cursors so resume cannot double-count a selection. A single-reference class is
ineligible even when the incoming current batch contains another point of that
class or two augmentations can be generated from its one replay image.

## 5. Loss, gradients, RNG and unchanged boundaries

```text
H = sum_l hierarchy_decay^(l+1) * SupCon(projected_prefix_l, HSSK_labels_l)
GT = SupCon(full_projected_replay, verified_true_labels)  # zero if unavailable
SSE = (GT + H)/2
SelEx = (1-sse_weight)*USE + sse_weight*SSE
L_online = L_CE + L_EC_weighted + gamma*SelEx
```

Preserve shared production projector dimension 65536, internal kernels,
temperatures, smoothing, level weights and default gamma 0.1; small dimensions
remain explicit tests/ablations. Current two-view samples participate in USE and
hierarchy SSE regardless of DEAN known/seen/unseen classification. Replay provides
hard fit anchors and the separate GT term, not extra current USE/SSE rows in v1.
No pseudo-label is disguised as GT when replay is absent.

Normalize projected FULL vectors along features, slice positive-width prefixes
as views without prefix renormalization, use stable per-row confusion/epsilon
normalization and existing finite-zero counters. Pairwise matrices are sample by
sample; no 65536 by 65536 allocation. Model/projector gradients come from live
view losses only; hierarchy/replay feature extraction and memory have no graph.
The projector AdamW group persists immediately before the final expandable
classifier group, preserving optimizer state across classifier expansion.

Distinct deterministic domains cover HSSK fits, current-view augmentation and
replay-view augmentation. Replay selection uses its canonical round-robin cursor,
not an RNG or a loader generator. Stochastic-domain seeds use a stable hash of online
seed, algorithm version, stream batch, level/K/restart and domain; never Python
hash or shared global RNG. Save any advancing generator states/counters. Changing
augmentation seed cannot change the same-model hierarchy; changing HSSK seed
cannot reorder the paired stream or consume DEAN RNG. Gamma=0 or disabled SelEx
bypasses all provider/replay/projector/image/RNG work and preserves official DEAN.

## 6. State lifecycle and deterministic resume

This table is a future checkpoint REQUIREMENT, not existing resume support.

| State | Lifetime / owner | Build, freeze, replacement | Checkpoint / deterministic restore |
|---|---|---|---|
| Current IDs, paths, detached eval rows, selected replay eval rows | One incoming batch | Extract once before epoch 0; discard features at batch completion; never append history | Persist IDs/replay selection and input/feature fingerprints, not raw pixels; reconstruct from verified dataset only if needed |
| HSSK sample assignments, centers/radii, ancestor tree and status | One batch, provider snapshot | Build/reuse once, immutable for 15 epochs; skip status is also frozen | Save exact detached snapshot tensors/mappings/hash, source batch, support, frozen skip/reuse decision and epoch cursor; never refit on mid-batch resume |
| HSSK bounded prototype tracks | Across batches, HSSK-only | Read pre-batch; update once after successful build, before epochs; no update on skip/reuse | Save complete bounded centers/counts/ages/provenance/IDs, allocator and last-update batch; restore without duplicate update |
| Prototype-ablation EMA | Across batches only when explicitly selected | Original tested update/rebuild order | Separate provider-tagged schema; never load as HSSK memory |
| Labelled replay references | Across batches, replay manager | Canonical admission; bounded eviction; round-robin complete pairs, at most 16 classes/32 references; freeze selection per batch | Save references/labels/provenance/ages, pre/post class cursor, selected sample/class IDs, selection hash/version, capacity and eligibility-manifest hash; restore selection without cursor advancement; validate paths and split |
| Live two-view tensors / graphs | One optimizer update | Fresh views each update; discarded after backward | Do not serialize graphs/pixels; save RNG state and next-update cursor |
| Projector, LoRA, dynamic DEAN classifier, teacher, optimizer and scheduler/scaler | Run | Original DEAN lifecycle plus persistent projector group | Full state_dicts, parameter-group identity/order, teacher and discovery/session state; base checkpoint alone is insufficient |
| Paired stream | Run, independent stream owner | Exact manifest/cursor; unaffected by fit or views | Save manifest hash/content reference, session/batch cursor, sampler/generator state and any needed within-batch official minibatch order |
| RNG domains / counters | Run | Independent HSSK/replay/view domains, official Python/NumPy/CPU/CUDA preserved | Save official and private RNG states, seed derivation version, counters, online seed and deterministic backend/device/runtime metadata |
| Experiment identity / metrics | Run | Immutable effective config, versions, hashes | Save upstream/SelEx/code SHA, provider/schema/config/replay/dataset/base hashes; journal committed log offset and resume event |

Checkpoints occur atomically at safe boundaries AFTER an optimizer step and
zero-grad bookkeeping, before generating the next views, or at a completed
batch boundary. No partial gradient accumulation is supported. If a crash occurs
while building hierarchy, resume from the previous complete checkpoint; do not
partially replay memory mutations. Mid-batch checkpoints include whether the
one-time memory update already happened and exact fixed hierarchy/GT selection.
Restore model/optimizer and all states before any RNG-consuming action. Do not
initialize a new random projector over restored weights or rebuild the hierarchy
with the already-updated LoRA model. Restore the original official `on_x` transform
realization by replaying its saved pre-transform RNG/input order then restoring
post-transform RNG; a mere eval reread would change DEAN's fixed training tensor.
If faithful reconstruction is unsupported, fail mid-batch resume explicitly;
batch-boundary-only support must be declared, never approximated silently.

Bitwise continuation is required on the same deterministic runtime/device stack.
Cross-device/runtime resume validates semantics and reports that bitwise equality
is not guaranteed; it must not claim exact replay. An initial paired run without
a historical base RNG sidecar is distinct from a saved online resume checkpoint.
Strict historical replay still requires its sidecar; neither protocol substitutes
for the other.

## 7. Departure register from pinned SelEx

Reference source locations are at immutable commit c3d3b42; trainer abbreviates
`methods/contrastive_training/contrastive_training.py`, BSSK abbreviates
`methods/clustering/faster_mix_k_means_pytorch.py`. Exact equations and loss source
locations remain in the historical integration audit. This register covers the
new online design and retained corrected-kernel departures; future changes need
new rows before implementation.

| Reference behavior / source | Adapted behavior | Reason | Scientific impact | Required test / ablation |
|---|---|---|---|---|
| Trainer 502-555 fits whole offline dataset with known class K | Current batch + eligible replay; bounded seed-only HSSK memory; support-bounded local penalized K | Stream cannot access whole past/future dataset or true novel count | Online cluster count can differ from offline HSSK; not a literal reproduction | Fixed-support K bounds; penalty/cap/min-size sweep; memory off/on |
| Trainer 558-590 clusters known prototypes then re-fits sample features at each higher level | Weighted HSSK coarsening only on leaf/child centers; nested sample inheritance; no DEAN pseudo-anchor | Locked online architecture, bounded cost and stable assignments | Loses coarse raw-feature reassignment; strictly nested hierarchy changes targets | Parent composition; coarsening-only vs reference-style offline fixture ablation |
| No EMA seed history in offline HSSK | Independent bounded HSSK-track memory, no DEAN EMA | Retain bounded temporal evidence | Temporal bias/representation drift possible | Memory capacity/eviction/seed influence; no raw bank; memory-free ablation |
| Known centers/labels and BSSK phases, BSSK 174-303 | Same anchor role and two-phase fit, deterministic capacity repair, support gates and guarded empty centers | Batch 64 and degenerate support | Some offline-accepted partitions rejected | Nondegenerate BSSK parity; hard anchors; final support constraints; phase parity |
| Trainer 507-515 uses first augmented training view for extraction | Deterministic eval-view embeddings in eval/no_grad | Stable hierarchy independent of loss augmentation | Less augmentation noise in discovery | Alter view RNG only and compare identical hierarchy |
| Offline full labelled subset for GT SSE | Bounded verified base-labelled replay; complete two-reference class groups via canonical round-robin; unavailable GT omitted | No oracle stream labels; bounded resources; no singleton GT anchor | Less supervised/class coverage; replay adds GT vs historical prototype control | Poison GT provenance; singleton exclusion; order-independent selection/hash and resume cursor; replay-matched prototype control |
| Periodic full hierarchy at epoch boundary, historical audit E8 | One snapshot per incoming batch, fixed for 15 updates | Online lifecycle | Different refresh cadence and model drift within batch | Build-call count and immutable hash over all 15 epochs |
| Whole-dataset support normally assumed | Compatible previous-HSSK transfer then explicit batch skip/error | Do not fabricate supported clusters or change provider | Some batches contribute no auxiliary gradient; selection bias must be logged | Empty/singleton/reuse-incompatible; coverage and skip-rate ablation |
| Reordered labelled/unlabelled outputs consumed as loader order, trainer 566-590 | Explicit sample-key mappings and disjoint global group IDs | Correct alignment and avoid collisions | Corrects erroneous positive/target pairs | Sparse/permuted IDs, global uniqueness, both views share mapping |
| Global/optional cosine norm and unnormalized live geometry, historical audit E1/E3 | Feature-axis normalized 768-D geometry; Euclidean-only production | Correct geometry and consistent scales | Intentional numerical difference | Nondegenerate corrected-reference fixtures; geometry-scale ablation |
| Broadcast confusion row-sum division and unguarded denominators, E4 | Stable row normalization, epsilon, non-self normalization and stable logsumexp | Finite mixed-precision arithmetic | Changes asymmetric target weights; fixes bugs | Asymmetric confusion/zero-mass/extreme logits/AMP tests |
| Single semantic sample has other view as positive, E6 | Require distinct-real-image support; differentiable zero per invalid term | Prevent spurious semantic evidence | Fewer positive anchors than literal reference | Singleton-vs-two-real-sample tests; skip counters |
| Hard-coded CUDA/unguarded empty BSSK groups | Device-safe deterministic fit, finite degeneracy handling, structural errors hard | CPU tests and robust online support | No intentional bug compatibility mode | CPU/AMP/empty/nonfinite validation |
| SGD/full block tuning and SelEx-only objective, historical audit E7 | Shared DEAN LoRA/AdamW + persistent projector, CE+weighted EC+gamma*SelEx | Preserve working DEAN learner | Different optimization/objective; not offline reproduction | Gamma-zero identity, optimizer expansion state, gamma sensitivity |
| Fixed 65536-D reference head | Same production head, configurable small/768-D ablation | Test cost and research ablation | Nondefault dimensions change contrastive geometry | Small-dim tests plus separate production-head probe; dimension ablation |
| Offline training/checkpoint regime | Isolated domain RNG, paired stream and complete online-state resume | Reproducible paired comparison | Same stream alone does not make historical full-run metrics comparable | Manifest/RNG isolation and uninterrupted-vs-resumed equality |

## 8. Architecture-level acceptance tests

These are proposed tests, not tests executed by this documentation change.

1. Load the NEW primary config through the real CLI/worker factory; assert the
   constructed provider is `OnlineHSSKHierarchyProvider`. Replace prototype
   provider constructor/build with raising spies: primary, reuse and skip must
   never invoke them. Default real research selection must resolve to HSSK.
2. Explicit ablation selects only `DeanPrototypeHierarchyProvider`, matches frozen
   d7ba087 numerical outputs/gradients/counters/RNG, and cannot share memory with
   HSSK. Preserve all 273 baseline tests; update default-selection expectations
   only by adding separate primary-config tests, not weakening legacy fixtures.
3. Hold images, features, GT replay, HSSK memory/config/seeds fixed; permute DEAN
   pseudo-labels, classifier inventory and VFA diagnostics. Assert identical HSSK
   assignment/snapshot hash, selected K and updated memory, and no DEAN mutation.
4. Reorder sparse sample IDs and independently generate two different views.
   Assert both views map to exactly the same leaf/ancestor assignments; no
   positional/duplicated-view lookup changes support counts.
5. Instrument build/memory writes: exactly once before epoch 0, unchanged snapshot
   hash/centers/assignments during epochs 0..14 despite LoRA updates, one rebuild
   at the next batch; current-view RNG changes cannot alter build at fixed model.
6. Spy on the GT SupCon call: only verified replay labels/images enter it. Inject
   pseudo labels into the replay constructor/provenance layer and reject them;
   poisoned current `gt_y` must be unreachable. No replay omits only GT when a
   valid hierarchy exists. Unanchored hierarchy grouping still has no GT status.
7. Test empty/singleton/duplicate geometry/too-many anchors/final-small-cluster
   cases under both insufficient-support policies. Assert prior HSSK reuse only
   when every compatibility gate passes, otherwise explicit whole-batch zero or
   error; prototype spies never run; correct counters and finite gradients.
8. Verify higher levels read only leaf/child centers/counts/GT-derived anchors,
   not DEAN labels or current feature rows again; sample ancestors are nested;
   novel/free IDs globally unique, true anchors fixed; positive prefix widths.
9. Save after a safe step in epoch 7 and at a batch boundary. Resume on the same
   deterministic runtime and compare next views, all 15-epoch snapshots, K/IDs,
   memory/replay state, losses/gradients/model/optimizer and stream order to an
   uninterrupted run. No duplicate memory update or hierarchy build. Test crash
   before atomic checkpoint, bad schema/hash/provider and unsupported mid-batch
   reconstruction as hard errors. Compare journal offsets to avoid duplicate logs.
10. Enforce bounded replay/reference and HSSK memory sizes over many batches;
    recursively inspect saved state for historical raw images/feature banks.
    Assert eviction/admission reproducibility, provenance and class balance.
    Replay-specific acceptance fixtures must also prove:
    - One eligible replay reference never creates an anchored leaf, including
      when augmented twice or accompanied by current points/memory centers.
    - 32 selected references form exactly 16 complete pairs; smaller eligible
      pools produce only complete pairs, never more than 16 anchored classes.
    - Permuting DataLoader order before admission and selection leaves buffer,
      selected sample IDs, cursor transition and selection hash identical.
    - Save/resume restores the exact class cursor and frozen selection/hash,
      does not advance twice, and yields the same next-batch selection.
    - Two views of one sample count as one distinct real replay reference;
      duplicates cannot satisfy a complete class or inflate counters/A.
11. Disabled/gamma-zero provider constructors, replay image loads and RNG scopes
    are never called; official DEAN loss/output/stream and optimizer stay exact.
    Preserve projector group and AdamW state across classifier expansion.
12. Run complete CPU suite, Python 3.9 compilation, finite backward/AMP and
    batch-64 resource guards (no projection-dimension-squared allocation).
    GPU smoke is a separate later authorized gate; HSSK is not scientifically
    validated by changing its default or by passing architecture tests alone.

## 9. Configuration and document precedence

This contract is the source of configuration truth for the primary profile:

| Setting | Locked value / allowed values |
|---|---|
| Replay buffer capacity | 320 references |
| Replay references per class | At most 2 stored; exactly 2 distinct verified references for each selected class |
| Selected replay batch budget | 32 references, at most 16 complete classes; no padding |
| Selection | Canonical deterministic class round-robin with checkpointed cursor |
| `insufficient_support_policy` | `reuse_then_skip` (primary) or `reuse_then_error` |

Do not use the obsolete `selex_labeled_replay_per_class=5` or
`hssk_degenerate_policy` from older prompts as configuration or aliases.
Official DEAN `--n_replay 5` remains the unrelated VFA parameter.

A NEW primary HSSK config explicitly records source, K/support policy, replay
eligibility manifest/budgets, independent memory, reuse threshold/age, RNG/schema
versions and loss settings. Freeze `selex_online_reference_v1.yaml` as the
historical prototype ablation; never overwrite it to make HSSK look compatible.
Its explicit historical selection may resolve missing source metadata to
`dean_prototypes` only through an identified legacy config adapter with a logged
ablation identity; it is not the default loader path. New configs must specify
source; unknown/missing source is an error, never a fallback.

Old release/design documents describe implemented d7ba087 history. Their use of
"primary" or "real integration" does not override this contract. In the future
implementation update their status wording and links, while preserving original
release results and source audit evidence. This step changes only this contract
and `DEAN_SELEX_ARCHITECTURE_REALIGNMENT.md`; no code/config/default is changed
on disk yet. The earlier default-until-smoke and raw-row higher-level proposals
are superseded, not alternative v1 modes.
