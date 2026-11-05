# Architecture realignment: Online HSSK + USE/SSE

Status: **reviewed architecture; locked v1 contract; documentation only**.
The authoritative invariants, lifecycle, failure policy and acceptance tests are
in [DEAN_SELEX_DESIGN_CONTRACT.md](DEAN_SELEX_DESIGN_CONTRACT.md). No HSSK code is
implemented by these documents. The contract supersedes the first draft's
prototype-default, DEAN-keyed memory and raw-row higher-level proposals.

## 1. Branch and baseline

Before this design work the tree was clean; local HEAD, source branch and live
origin source branch all matched
`d7ba08745a8043a87d3126da8366a4720d81951f`.
`experiment/dean-selex-hssk-online` was created and pushed from that exact commit.
`experiment/dean-selex-online` was not modified, rewritten or force-pushed.
This follow-up remains on the HSSK branch, with no new commit or push.

The 273 passing tests, Python 3.9 compilation and production-projector CPU probe
belong to the prior implementation release; they are not HSSK validation.
This step only writes this report and the design contract. No GPU, test suite,
merge or code/config edit is performed.

## 2. Correct identity and architecture

```text
DINO/LoRA embedding -> Online HSSK hierarchy -> USE + SSE -> DEAN online loss mixture
```

The hierarchy branch is detached; separately generated live two-view embeddings
and the shared projector carry USE/SSE gradients to LoRA.

| Provider | Scientific identity | Selection |
|---|---|---|
| `hssk_online` | **Online HSSK + USE/SSE** | Primary in the new research design/config; never silently substitutes another provider |
| `dean_prototypes` | **SelEx-inspired prototype hierarchy** | Explicit historical ablation; retain exact tested numerical behavior |
| Synthetic path | Synthetic test scaffolding | Existing separate fixtures, not either research hierarchy |

Both research providers reuse the tested DINOHead, corrected USE/SSE kernels,
optimizer/projector group, paired stream and logging. The existing prototype
implementation is not a full HSSK/Self-Ex discovery implementation. Calling its
loss kernels "reference_v1" does not change this distinction.

**Default decision is now changed from the first draft:** HSSK is the primary
provider in the future new active research configuration. The frozen historical
reference YAML remains an explicitly selected prototype ablation. No on-disk
runtime default is modified in this documentation-only step. Passing smoke tests
remains an execution gate, not a reason to silently route primary runs to EMA.

## 3. Read-only DEAN replay audit: facts, not inferred behavior

Audited clean external checkout:
`C:/Users/ASUS/Desktop/Code/DA/OCGCD-dean-audit`, HEAD
`28b9384cd7ea9b123b03d045bac2e92d8d0b2870` (the wrapper's pinned upstream).
Line numbers below are physical source lines of that SHA, not runtime-patched
AST line estimates. `Ours_train_online.py` executes in its module-level
`if __name__ == '__main__'` block (line 76); it has no enclosing training function.
The wrapper patches the compiled AST in memory, not those source files.

### Evidence map

| Source file, function/block and exact lines | Observed data flow / conclusion |
|---|---|
| `Ours_train_online.py`, `__main__`, 131; 136 | `n_replay` defaults to 5; `prop_samples` defaults to 0.8. The replay help string is not evidence of a raw replay buffer |
| `dataset/__init__.py`, `load`, 23, 35-36; 154-156 | CUB dispatches to `get_cub_datasets` with actual `args.prop_samples` and returns `train_labelled` separately from incremental unlabelled datasets |
| `Ours_train_online.py`, `__main__`, 182-184; 380 | Loads `dt_train_labelled` and its DataLoader; `dlod_tr_prv = dl_train_labelled` is merely an alias, not a replay manager or online GT-SSE input |
| `dataset/cub.py`, `get_cub_datasets`, 194-226 | Creates whole train split, selects base classes, samples labelled instances, retains complementary base-unlabelled IDs for the online stream |
| `dataset/get_data_utils.py`, `subsample_instances`, 4-12 | Resets NumPy to **0 inside this function** and samples `dataset.uq_idxs` without replacement. Do not reconstruct the labelled pool by assuming only run seed=1 or a generic 80% draw |
| `dataset/cub.py`, `CUBirds._load_metadata`, 51-71; `__init__`, 48 | CUB metadata contains true targets and filepaths. `uq_idxs` is initialized to `range(len(dataset))` after split loading; it is stable for this dataset/split/order, not a universal image ID |
| `dataset/cub.py`, `subsample_dataset`, 136-142; `online_subsample_dataset`, 126-134 | Filters `.data` and `.uq_idxs` together; online subsampling also filters `.ys`. IDs are preserved across subset copies |
| `Ours_train_online.py`, `__main__`, 392-395; 445-469 | Stream yields `(on_x,on_y,on_z)`, with `on_z` current sample IDs. Builds current-only dummy dataset, deterministic eval transform, extracts embeddings under no_grad |
| `Ours_train_online.py`, `__main__`, 512-522; 526-565 | Base predictions and seen/unseen energy split produce DEAN pseudo labels; unseen feature rows come from current batch embeddings, not historical replay images |
| `Ours_train_online.py`, `__main__`, **569-580** | VFA uses `unseen_feats = feats[idx_unseen].cpu()`, per-dimension std and `torch.normal(uns_f, unseen_std)` **n_replay times per unseen row**. Concatenates real unseen features with newly synthesized feature tensors |
| `Ours_train_online.py`, `__main__`, **584-595** | AP fits proliferated features to estimate cluster count; GMM fits/predicts on real unseen features. Synthetic replicas are intermediate clustering support, not GT image replay |
| `Ours_train_online.py`, `generate_dataset`, 29-39; `merge_dataset`, 43-58 | Copies/merges current `.data`, `.uq_idxs`, `.ys`; no maintained historical replay store is introduced here |
| `Ours_train_online.py`, `__main__`, 601-629 | Overwrites old/new current `.ys` with DEAN predictions/offset pseudo labels; merges and applies training transform. A base-range `.ys` label is still a prediction |
| `dataset/cub.py`, `CUBirds.__getitem__`, 99-122 | Normal mode returns image, transformed metadata GT target, stable ID. Binary mode returns `.ys` as training target PLUS separate metadata `real_target` and ID |
| `Ours_train_online.py`, `__main__`, 631-632; **709-726** | Binary mode yields `on_x,on_y,gt_y,on_z`. `gt_y` is used for pseudo-accuracy diagnostics, not authorized labelled replay. Its presence in memory does not grant supervision permission |
| `Ours_train_online.py`, `__main__`, 738-771 | Fifteen configured inner epochs use `model_now(on_x)`, CE on pseudo `on_y`, weighted EC and optimizer backward/step; no `n_replay` raw-image GT training loop |
| `Ours_utils/utils.py`, `Memory_Buffer`, 23-72 | A utility class can store sample tensors; the active entrypoint imports only `extract_embedding, discover_acc` from this module (line 25) and has no `Memory_Buffer` invocation. It is **not** the implementation of `--n_replay` |
| `dean_integration/runtime.py`, `patch_real_selex`, 58-71 | Existing prepare hook follows classifier expansion; augment receives original CE/EC, student, pseudo `on_y` and stable `on_z`, **no replay argument** |
| `dean_integration/selex_real_online.py`, `LabeledReplay`, 58-72; `RealSelExOnline.augment`, 199-214 | Existing typed true-replay API validates alignment and is explicitly caller-supplied. Production wrapper currently supplies none; a label tensor's dtype alone cannot prove GT provenance |
| `dean_integration/selex_real_online.py`, `CurrentBatchImages.from_cub`, 26-39; `prepare`, 123-147 | Current adapter reads filepaths only, not metadata GT. Existing prototype path extracts eval/no_grad features and restores training flags; preserve this tested path when wrapped |

### Definite answer about `--n_replay 5`

- It generates **synthetic feature vectors**, five per current unseen real
  feature when VFA is active. It does not configure stored raw samples, stable
  image-index replay, historical real features or a labelled replay capacity.
- The transient VFA tensor list has no true GT label per synthetic row and no
  distinct stable sample ID. It must never enter the GT-SSE path, positive-pair
  counts or HSSK real-support counts. With VFA disabled it is not generated.
- Real CUB image rows do have stable split-local IDs and metadata GT, and the
  actual base `train_labelled` pool is an eligible source. Current unlabelled
  rows also physically expose GT for evaluation; that is **not** labelled access.
- Therefore the existing `n_replay` path cannot legally supply supervised SSE.
  A **separate bounded labelled-reference replay manager** is required, sourcing
  only the verified original base-labelled split, with its own capacity/batch
  size, label provenance and manifest hashes. It does not change `n_replay=5`.

The source determines how the pool was made; this read-only audit does not claim
that the seed-1 checkpoint encodes its exact labelled sample list. Before use,
obtain or faithfully reconstruct/verify the original list from the pinned split
procedure and dataset metadata, including the internal seed-0 sampling detail.
Fail a requested GT-enabled run if eligibility cannot be established; do not
retrain 30 base epochs and do not promote the remaining base-unlabelled GT.

Upstream immutable source permalinks:
[online/VFA](https://github.com/KU-VGI/OCGCD/blob/28b9384cd7ea9b123b03d045bac2e92d8d0b2870/Ours_train_online.py#L569-L595),
[CUB labels/IDs](https://github.com/KU-VGI/OCGCD/blob/28b9384cd7ea9b123b03d045bac2e92d8d0b2870/dataset/cub.py#L99-L142),
[labelled split](https://github.com/KU-VGI/OCGCD/blob/28b9384cd7ea9b123b03d045bac2e92d8d0b2870/dataset/cub.py#L194-L226),
[instance sampler](https://github.com/KU-VGI/OCGCD/blob/28b9384cd7ea9b123b03d045bac2e92d8d0b2870/dataset/get_data_utils.py#L4-L12).
All conclusions above were obtained from local source, without executing training.

## 4. Locked Online HSSK flow

The full numerical/failure policy is contract sections 2-5. Execution order:

1. Let DEAN finish energy/AP/GMM/VFA discovery and classifier expansion unchanged.
   The HSSK branch receives detached, deterministic eval embeddings `[B,768]`,
   current sample IDs and eligible replay, not DEAN's known/seen/unseen decisions.
   DEAN label arrays can remain read-only context for diagnostics/legacy mode but
   cannot filter HSSK inputs or seed its memory.
2. Select bounded true replay once in class-complete groups: capacity 320
   references, at most 2/class, batch budget 32 references. Each selected class
   must provide exactly 2 distinct verified references, giving at most 16
   anchored classes. Classes with fewer than 2 eligible references cannot be
   selected for GT anchors or supervised SSE. Current samples, second views and
   memory centers cannot complete a pair. Underfilled pools yield fewer complete
   pairs without padding. Compute eval embeddings from the same backbone snapshot
   and fix GT assignments as anchors. Retain only references across batches.
   Canonicalize class/sample IDs before admission and selection; use deterministic
   class round-robin with a checkpointed cursor, independent of loader order.
   Freeze selected IDs and hash through all 15 epochs; resume restores them
   without reselection or a second cursor advance (contract section 2).
3. Read **HSSK-owned** bounded prototype memory as weak initialization candidates.
   It has independent tracks and geometric matching, not DEAN EMA/class IDs.
   V1 uses seed-only memory, zero prior weight after initialization. Weighted
   anchors are a separate future ablation, never fake extra raw rows/support.
4. Enumerate support-feasible K. A is the number of selected replay classes with
   exactly 2 distinct verified references each; `A <= 16`, selected count `2*A`.
   Single-reference classes are excluded from anchors and selected replay rows;
   neither views nor current points nor memory can increase A.
   Minimum support=2 distinct real images per leaf; no contribution from second
   views or memory. Free clusters need two current samples. Fit balanced then
   unconstrained semi-supervised refinement, hard GT assignments throughout.
   Recheck support after refinement. Versioned local score uses real-row mean
   squared distance plus 0.02 per free cluster; cap free clusters at 16 and total
   K by floor(real_support/2). This is explicitly an online K-policy adaptation.
5. Build higher hierarchy by weighted semi-supervised coarsening **of HSSK leaf
   centers**, recursively on child centers. Compose sample-to-leaf-to-ancestor
   mappings. Do not refit current raw rows at each level, use DEAN pseudo-anchors,
   or elevate HSSK-predicted current labels to GT. Stop at supported depth and
   positive projection-prefix width. This is an intentional reference departure.
6. On insufficient support, try previous-HSSK-tree transfer only under the
   contract's lineage/age/distance/GT/support compatibility gates. Otherwise
   whole-batch SelEx skip with counters or config-selected error. Never select
   `dean_prototypes`, invent a trivial valid hierarchy, or count memory as data.
7. Freeze the build/reuse/skip result for all 15 epochs. Update HSSK memory once
   after a successful fresh build, independent of the prototype ablation's EMA.
   Both fresh views of each sample ID read the same fixed assignments.
8. Use existing current-view USE/hierarchy SSE and verified replay-only GT SSE.
   Add unchanged gamma-weighted internal mixture to official DEAN CE+weighted EC.
   Preserve final classifier optimizer group, projector state, RNG isolation,
   component logs and exact disabled/gamma-zero bypass.

Scientific identity: **SelEx adapted to online discovery with HSSK**, not literal
offline reproduction. The contract departure register requires reference,
adaptation, reason, scientific effect and test/ablation for every difference.
Its explicit center-only coarsening supersedes the earlier proposal to rerun
higher-level BSSK over real batch rows. Its separate HSSK memory supersedes the
proposal to use DEAN-keyed EMA as seed memory.

## 5. State lifecycle and resume

The normative state table is contract section 6. In brief:

| Scope | State | Boundary / checkpoint rule |
|---|---|---|
| One incoming batch | Current/replay eval features and IDs; selected GT replay; build/reuse/skip result and fixed hierarchy | Build before epoch 0, freeze through epoch 14; save exact snapshot/status/IDs/hash/epoch cursor for safe mid-batch resume; discard detached feature rows afterwards |
| One optimizer update | Fresh views and autograd graphs | Discard after backward; persist RNG/cursors at atomic post-step boundary, not graphs/pixels |
| Across batches | Independent bounded HSSK tracks, replay references/ages, class round-robin cursor, allocators and counters | Save full bounded state, pre/post class cursor, selected class/sample IDs and versioned selection hash plus last-update batch; restore frozen selection without advancing again; no duplicate memory update or DEAN EMA import |
| Run | LoRA/teacher/projector/classifier/optimizer/scaler, paired stream, private/official RNG and provenance | Restore together before stochastic work; validate all manifest/config/schema/provider hashes |

Mid-batch resume cannot rebuild from the now-updated LoRA model. It must restore
the exact frozen snapshot and faithfully reconstruct original DEAN fixed training
augmentation via saved pre-transform RNG/order, not replace it with an eval image.
If reconstruction is not implemented, declare batch-boundary-only resume and
reject unsupported mid-batch checkpoints. Same-stack deterministic continuation
is the target; cross-device execution cannot claim bitwise replay.

Replay telemetry is emitted once per incoming batch, not per view or epoch:
`replay.class_ineligible.insufficient_distinct_samples` counts buffered classes
with fewer than 2 eligible distinct references; `replay.class_selected` is A;
`replay.reference_selected` is 2*A. `replay.selection_hash` is the canonical
SHA-256 string identity, not a numeric counter. Checkpoint counter-update state
to prevent duplicate increments on resume. The hash covers versioned canonical
manifest/batch/cursor/selected-pair metadata as defined in the contract.

The contract is the configuration source of truth: replay capacity **320**,
per-class capacity **2**, selected batch budget **32**, and
`insufficient_support_policy: reuse_then_skip | reuse_then_error` (the former
is primary). Obsolete `selex_labeled_replay_per_class=5` and
`hssk_degenerate_policy` are not accepted design settings or aliases. Official
DEAN `--n_replay 5` is unchanged and refers only to VFA synthetic features.

## 6. Localized implementation scope, after review

### Existing files that can remain byte-identical

- `dean_integration/selex_projector.py`, `selex_reference.py`, `selex_hierarchy.py`:
  shared head/loss mathematics and legacy EMA preserved. A sample-keyed adapter
  meets the kernel lookup contract without changing USE/SSE equations.
- `dean_integration/selex_rng.py`: retain the tested RNG isolation scope; new
  domain owners/state serializers can wrap it without changing legacy draws.
- `dean_integration/paired_stream.py`, `base_checkpoint.py`, `checkpoint_audit.py`,
  `official_load_audit.py`, `preflight.py`, `config.py`, `adapters.py`,
  `online_step_smoke.py`, `smoke.py`, `selex_online.py` and `__init__.py`.
- `configs/dean/selex_online_reference_v1.yaml`, official CUB configs, lockfile
  and synthetic config. The frozen reference file is explicitly the ablation.
- All existing test files; add separate architecture/provider/resume fixtures.
- Offline SelEx code, upstream checkout, datasets, stored checkpoints and runs.

### Existing files requiring localized edits

| File | Future responsibility |
|---|---|
| `dean_integration/selex_real_online.py` | Provider dispatch, independent memory and typed replay, sample keys, frozen status, checkpoint state; preserve tested legacy route and optimizer behavior |
| `dean_integration/selex_config.py` | Strict source/support/memory/replay/reuse/schema config; primary HSSK semantics and explicit historical config adapter |
| `dean_integration/experiment_config.py` | New active research config selection/identity; frozen prototype config becomes explicit ablation, not default; exact gamma-zero bypass retained |
| `dean_integration/runtime.py` | Minimal fail-closed hook to admit verified labelled references and persist full online state at safe boundaries; existing hooks have no replay argument, so producer wiring must be explicit |
| `dean_integration/external.py` | Only if required to thread the new audited runtime/checkpoint hook flags; no upstream filesystem patch |
| `scripts/run_dean.py` | New primary config, eligibility verification/worker wiring/resume preflight and state-path propagation; preserve official args/strict replay/paired stream |
| `dean_integration/provenance.py` | Provider/replay/hash/skip/reuse/checkpoint identity and log validation |
| `docs/DEAN_SELEX_REFERENCE_V1.md`, `DEAN_SELEX_REAL_INTEGRATION_PLAN.md`, `DEAN_SELEX_ONLINE_EXPERIMENT.md`, `README.md` | Future terminology/status/link updates only; preserve historical release metrics and source evidence |

The first draft said runtime/default selection could remain unchanged; the locked
replay producer and complete-resume requirements make that assertion too broad.
Only minimal wrapper/AST changes are intended, never permanent upstream edits.

### New files proposed

- `dean_integration/selex_hierarchy_provider.py`: protocol/context/result,
  explicit factory, legacy wrapper, sample-keyed immutable snapshot.
- `dean_integration/selex_hssk_online.py`: independent HSSK memory, deterministic
  bounded fit, center coarsening, reuse/support gates and global ID allocator.
- `dean_integration/selex_labeled_replay.py`: verified bounded references and
  deterministic admission/selection/eviction; no fake GT inference.
- `dean_integration/selex_online_state.py`: versioned atomic full-state resume,
  hash validation, hierarchy/replay/RNG serializers and safe-boundary protocol.
- `configs/dean/selex_online_hssk_v1.yaml`: explicit new primary profile;
  optional separate replay-matched prototype profile for fair comparison.
- `tests/test_dean_hierarchy_provider.py`, `test_dean_hssk_online.py`,
  `test_dean_labeled_replay.py`, `test_dean_selex_resume.py`.
- `docs/DEAN_SELEX_HSSK_ONLINE.md`: implementation runbook after review.

## 7. Required architecture-level tests

Contract section 8 defines detailed assertions. Minimum acceptance:

- Real primary CLI/config -> `OnlineHSSKHierarchyProvider`; prototype constructor
  and build replaced by raising spies remain uncalled, including reuse/skip.
- Frozen legacy parity under explicit ablation; HSSK/prototype memories disjoint.
- Holding features/replay/HSSK seed+memory fixed while changing DEAN pseudo labels
  leaves HSSK assignment/K/memory unchanged and never mutates DEAN discovery.
- Two views keyed by the same sparse sample ID share all hierarchy assignments.
- One build before 15 epochs; hash/geometry/assignments immutable until next batch.
- GT SupCon receives eligible replay only; reject pseudo-GT and poison/leak tests.
- A singleton replay reference never creates a GT-anchored leaf, even with two
  views, current samples or memory centers available. 32 selected references
  create at most 16 anchored classes; smaller selections consist only of pairs.
- Permuting loader order before admission/selection leaves selected IDs/cursors
  and selection hash identical. Save/resume restores the same class cursor,
  frozen selection/hash and next-batch selection without duplicate advancement.
- A second view is not a distinct real replay sample; deduplicate reference IDs
  before support counts, eligibility, A and telemetry are computed.
- Degenerate support -> compatible HSSK reuse, explicit finite skip or error;
  correct counters and no hidden prototype invocation.
- Center-only higher-level inputs, nested sample ancestors, unique group IDs,
  anchored labels, bounded K/support, empty/singleton/all-known/all-novel cases.
- Complete save/resume matches uninterrupted snapshot/memory/replay/IDs/RNG/views,
  next batches, model/optimizer and logs; unsupported state is rejected clearly.
- Bounded-state inspection and replay eviction; finite USE/SSE gradients, optimizer
  state retention, official disabled/gamma-zero identity and full existing suite.

Fair comparisons must include a replay-matched prototype ablation; otherwise
extra labelled replay supervision is confounded with the HSSK hierarchy effect.
Historical seed-1 full-run scores are not paired HSSK controls. Both arms must
share base checkpoint, paired manifest, online seed, official hyperparameters,
loss/projector settings and, where comparing hierarchy, identical replay budget.

## 8. Design review checkpoint

The architecture is locked by the contract; it is not an open choice between
prototype-default and HSSK-default, nor center coarsening and raw-row refitting.
Numerical K/reuse thresholds and replay budgets are explicit versioned settings
requiring sensitivity studies, not scientifically validated optimums. Exact
seed-1 eligible replay membership remains an operational preflight requirement;
the audit proves its source procedure, not an inspected historical manifest.

Only these two documents are changed. No implementation, tests/GPU execution,
commit, further push or merge is authorized by this step. Run `git diff --check`
and stop for review.
