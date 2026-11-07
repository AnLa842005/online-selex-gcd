# Phase 4: explicit HSSK lifecycle and boundary resume

Status: **CPU-validated production boundary lifecycle/resume**; no Phase 4 GPU/full run executed.
Base commit: `a0ca7b5587f4b746d213d602602f3ae1dd7c0798`.
Branch: `experiment/dean-selex-hssk-phase4`. Date: 2026-11-08.
The DESIGN_CONTRACT and architecture realignment remain unchanged.

## Reviewed research decision and root cause

The user-approved rule keeps the reported real batch A=16, K_free=3 at one
valid leaf level. Separate ceil-halving requests two free parents, each requiring
two distinct free children: four required, three available. Anchored children
are fixed to anchored hypergroups and cannot fill that deficit. Neither image
support weights nor memory seeds create extra distinct child centers.

This is an intentional support stop, not a missing production hook. At the
base commit, `runtime.py:66` invokes prepare after native classifier expansion;
`selex_real_online.py:147-163` dispatches HSSK; `selex_hssk_runtime.py:34` calls
the provider; `selex_hssk_provider.py:65-71` calls the actual higher coarsener.
`selex_hssk_hierarchy.py:69-72` rejects the infeasible free-parent count before
fitting. Child maps are appended only after acceptance (`selex_hssk_provider.py:76`).
The source policy matches DESIGN_CONTRACT section 3, items 7-8.

No ceil-to-floor conversion, minimum-support reduction, forced level, anchor/free
merge, leaf K change or DEAN fallback is introduced. New diagnostics expose:

```json
{"status":"LEAF_ONLY","reason":"insufficient_free_child_support",
 "child_count":3,"requested_parent_count":2,
 "minimum_children_per_parent":2,"required_child_count":4}
```

A separate implementation drift was found against contract section 3.7:
Phase 3 checked total-count reduction but allowed one anchored child to remain
unchanged while free groups shrank. Phase 4 now stops at
`anchored_side_no_strict_reduction` when a present anchored side cannot shrink,
as the locked contract requires. This adds no level and does not affect the
reported A=16/F=3 stop or the supported A=16/F=4 and F=6 fixtures.

Other stops explicitly report `no_strict_reduction`, `weighted_fit_rejected`, or
`projection_prefix_limit`. The provider counts every successful BUILD as leaf-only
or multi-level, levels, and stop reason. It separately counts higher usage across
BUILD/REUSE decisions. Summary/provenance report both higher-BUILD fraction and
higher-batch fraction; skipped batches do not count as using higher hierarchy.
A leaf-only-dominated full run is a research finding, not evidence of full
hierarchical SelEx benefit. Any future policy ablation needs another config/branch.

## Supplied GPU evidence and evidence limits

The user reports an A40 Phase 3 smoke at the base commit, DEAN upstream
`28b9384cd7ea9b123b03d045bac2e92d8d0b2870`, seed 1 / online seed 17:
one incoming batch, two requested/completed updates, exit 0, no batch-limit failure.
Replay: 32 references, 16 complete two-reference classes; manifest SHA
`f425e2b7a4ff4d2c994d4b153743115dd75b85733c93656128fb9d714d31e6a5`.
HSSK: BUILD, A=16, K=19, K_free=3, levels=1, child_maps=[], 19 new tracks,
zero evictions. Snapshot SHA:
`2b1c83d267e0ba973b4f730bf73d9e26b9269208b321f5890a0c048c528292f7`.
Total losses: 7.772308349609375 and 7.574171543121338. Projector/LoRA gradients
were finite/nonzero. Reported peak allocated/reserved CUDA bytes:
12,071,806,464 / 12,654,215,168. This audit did not fetch the original pod archive.
These observations do not establish Phase 4 GPU resume equivalence or accuracy.

## Additional source audit

### Singleton and absent groups

Leaf support includes current plus verified replay images. USE/hierarchy SSE use
only current images; replay is the separate GT-SSE term (contract section 5).
A replay-anchored leaf may thus have zero or one current image. In
`selex_reference.py:51-75`, current singleton groups invalidate both view anchors;
lines 90-96 count absent/singleton current groups, and line 158 counts hierarchy
SSE absent groups. Counts accumulate per update. The reported 12 singleton,
10 empty and 24 no-positive counters over two updates are consistent with this
mechanism, but exact per-leaf counts need the original assignments/logs.
No fake GT, additional replay contrastive rows or second-view support is added.

### Metrics and FUSE probe

The old metrics path already appended and closed JSON records in `_event`;
absence in an observed console listing does not prove no file was written.
Phase 4 gives each record a monotonic event ID, logs `hierarchy`, committed
`update`, `batch_complete`, `resume`, `complete` and `failed`/`error` events.
Updates are emitted after native `opt.step`, not merely after loss construction.
Summary carries the last record and the same aggregate counters. Each append
closes the handle for realtime readback. Checkpoints bind a journal prefix by
byte offset and SHA256. A fresh resume output copies only that verified prefix;
uncommitted trailing records are excluded, and logical updates are not duplicated.
Historical one-batch smoke events remain compatible with its existing tests.

`preflight.py` previously used `TemporaryFile(dir=output)`. It now uses a named
exclusive `mkstemp` file: write, flush, fsync where supported, close, readback,
then unlink only that exact probe. Unsupported fsync is tolerated only for the
explicit unsupported-operation error codes; other I/O failures propagate.
No user directory is removed. Tests simulate missing O_TMPFILE support and
readback failure/cleanup. Production checkpoints belong on local SSD; FUSE is
for verified persistence, not a claim of POSIX atomic-durability equivalence.

### Native loop and mutation boundaries

Pinned `Ours_train_online.py:446` is the incoming loop. Line 709 iterates the
prepared real batch (native inner loader capacity 128, incoming size 64), line
738 starts 15 epochs, line 771 calls `opt.step`. The incoming count is preserved;
no batch/replay/projector reduction occurs. The native evaluation loop reuses the
name `batch_idx` (line 807), so lifecycle saves the incoming session/batch at entry
and never uses the clobbered evaluation index for checkpoint position.

The session hook runs after student/LoRA initialization and before the incoming
iterator. The incoming hook is the first statement before discovery. The
post-step hook commits update counters. The batch-end hook runs after native
batch evaluation and class-count updates; checkpoint RNG therefore includes
all native discovery, training and evaluation RNG consumption. The original
student/teacher/classifier calculations, energy discovery, VFA and CE/EC remain
native. No permanent upstream edits are made.

## State machine and supported resume boundary

```text
READY -> incoming guard -> discovery -> BUILD/REUSE/SKIP decision
      -> FROZEN (15 committed native optimizer updates; same snapshot)
      -> native batch evaluation -> BOUNDARY -> atomic checkpoint
      -> next incoming guard -> fresh decision
```

`FROZEN_REUSE` in update telemetry means repeated use of one immutable snapshot,
not another provider build or cross-batch REUSE decision. Failed higher builds
roll back provider memory/allocator/previous snapshot; lifecycle preparation also
rolls back replay selection and eval RNG on error. SKIP never fabricates a
hierarchy. Failed operations record errors instead of publishing success.

**Resume is deliberately completed-batch-boundary-only in v1.** Mid-batch save
and resume are hard errors. This is the explicit fallback allowed by contract
section 6: preserving DEAN's original fixed `on_x` augmentation realization is
required before mid-batch resume can be offered. Phase 4 does not approximate it
with eval images, retain raw images in checkpoints, or silently restart a batch.
A crash inside a batch resumes the previous complete batch checkpoint and
re-executes that unfinished batch from its boundary RNG/stream position.

The equivalence fixture runs 30 updates continuously versus 15 updates, save,
fresh construction, restore, and 15 more updates. It uses actual HSSK leaf/higher
fit, current/replay views, projector, USE/SSE and AdamW, and executes the pinned
native classifier optimizer expansion AST on CPU. It compares losses, model
and optimizer state, private/global RNG, replay cursor/hash, hierarchy hashes and
counters. Projection dimensions are intentionally small only in CPU fixtures.
The numerical fixture uses a small LoRA-style model and CE with EC disabled;
existing native-loss/disabled/gamma-zero tests continue to cover official parity.
This is not a full real-DINO/CUB/CUDA execution claim.

## Checkpoint schema: dean-hssk-boundary-v1

| Field | Saved state / strict restore |
|---|---|
| schema and state hash | Exact version and deterministic recursive digest of tensors, arrays and metadata; unknown/missing fields fail |
| identity | Git/upstream SHA, official/SelEx config hashes, actual dataset-image content hash, DINO/base hashes, eligibility-manifest hash, stream hash, seed and online_seed |
| runtime | Python/Torch/NumPy/CUDA versions, CUDA device names, deterministic flags, cuDNN flags and CUBLAS workspace setting; incompatible stack rejected |
| engine | BOUNDARY/freeze step=15, batch/committed update count, counters, optimizer group descriptions, event cursor, private RNG domains |
| hierarchy | Exact current snapshot/status/assignments/ancestors/radii/hash; successful source tree, provider identity, allocator, bounded independent memory and counters |
| replay | Verified references, admission ages, completed selection payload/hash, pre/post cursor, last batch and cumulative counters; no second selection on restore |
| modules | Student/LoRA, teacher, dynamic classifier, teacher classifier and projector weights, training flags and requires_grad flags |
| optimizer | AdamW moments/options and ordered parameter names; restored groups resolve by names, not process-specific object IDs; classifier remains final |
| discovery | Native class counts, seen classes, score summary and session knowledge variables; no current raw images or unbounded historical feature bank |
| position | Single supported incremental session, exact next incoming batch; paired manifest order restored without reordering |
| official RNG | Python, NumPy, Torch CPU and all initialized CUDA devices |
| journal | Source metrics path, committed prefix offset and SHA256; prefix required on resume |

Atomic post-batch checkpointing clears gradients, writes a same-directory temp
file, fsyncs, and replaces `online_state.pth`. A write failure preserves the
previous checkpoint. This overwrites only the checkpoint owned by the new run,
never baseline artifacts. Restores require trusted local checkpoints; they use
Torch serialization and are not an untrusted-pickle sandbox.

Models and classifier shape are restored before named optimizer reconstruction.
All private domains are restored, followed by the official RNG after setup.
Initial constructor/setup draws are discarded by restoration; the saved snapshot
is never rebuilt using already-updated LoRA. Dataset image hashing detects changed
pixels as well as the existing metadata/eligibility checks. No requested resume
falls back to fresh state. Resume targets a fresh output directory and requires
the old checkpoint plus its referenced or adjacent hash-verified metrics journal
and the same assets. No journal is silently synthesized if it is missing.

## Explicit execution gates

- Existing `online-smoke` / `smoke-worker`: unchanged one incoming batch,
  one or two optimizer updates; distinct batch-limit error remains intact.
- New `lifecycle-smoke` / `lifecycle-worker`: exactly two incoming batches,
  15 updates each, hard failure on missing/extra batches or updates.
  It does not require real CUB to yield a particular hierarchy depth; CPU fixtures
  cover both valid leaf-only and support-feasible multi-level cases.
- Full HSSK requires `--action train --hssk-production`, active reference HSSK,
  paired mode and a clean pinned checkout. Merely supplying `--smoke-updates`
  or a resume file cannot enable it. Default APIs/actions retain their gate.
- `--resume-online-state` is accepted only by this explicit lifecycle, never the
  historical strict replay or one-batch smoke protocol.
- Disabled/gamma-zero ordinary DEAN paths remain unchanged; these controls omit
  `--hssk-production`. Historical `dean_prototypes` remains an explicit ablation.

Production means the tested boundary-state protocol is available. It does not
claim GPU scientific validation. Lifecycle enables deterministic Torch algorithms
and pins the CUBLAS workspace setting; unsupported deterministic CUDA operations
must fail, not silently relax determinism. Full-run provenance checks final
completion, committed counts, hierarchy aggregate and checkpoint byte hash before
publishing COMPLETE. Nonzero child exits remain FAILED even if an old complete
summary is present.

## Intentional differences from SelEx c3d3b42

No loss formula, temperature, gamma or model dimension changes in Phase 4.
The missing anchored-side stop is corrected to enforce the existing numerical
policy; no new research policy is introduced. All research departures in DESIGN_CONTRACT section 7 remain.

| Reference behavior | Adapted behavior | Reason | Scientific impact | Test / ablation |
|---|---|---|---|---|
| Offline whole-dataset refresh | Existing locked per-batch freeze, now repeated across online batches | Production stream lifecycle | No new target policy; observed depth varies with support | Leaf-only/multi-level real fitter fixtures; depth coverage reporting |
| Offline checkpoint regime | Strict complete online boundary checkpoints and isolated RNG/journal restoration | Reproducible interrupted paired runs | No new loss; mid-batch resumes explicitly unsupported | Continuous versus interrupted CPU equivalence; future CUDA equivalence smoke |
| Offline logging | Depth/stop coverage and committed update journal | Make support-driven leaf-only behavior visible | Prevent misleading full-hierarchy claims | Aggregate/counter/journal tests; future policy changes require separate ablation |

The radius lookup optimization keeps an additional tiny immutable scalar JSON
payload rather than reparsing all 768-D centers for each anchor. Copy-out values,
serialized canonical payload, snapshot hash and loss arithmetic are unchanged
by that lookup optimization. New Phase 4 diagnostic fields do change snapshot
hashes versus Phase 3; checkpoints are source/version-bound and do not silently
import Phase 3 trainer state.

## RunPod staging and future lifecycle smoke (not executed)

Use real paths discovered on the pod; do not guess asset locations or redownload
CUB. Network volume stores the original CUB archive, checkpoints, baseline
provenance and final run archives. Local SSD stores extracted CUB, verified copies
of checkpoints and live output. Preserve the original DINO asset pathname on resume: the existing
paired manifest binds official arguments including that path. A relocated checkpoint
can read its hash-verified adjacent metrics journal, but asset relocation must not
silently rewrite stream identity. Expected asset hashes must come from the seed-1
baseline archive/provenance, not freshly computed hashes alone.

```bash
set -Eeuo pipefail
: "${REPO:?}" "${PHASE4_SHA:?}" "${UPSTREAM:?}"
: "${SSD_ROOT:?}" "${PERSIST_ROOT:?}" "${CUB_ARCHIVE:?}"
: "${DINO_SOURCE:?}" "${BASE_SOURCE:?}" "${DINO_EXPECTED_SHA:?}" "${BASE_EXPECTED_SHA:?}"
: "${ELIGIBILITY:?}"
test "$(git -C "$REPO" rev-parse HEAD)" = "$PHASE4_SHA"
test -z "$(git -C "$REPO" status --porcelain)"
test "$(git -C "$UPSTREAM" rev-parse HEAD)" = 28b9384cd7ea9b123b03d045bac2e92d8d0b2870
RUN_ID="hssk_phase4_$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_RUN="$SSD_ROOT/$RUN_ID"
mkdir "$LOCAL_RUN"
mkdir "$LOCAL_RUN/data" "$LOCAL_RUN/checkpoints"
# Inspect the trusted archive layout before extraction; it must contain CUB_200_2011/.
tar -tf "$CUB_ARCHIVE" > "$LOCAL_RUN/cub-archive-contents.txt"
tar --no-same-owner --no-same-permissions -xf "$CUB_ARCHIVE" -C "$LOCAL_RUN/data"
cp -- "$DINO_SOURCE" "$LOCAL_RUN/checkpoints/dino.pth"
cp -- "$BASE_SOURCE" "$LOCAL_RUN/checkpoints/base.pth"
printf '%s  %s\n' "$DINO_EXPECTED_SHA" "$LOCAL_RUN/checkpoints/dino.pth" \
                     "$BASE_EXPECTED_SHA" "$LOCAL_RUN/checkpoints/base.pth" \
                     > "$LOCAL_RUN/expected-assets.sha256"
sha256sum -c "$LOCAL_RUN/expected-assets.sha256"
sha256sum "$LOCAL_RUN/checkpoints/dino.pth" "$LOCAL_RUN/checkpoints/base.pth" > "$LOCAL_RUN/asset-sha256.txt"
cd "$REPO"
COMMON=(--config configs/dean/cub_official_v0.yaml --upstream "$UPSTREAM"
  --dataset-parent "$LOCAL_RUN/data" --checkpoint "$LOCAL_RUN/checkpoints/dino.pth"
  --base-checkpoint "$LOCAL_RUN/checkpoints/base.pth" --paired-online --seed 1 --online_seed 17
  --use_selex_online --selex-implementation reference_v1
  --selex-config configs/dean/selex_online_hssk_v1.yaml
  --replay-manifest "$ELIGIBILITY"
  --replay-manifest-sha256 f425e2b7a4ff4d2c994d4b153743115dd75b85733c93656128fb9d714d31e6a5)
python scripts/run_dean.py --action verify --hssk-production "${COMMON[@]}" \
  --output "$LOCAL_RUN/preflight-output" > "$LOCAL_RUN/preflight.json" 2>&1
# Future authorized GPU action only; hard two-batch/30-update lifecycle budget.
set +e
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=60s 30m \
  python scripts/run_dean.py --action lifecycle-smoke "${COMMON[@]}" \
  --output "$LOCAL_RUN/smoke" 2>&1 | tee "$LOCAL_RUN/launcher.log"
statuses=("${PIPESTATUS[@]}")
set -e
printf 'python_timeout_exit=%s\ntee_exit=%s\n' "${statuses[0]}" "${statuses[1]}" > "$LOCAL_RUN/exit-status.txt"
sync
# Persist content without preserving FUSE-incompatible timestamps/permissions.
ARCHIVE="$LOCAL_RUN.tar"
tar --exclude="$RUN_ID/data" --exclude="$RUN_ID/checkpoints" \
  -cf "$ARCHIVE" -C "$SSD_ROOT" "$RUN_ID"
EXPECTED_ARCHIVE_SHA=$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)
cp -- "$ARCHIVE" "$PERSIST_ROOT/$RUN_ID.tar"
printf '%s  %s\n' "$EXPECTED_ARCHIVE_SHA" "$PERSIST_ROOT/$RUN_ID.tar" > "$LOCAL_RUN/persist.sha256"
sha256sum -c "$LOCAL_RUN/persist.sha256"
cp -- "$LOCAL_RUN/persist.sha256" "$PERSIST_ROOT/$RUN_ID.sha256"
sync
# Only after successful readback/hash verification may the pod be stopped.
```

No `cp -a` to geesefs. On OOM/timeout, preserve logs/peaks/exit status and stop for
review; do not lower 65536 dimensions, batch 64 or replay 32, and do not retry.
The future smoke directory is
`$LOCAL_RUN/smoke/experiments/cub_hssk_lifecycle_v1/seed1/`.
It contains metrics, summary, stream/eligibility manifests, console/execution
logs and `online_state.pth`. Aggregate provenance lives in its parent directory.
The old journal must remain available to resume. A future resume uses the same
COMMON arguments plus `--stream-manifest <saved-stream>` and
`--resume-online-state <saved-online_state.pth>` into a NEW output directory.
Do not infer model quality from two batches; do not start full training merely
because a command is documented. Pod watchdog/stop automation remains an
operational wrapper requirement; this runbook does not claim it was executed.

## Validation and remaining limits

Final validation: **388/388 CPU tests passed, no skips, in 226.785 seconds**
(365 existing +23 Phase 4). The focused Phase 4/Phase 3/one-batch-guard run passed
55 tests in 80.705 seconds. The final full suite includes subsequent completion
metadata checks. The suite runs in Python 3.11.9 with pinned external fixtures
available through `DEAN_UPSTREAM_CHECKOUT`. Actual **CPython 3.9.13** compiled all
14 changed Python files; this is compilation, not a full Python 3.9 dependency
runtime test. `git diff --check` and a whole-file trailing-whitespace audit of
all 15 changed/new files passed.

```powershell
$env:DEAN_UPSTREAM_CHECKOUT = (Resolve-Path ../OCGCD-dean-audit).Path
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

No tests are claimed skipped/passed by replacing the production fitter or loss
with a fake implementation. Gate/error/atomic-write tests use explicit injected
failures; the real CPU numerical/lifecycle equivalence tests execute the shared
fitter, coarsener, losses, projector and native head/optimizer-expansion AST.
The CPU suite uses small heads; production remains 65536/2048/256. Same-stack
CUDA bitwise continuation still requires a separately authorized real-asset test.
A40 Phase 3 memory figures do not include full-state serialization/copy spikes;
full teacher/student/projector and AdamW checkpoints can be large and synchronous
per-batch writes add CPU RAM, storage and wall-time cost. Stage them on SSD and
measure this overhead during lifecycle smoke. Mid-batch resume, multiple
incremental sessions and cross-runtime/device exact replay are not supported.
The scientific hierarchy policy remains unchanged even if most real batches
remain leaf-only. No GPU or full training was run during implementation.


## Exact release file scope

- `dean_integration/external.py`
- `dean_integration/paired_stream.py`
- `dean_integration/preflight.py`
- `dean_integration/provenance.py`
- `dean_integration/runtime.py`
- `dean_integration/selex_hierarchy_provider.py`
- `dean_integration/selex_hssk_hierarchy.py`
- `dean_integration/selex_hssk_provider.py`
- `dean_integration/selex_hssk_runtime.py`
- `dean_integration/selex_real_online.py`
- `dean_integration/selex_lifecycle.py`
- `dean_integration/selex_online_state.py`
- `scripts/run_dean.py`
- `tests/test_dean_phase4.py`
- `docs/DEAN_SELEX_PHASE4.md`

Contract, realignment, configs, loss kernels, projector, legacy EMA, stored
checkpoints/datasets and upstream checkout are unchanged. The original
`selex_smoke_gate.py` one-batch hard guard is byte-identical to a0ca7b5.
Protected branches remain `experiment/dean-selex-online` at d7ba087 and
`experiment/dean-selex-hssk-online` at a0ca7b5. Only the Phase 4 branch is to be
committed/pushed after validation; no PR, merge, GPU or full training action.
