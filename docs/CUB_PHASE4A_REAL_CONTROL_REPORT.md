# Phase 4A real CUB control: negative result and closure

**Status: Phase 4A closed as a negative/control ablation.**
`frozen-dino-prototype-consolidation-v0` is not Online SelEx. It does not solve
fragmentation on the reported real CUB execution. No threshold was tuned using
novel/probe labels, and this task performs no new real workload.

Machine-readable summary:
[`results/cub_frozen_dino_consolidation_phase4a.json`](../results/cub_frozen_dino_consolidation_phase4a.json).
This is a curated transcription of the user's supplied RunPod execution summary,
with local consistency checks. Raw artifacts were not supplied locally and were
not independently inspected. Do not present this as a fresh execution or a full
raw-artifact audit.

## Execution and retained artifacts

- Executed implementation commit: `64fc515894f30a0d48ce3f90294945408a5dd9c7`.
- RunPod CPU Pod: 8 vCPU; Python 3.13 virtual environment; cache samples: 11,788.
- User reports execution complete and no Pod remains running; no cloud status
  query or resource operation was performed here.
- Raw artifacts remain on the persisted RunPod Network Volume:
  `/workspace/runs/cub_phase4a_control_20261102_062837`.
- Exact package/BLAS versions, runtime seed/config, raw hashes, merge IDs and
  similarity were not supplied. The directory suffix is not proof of a UTC
  execution timestamp. The Phase 3B reference seed is 1; do not fill in an
  unverified runtime seed or cache hash from matching metrics.
- No dataset, feature matrix, checkpoint, raw learner state, per-class map or
  large log is copied into Git. The existing Phase 3B record remains immutable.

## Real CUB comparison

Values below are accuracy fractions, not percentages. Final means final-session
raw emitted stream predictions. Probe means eligible held-out fixed-mapping
accuracy. All exact supplied values are retained in the JSON.

| Quantity | Control off | Consolidation on |
|---|---:|---:|
| Final All | 0.2912801484230056 | 0.2912801484230056 |
| Final Old | 0.4153846153846154 | 0.4153846153846154 |
| Final New | 0.2675018422991894 | 0.2675018422991894 |
| Average online accuracy | 0.36920810510124125 | 0.36920810510124125 |
| Cumulative All | 0.23309608540925267 | 0.23309608540925267 |
| Cumulative Old | 0.30820547031354234 | 0.3122081387591728 |
| Cumulative New | 0.19552886219552887 | 0.19352686019352686 |
| Probe All | 0.19295823265447015 | 0.19295823265447015 |
| Probe Old | 0.36234396671289876 | 0.36234396671289876 |
| Probe New | 0.025085910652920963 | 0.025085910652920963 |
| Average forgetting | 0.06604838911435613 | Not supplied |
| Final active prototypes | 2067 | 2065 |
| Final active novel clusters | 1967 | 1965 |
| Retired clusters | 0 (control disabled) | 1 |
| Ever-created novel clusters | 1967 (derived from no retirement) | 1966 |
| Maximum merge-chain depth | 0 (control disabled) | 1 |
| End-to-end elapsed, approximately (s) | 15.139 | 17.152 |

Consolidation-on session details supplied by the user:

| Session | Merges | Active prototypes after session | Consolidation overhead, approximately (s) |
|---|---:|---:|---:|
| 1 | 0 | 315 | 0.025 |
| 2 | 0 | 597 | 0.118 |
| 3 | 0 | 938 | 0.345 |
| 4 | 1 | 1374 | 0.754 |
| 5 | 0 | 2065 | 1.781 |

Active-probe fragmentation after consolidation: mean excess clusters **4.5**,
duplicate cluster count **900**, fragmented class fraction **0.93**. No complete
per-class or raw historical fragmentation map was supplied; none is reconstructed.
The end-to-end and per-session timing figures are approximate and their full
measurement scopes were not supplied. They are not an exact additive timing
decomposition or a general performance benchmark.

Only one explicit merge occurred, in session 4. The final prototype difference
is two, not two merges: on created 1966 novel IDs and retired one, leaving 1965;
off had 1967 novel prototypes without retirement. One fewer subsequent birth plus
one retirement explains the bookkeeping difference. The supplied summary does
not identify which sample avoided a birth or the retired/survivor IDs. Historical
stream predictions must not be rewritten to make counts or metrics agree.

## Calibration audit: separate three claims

1. The user reports the original assignment artifact **passed structural import**
   through the existing importer. This checks the recorded artifact structure,
   checksum/identity/count/objective consistency; it is not feature recomputation.
2. The user reports **strict runtime matching failed** in the Python 3.13 runtime.
   The runner recomputes the full calibration record and requires equality, so
   the failure does not identify the differing field or its cause by itself.
3. Recalibration using **labelled known base only** reproduced all supplied
   Phase 3B control-off metrics exactly. This is metric reproduction, not proof
   that the original and recalibrated calibration artifacts are identical.

Reported runtime thresholds, retained as decimal strings as well as numeric JSON
values to avoid losing the supplied decimal spelling in a float serialization:

- Assignment: **`0.76694905900245858`**.
- Consolidation: **`0.84814506320210814`**.

These remain separate quantities. The consolidation threshold is not a replacement
assignment threshold. No novel labels, probe labels, true novel class count or
final metric feedback is authorized for calibration.

The mismatch is an **unresolved numerical/runtime compatibility issue**. Its cause
is not established. Python version, NumPy/BLAS behavior, changed candidate ordering
or any other explanation must not be asserted without field-level artifacts and
reproducible evidence. No strict comparison is weakened, importer behavior changed,
artifact checksum rewritten or tolerance silently introduced in this task.

The original calibration artifact and structural-import output are not attached
locally. The historical Phase 3B JSON therefore retains its original
`pending_import` / `threshold: null`. The new record separately captures the
reported RunPod audit and runtime recalibration thresholds. It does not pretend
that the artifact was imported or independently validated locally. Later audit
should obtain both original and recalibrated calibration JSONs, exact versions,
base/cache identities and a field-level difference before deciding whether any
compatibility change is justified. An artifact comparison is not permission to
rerun a real workload.

## Scientific conclusion and closure

Greedy novel–novel cosine consolidation was almost inactive on this real CUB run:
one merge in five sessions, only **2/2067** fewer final prototypes, and Probe New
unchanged at about **2.51%**. The base-known calibrated merge threshold did not
transfer effectively to the novel prototypes in this control. Phase 4A did not
resolve fragmentation. Keep this result as a negative/control ablation and keep
the disabled prototype baseline as default.

The change in cumulative Old/New is a redistribution, not an improvement:
Cumulative All is unchanged and New decreases. The observation does not establish
that every consolidation variant fails, that frozen DINO cannot support novel
classes, or that a future hierarchy will improve results. Do not lower the
threshold after inspecting novel/probe labels. Do not call this Online SelEx or
directly compare its scores with offline SelEx 73.95%, whose protocol and training
exposure differ.

Phase 4A implementation, tests and the reported real-run negative result are
closed. The unresolved calibration audit remains a tracked reproducibility issue;
closure does not mean that issue has been diagnosed. Phase 4B is frozen at
[specification/plan only](ONLINE_SELEX_EUCLIDEAN_PHASE4B_SPEC.md). No Phase 4B
algorithm, training or real workload is implemented or run by this task.

## Validation of this recording/specification task

- Pre-edit checks: branch `research/online-selex`, clean working tree, HEAD
  `64fc515894f30a0d48ce3f90294945408a5dd9c7`; `git pull --ff-only origin
  research/online-selex` reported already up to date. No user edits were present.
- Project Python 3.11.9: **128 tests passed** with
  `python -m unittest discover -s tests -v` (46.916 seconds), including all 119
  implementation tests and nine new curated-record/schema/provenance/link checks.
- The nine new standard-library-only tests also passed on Python 3.12. This is
  not a reproduction of the remote Python 3.13 calibration mismatch.
- Compilation passed for Python sources in `online_gcd`, `scripts`, `tests` and
  `project_utils`. CLI help passed for baseline/synthetic runners, fixture runner,
  consolidation benchmark, extractor, manifest builder and calibration importer.
- Offline reproduction config comparison returned `differences: {}`. Runtime
  algorithms, strict calibration/import behavior, baseline/configs, split/protocol,
  feature extraction/cache, offline methods/losses and the historical Phase 3B
  JSON/report are unchanged from the execution commit.
- `git diff --check` passed. Only documentation, the 6.6 KB curated JSON and schema
  tests belong to this task; no raw artifact, private dataset or large file is
  staged. Tests use synthetic/fake inputs, not the persisted real cache.
- Leakage review: no threshold is tuned from novel/probe labels; the new record
  does not enter learner configuration. Phase 4B remains text-only and its
  evaluator-only acceptance criteria cannot be learner inputs. No GPU extraction,
  real training, dataset download, cloud operation or real CUB workload ran here.
