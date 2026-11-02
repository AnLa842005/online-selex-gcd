# Novel-prototype consolidation control (Phase 4A)

Experiment: **`frozen-dino-prototype-consolidation-v0`**. This is a CPU ablation
control, **not Online SelEx**. Consolidation is disabled by default. The existing
baseline identity, state schema, predictions and metrics are preserved when it
is disabled. Feature extraction, DINO definition, SSB split, stream order and
probe eligibility are unchanged.

## Current status: closed negative control

The user reports that real CUB CPU execution at `64fc515` has completed. It yielded
one merge, final active prototypes 2067 -> 2065, and unchanged Probe New at about
2.51%. Phase 4A did not resolve fragmentation and is retained as a negative/control
ablation. See the [real CUB report and calibration audit](CUB_PHASE4A_REAL_CONTROL_REPORT.md)
and [curated record](../results/cub_frozen_dino_consolidation_phase4a.json).
The base-known calibrated threshold did not transfer effectively in this run;
do not tune it using novel/probe labels or rename the control as Online SelEx.
Cumulative All is unchanged while New decreases; Old/New redistribution is not
an improvement. The raw artifacts remain on the persisted RunPod Network Volume
at `/workspace/runs/cub_phase4a_control_20261102_062837`.

This documentation task records supplied evidence, not an independent raw-artifact
audit or a new workload. The following method/fixture notes describe the retained
control implementation. [Phase 4B is frozen at specification/plan only](ONLINE_SELEX_EUCLIDEAN_PHASE4B_SPEC.md).

## Motivation and scope

The authoritative clean frozen-DINO CUB baseline discovered 1967 novel clusters
for 100 true novel classes, with Probe New 2.51%, Final All 29.13% and 2067 active
prototypes. See the immutable [Phase 3B report](CUB_FROZEN_DINO_PHASE3B_REPORT.md)
and its tracked summary JSON. These observations motivate testing consolidation;
they do not set the merge threshold or a target cluster count.

This control isolates geometric consolidation from future representation learning.
It implements no USE, SSE, HSSK, replay, encoder updates or hyperbolic geometry.
It does not implement Online SelEx's learned hierarchy or research contribution.
Known prototypes never participate in merges, even if a novel vector is identical
to a known vector. No known-to-novel reconciliation, splitting, pruning or target
cluster budget is implemented. The algorithm receives no true novel labels,
probe labels, introduction schedule or true number of novel classes.

## Calibration without novel-label leakage

Assignment and consolidation thresholds are separate. The CUB runner retains
its existing known-only assignment calibration unchanged. The synthetic runner
retains its configured assignment threshold. Neither is overwritten by merge
calibration. Results explicitly record `assignment_threshold` and
`consolidation_threshold`, with distinct calibration records.

The merge calibrator accepts only labelled base-known groups and their base/cache
identities. For each known class, preserve input base order and split at
`floor(n/2)` into two non-empty halves; normalize individual features, average each
half and normalize both sub-prototypes. There is one positive cosine per class.
Negatives are all unordered cross-class pairs among these sub-prototypes: four
negatives for each unordered pair of classes. For K known classes, counts are
K positives and `4 * K * (K-1)/2` negatives. Balance the two categories:

`0.5 * (positive_accepted / positive_count + negative_rejected / negative_count)`.

Acceptance is `cosine >= threshold`; rejection is `cosine < threshold`. Candidates
include -1, 1, all observed scores, and the next representable float64 values above
those scores within [-1, 1]. Compare exact integer objective numerators because
positive and negative sample counts differ. Choose the higher threshold on ties.
The checksummed record includes method, split rule, base feature hash, identities,
class/sample/pair counts, candidates, acceptance/rejection counts, objective and
selected threshold. Fewer than two classes, fewer than two samples in any class,
invalid vectors or cancelling half means fail explicitly; no class is silently
skipped. Calibration finishes before constructing the privileged evaluator.

This is transfer from known-class geometry, not an estimate guaranteed to work on
novel classes. It uses no novel/probe labels, final CUB metrics or class-count
oracle. A strict calibrated threshold may perform no merges; do not loosen it
based on probe scores.

## Session timing and metrics

Each session proceeds in this order:

1. Deliver current stream observations sequentially; assign against the current
   bank and immediately apply the same normalized EMA as the baseline.
2. Record session-local and cumulative raw stream metrics from emitted IDs.
   Preserve every historical prediction, including subsequently retired IDs.
3. Consolidate active novel prototypes using only their current aggregate state.
4. Evaluate only eligible probes against the consolidated bank without updates.
5. Return metrics, merge audit and lineage state; CLI `--save-state` persists the
   final learner state. The state API can also save any session boundary.

Existing stream accuracy, NMI/ARI, fragmentation and fixed-mapping metrics remain
raw. Later sessions can emit different predictions from the disabled baseline
because consolidation changes the bank used for future observations. Existing probe accuracy, forgetting and identity churn use current inference
IDs. Reports additionally contain:

- `prototypes_before_consolidation`, `prototypes_after_consolidation`;
- `novel_prototypes_before`, `novel_prototypes_after`, `merge_count`;
- cumulative `retired_cluster_count`, `maximum_merge_chain_depth`;
- `assignment_threshold`, `consolidation_threshold`;
- `consolidation_time_seconds`, `similarity_time_seconds`, `merge_time_seconds`;
- `raw_stream_accuracy`, `raw_stream_fragmentation` (current stream session);
- `raw_historical_fragmentation` (all emitted stream predictions to date);
- `consolidated_probe_accuracy`, `active_probe_fragmentation` (eligible probes);
- per-session `merge_history` and configured block size.

Raw stream and active probe fragmentation cover different populations. Their
numerical difference is not itself evidence that merging improved a common
population. Rejections are excluded from distinct-cluster counts exactly as in
the existing fragmentation function, so low fragmentation with high rejection
must not be interpreted as successful discovery. Reports retain the legacy
stream/probe metrics and rejection-related fields for context.

The privileged evaluator keeps the raw historical identity mapping unchanged.
For probe scoring only, resolve mapped IDs through aliases. If multiple mapped
IDs collapse, prefer the active survivor's original mapping; if it has none,
choose the lexicographically smallest mapped retired ID. Record conflicts as
`canonical_mapping_conflicts`. Never rematch on probe labels or rewrite raw
stream IDs. Identity churn still reflects changes in emitted IDs after merges;
it is not a lineage-adjusted churn metric. All evaluator labels and mappings
remain outside the bank and calibrator.

## Deterministic greedy algorithm

For active novel prototypes only, compute cosine similarity and select the
highest-scoring pair. Break ties by the earliest creation session among the pair,
then the lexicographically sorted persistent ID pair. Merge iff the similarity is
at least the calibrated threshold. Recompute affected similarities and repeat.
This is greedy centroid agglomeration, not connected components or single-link.
For example, vectors at 0, 20 and 40 degrees with a 25-degree acceptance radius
produce one merge: the new centroid is too far from the remaining endpoint.

Choose the survivor by oldest creation session, then lexicographically smallest
ID. Its vector becomes the normalized observation-count-weighted mean of the two
current vectors. Sum observation counts, preserve novel origin, take earliest
creation and latest update sessions. These are weighted **EMA prototype** means,
not exact replay-based historical sample means. The survivor is not marked as a
new stream observation. Cancelling centroids fail before retiring either member
of that pair. Only an explicit merge removes an active prototype.

## Cluster lineage and state

Every merge records retired ID, survivor ID, session and similarity. Aliases map
retired IDs to a canonical active survivor. Live aliases are transitively
compressed after consolidation; explicit `resolve_alias(..., compress=True)`
also compresses paths. Read-only resolution does not mutate state during probes.
Original retirement edges remain in `merge_history`, allowing the maximum merge
chain depth to remain meaningful after path compression.

Future predictions and observations query only active prototypes, so they emit
canonical IDs. The monotonically increasing novel-ID counter never reuses a
retired ID. Historical predictions keep the IDs originally emitted. The baseline
keeps schema v1 unchanged; consolidation uses strict JSON schema v2 with its own
experiment version, both thresholds, block size, aliases, active prototypes and
merge history. Loading rejects cycles, missing targets, active IDs used as retired
aliases, known IDs in lineage, inconsistent retirement history/counters and
unknown fields. It rebuilds the Phase 3C matrix cache. Saving/restoring learner
state preserves future inference, EMA updates, subsequent merge history and IDs.
This is learner resume, not a complete evaluator/history resume command.

## Runtime and determinism

The algorithm reuses the Phase 3C contiguous normalized prototype cache. The
optimized implementation retains the best partner for each active novel row.
After merging, unchanged rows compare only against the new survivor. Rows whose
best partner changed or retired rescan their candidates in blocks. Retain O(N)
best-partner metadata and O(ND) vectors; score temporaries are at most one row by
`block_size`, with O(block_size * D) gathered vectors. There is no production
N-by-N score matrix or quadratic pair heap. In the worst case many rows need
rescanning, so worst-case repeated agglomeration can still be expensive.

`consolidate(implementation='reference')` is the small-fixture full-matrix oracle;
it recomputes all similarities after each merge and is not exposed by the real
CUB CLI. Tests compare it with optimized runs over several block sizes.

Pairwise similarities use fixed-order float64 NumPy einsum reductions without
BLAS dispatch. Merge norms and calibration use one BLAS worker. Enabled runners
also pin assignment to one BLAS worker, preserving the default baseline numerical
policy independently of the caller's requested thread setting. CUB results record
both requested and effective BLAS counts. All thread settings are restored on
exit. No GPU dependency is introduced. Determinism is tested across requested
counts 1, 2 and 4 in the same numerical environment; this is not a claim of bitwise
identity across arbitrary hardware or library versions.

Calibration time is logged/stored separately. Each consolidation logs similarity
and merge time separately, plus total time and before/after counts. Metric timing
excludes consolidation and probe inference; total session timing includes them.
No evaluator class labels are printed.

## Commands (fixtures only for this phase)

The original synthetic baseline remains the default:

```bash
python scripts/run_synthetic_online_baseline.py
python scripts/run_synthetic_online_baseline.py --consolidate-novel \
  --consolidation-block-size 64 \
  --output-json outputs/synthetic_consolidation.json \
  --save-state outputs/synthetic_consolidation_state.json
```

Use fresh output files with an existing parent directory. Run internal fake-cache
comparisons, writing both baseline/control records and both learner states into a
new directory (all metadata and embeddings are generated, not real CUB):

```bash
python scripts/run_consolidation_fixtures.py --output-dir outputs/fake_cache_control
python scripts/run_consolidation_fixtures.py --overfragmented \
  --output-dir outputs/overfragmented_control --consolidation-block-size 7
python scripts/benchmark_consolidation_cpu.py
python -m unittest discover -s tests -p test_online_consolidation.py -v
```

The internal fixture script imports the repository's test builder. Production
CUB CLI still rejects fixture caches and exposes no fixture bypass. The
`--overfragmented` geometry is a purpose-built test of merges, not a calibrated
model of the real CUB distribution. Keep generated outputs under ignored
`outputs/` or outside Git; do not commit generated result/cache/state artifacts.

## Commands for a future authorized replication

The reported real execution is now recorded separately above. These commands are
reference instructions for a future authorized replication, not commands run by
this documentation task. Reuse persisted clean manifests/cache without extraction
or training; verify the actual archived identities before claiming matched inputs:

```bash
python scripts/run_cub_frozen_dino_baseline.py \
  --manifests /path/to/persisted/manifests --cache /path/to/persisted/features \
  --output-json /path/to/new/baseline.json --save-state /path/to/new/baseline_state.json
python scripts/run_cub_frozen_dino_baseline.py \
  --manifests /path/to/persisted/manifests --cache /path/to/persisted/features \
  --consolidate-novel --consolidation-block-size 256 \
  --output-json /path/to/new/consolidation.json \
  --save-calibration /path/to/new/assignment_calibration.json \
  --save-consolidation-calibration /path/to/new/merge_calibration.json \
  --save-state /path/to/new/consolidation_state.json
```

Use the same existing clean config, seed, cache and manifests for both runs.
Archive merge count, lineage, raw stream and consolidated probe metrics,
fragmentation/rejection diagnostics and runtime, even if the control does not
help. Never tune the merge threshold on novel/test labels or these final scores.
Known/novel confusions and over-merging remain possible; new aggregate centroids
can move, and greedy merging is not an endpoint-purity guarantee.

The user reports original-artifact structural import passed on RunPod, but strict
runtime matching failed in its Python 3.13 environment. Known-base-only runtime
recalibration reproduced Phase 3B control metrics exactly. The cause remains an
unresolved numerical/runtime compatibility issue; exact metrics do not prove
calibration-record equality. Runtime assignment/consolidation thresholds are
`0.76694905900245858` / `0.84814506320210814`, separately recorded in the real-run
summary. These are not fixture thresholds.

Neither original nor recalibrated raw calibration JSON is attached locally. The
historical Phase 3B record therefore remains unchanged (`pending_import`, threshold
null), while the new record preserves the reported RunPod audit. No importer or
strict runtime comparison is changed. Obtain the original/recalibrated artifacts,
versions and field-level diff before diagnosing or changing compatibility behavior.

The [research roadmap](RESEARCH_ROADMAP.md) separates Phase 4A consolidation
control, Phase 4B Online SelEx Euclidean and Phase 5 hyperbolic work.

## Observed fixture evidence (2026-11-02)

All numbers in this section are artificial CPU fixtures, not real CUB results.
The default seed-1 synthetic run emitted 13 active prototypes, zero merges and
100% final raw stream/probe accuracy. The ordinary fake-cache fixture emitted 13
active prototypes and zero merges in both modes: final raw stream All
`0.9315068493150684`, probe All `0.8095238095238095`. A strict known-half threshold
can correctly leave a fixture unchanged; no threshold was adjusted to force an
improvement.

The purpose-built two-mode fake cache used known halves at ±15 degrees and novel
modes at ±10 degrees around distinct artificial axes. Baseline/control comparison:

| Quantity | Baseline | Consolidation control |
|---|---:|---:|
| Final active prototypes | 12 | 7 |
| Final active novel prototypes | 10 | 5 |
| Retired IDs / total merges | 0 | 5 |
| Final raw session All | 0.6027397260273972 | 0.7945205479452054 |
| Final probe All | 1.0 | 1.0 |

The control's cumulative raw history still contains five excess cluster IDs,
while its current eligible probes have zero excess clusters. Those populations
are different. The separate four-observation unit fixture explicitly verifies
two same-class merges, no cross-class merge and unchanged historical IDs.
Full-matrix reference and optimized fake-cache runs match exactly in all metrics,
merge history and saved learner state, excluding timing fields.

Synthetic kernel benchmark (96 initial novel prototypes, 64 dimensions, block
size 32, seed 123, three repetitions; Windows Python 3.11.9 / NumPy 1.24.3):
reference median `0.1666984999901615` seconds, optimized median
`0.0664595999987796` seconds, **2.51x** speedup. Both perform 92 merges to four
active novel prototypes with identical serialized state and history. The benchmark
threshold is a fixed artificial kernel fixture setting; experiment thresholds
still come exclusively from known-base calibration. There is no wall-clock
assertion in unit tests and no claim about real CUB runtime or quality.

## Implementation validation at 64fc515 (before the reported real execution)

- **119 tests passed** with `python -m unittest discover -s tests -v` in the project
  Python 3.11.9 environment (final run: 58.044 seconds). This includes all 92 prior
  tests plus 27 consolidation tests.
- Explicit disabled/default equivalence, pre-Phase-3C reference equivalence,
  optimized/full-matrix merge history and state equivalence, multi-session raw
  history, threshold separation, alias validation/compression, save/load/resume,
  block sizes and requested BLAS thread counts passed.
- A runtime-guarded enabled synthetic CLI run blocked torch/torchvision, project
  datasets/models/methods, SSB, checkpoint/dataset access, sockets and subprocess
  workloads. Known-base calibration records remain identical after replacing
  stream/probe features. Raw-metrics → merge → probe order is explicitly audited.
- Ran default synthetic baseline, enabled synthetic ablation, ordinary fake-cache
  comparison and over-fragmented fake-cache comparison. CLI-produced schema-v2
  state reloaded with matching separate thresholds.
- Python compilation passed for all files in `online_gcd`, `scripts`, `tests` and
  `project_utils`. CLI help passed for synthetic/CUB runners, fixture comparison,
  consolidation benchmark, manifest builder, extractor and calibration importer.
- Offline reproduction config comparison returned **`differences: {}`**.
- Protected baseline bank/assignment calibration, clean protocol/manifest builder,
  SSB/data/splits, model/feature extraction/cache contract, offline methods/losses,
  reproduction configuration/utilities and authoritative result records are
  unchanged from `77131b5`.
- `git diff --check` passed. Only source, test builders/tests and documentation are
  committed. Generated fake metadata/cache/result/state/log files are ignored;
  no real dataset, cache, checkpoint or result artifact is added. No GPU, training,
  real CUB workload or asset download was run.
