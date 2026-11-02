# Phase 4B-1: synthetic Online SelEx Euclidean implementation

Status: CPU implementation validated on artificial embeddings and fake feature
caches only. **No real CUB evaluation or research improvement is claimed.**
Approved specification amendments are in commit `d7acbf0`; the full base SHA is
`2ba7f40f3f2f817530557ba3a0721dbe9c3983d2`. See the
[spec-to-module/test mapping](PHASE4B_1_IMPLEMENTATION_MAPPING.md) and
[frozen decisions plus A1-A3](ONLINE_SELEX_EUCLIDEAN_PHASE4B_SPEC.md).

## Method and boundaries

The main identity is `online-selex-euclidean-frozen-evidence-v0`. Separate opt-in
mechanism identities are `online-selex-euclidean-instantaneous-geometry-v0`,
`online-selex-euclidean-lambda-zero-v0` and
`online-selex-euclidean-no-hierarchy-v0`. Mechanism ablations are separate arms,
not combined unregistered variants. Existing baseline/consolidation CLIs and
algorithms are unchanged. Schema dispatch alone adds support for the new state.
This implementation does not reuse/rename `ConsolidatingBank`.

The encoder is frozen and no model/checkpoint is loaded by the new runner. There
is no head training, USE/SSE loss reproduction, HSSK, replay, hyperbolic mapping
or desired class-count input. All fixture CLI features are 768-dimensional;
mathematical primitives accept a declared dimension for testing. The method is
a proposed hierarchy/evidence adaptation, not the reproduced offline SelEx.

For each stream item, compute the hard accepted set using the unchanged baseline
cosine gate. Within this set, compute squared-Euclidean logits with T=0.1, base
softmax p, group-balanced hierarchy distribution u, and q=.75p+.25u. Group mass
uses log-mean-exp of accepted children, shared equally within each group. Select
maximum q with persistent-ID ties. Lambda-zero and no-hierarchy inference use
an explicit baseline shortcut. If the gate is empty, preserve baseline stream
birth/probe rejection behavior. Parent support never lowers the assignment gate.

Leaves keep persistent IDs and baseline normalized EMA momentum 0.9/count updates.
Known leaves remain structural singletons but can receive unlabelled stream EMA,
as the spec requires. Composite parents affect inference only; they never replace
emitted class IDs or merge/split/delete leaves. Historical predictions stay raw.

## Causal ordering and zero replay

1. Compute candidate/evidence scores from pre-update state. Evidence uses the top
   novel-origin leaf plus up to eight alternatives, independently of gate acceptance.
2. Select the emitted ID, then apply birth or EMA once. Only afterward add the
   precomputed coactivation sums/opportunities and apply global deterministic
   degree pruning. A current birth cannot appear in its own evidence neighborhood.
3. Finish the stream session and record raw metrics. Rebuild complete-link groups
   from eligible evidence (at least three opportunities and tau_structure).
4. Evaluate current eligible probes read-only using the rebuilt memberships.
   Structural membership remains frozen during the next stream session; newly
   born leaves are implicit singletons until that boundary.

No current observation feature is retained after the call, except its contribution
to allowed aggregate prototypes/relations. Exactly-once sample delivery belongs
to the trusted runner, not a stored learner sample-ID history. Labels/schedules,
probe history and Hungarian mappings remain evaluator-only. The learner API
accepts only current opaque ID, session and feature; it never receives total true
novel classes or future-session metadata.

## Calibration A1 and geometry A2

Both new runners apply the existing known-only LOO assignment calibrator and use
its actual threshold; there is no CLI assignment override or fallback. The A1
calibrator receives authorized `(opaque known-base ID, feature)` groups only.
Input features are validated without extra normalization before observation;
seed vectors and observations undergo the baseline normalization lifecycle.

Sort IDs per known class, initialize two opaque pseudo-leaves, shuffle remaining
rows with an independent `random.Random(1)`, and deliver exactly one pseudo-session.
Hierarchy stays singleton, with rebuild prohibited. Birth/EMA/evidence follow the
same method. An external immutable birth-origin map covers both initial and born
leaves. It classifies retained eligible edges only for known-base calibration;
it is never attached to the learner. Empty positive or negative diagnostics and
classes with fewer than five samples fail explicitly. Balanced acceptance/rejection
uses exact integer numerators and the largest threshold on ties.

A2 instead uses `1 - 0.5 * squared_distance` on normalized final pseudo-leaf pairs,
calibrates a separate tau_geometry and has no opportunity gate/temporal evidence.
Runtime geometry edges are recomputed from pre-rebuild centroids, globally pruned
to degree eight, then passed through the same complete-link/group lifecycle/q.
There is no exponential distance score or new temperature for geometry edges;
T=0.1 is retained only in the already-frozen mixture logits. Numerical scores
are clipped at mathematical endpoints solely for floating-point roundoff.

Known-base-only identity hashes cover the authorized IDs/features, not full
partition/cache hashes that contain future data. Full source/cache identities
stay in runner provenance. Method artifacts also record source-code fingerprints
and Python/NumPy/BLAS versions without machine paths. Threshold fields and method
identities are explicit, and no Phase 4A threshold is borrowed.

The synthetic builder preserves original deterministic stream/probe generation,
but supplies a declared known-base geometry: two ID-sorted central anchors and
symmetric +/-0.25 tangent offsets for remaining known-base points. This exercises
valid calibration in 768D; it is a fixture, not a changed CUB population or an
extra method hyperparameter. A high-dispersion known-base fixture separately
verifies calibration starvation fails without fallback. Baseline equivalence
comparisons receive exactly the same generated data and calibrated assignment gate.

## State, node lifecycle and resume

Strict JSON schema 3 has distinct variant identity, both applicable thresholds,
frozen settings/config checksum, known-base calibration identities, leaf state,
bounded evidence sums/counts, active composite memberships, next_node_id,
observation/base counts and rebuild/pruning counters. Derived matrix/membership
caches and timing are not serialized. Schemas 1 and 2 retain their existing format
and loading paths; incompatible new state fails explicitly.

Singleton nodes are implicit. Rebuild reuses only identical active membership.
Disappearing membership retires and is returned in the runner's per-boundary audit.
Reappearing retired membership gets a new monotone ID. No retired registry or
birth-origin labels are retained in learner state. There is no leaf alias layer
because this method never retires leaves.

Validation rejects unknown fields/versions, invalid method/config identity,
nonfinite/zero/wrong-dimensional vectors, invalid thresholds/counts, duplicate
IDs/edges, overlapping or known-leaf membership, missing leaves, node references
that would introduce hierarchy cycles, invalid counters, excess degree and
geometry states containing temporal evidence. Cycles in the **undirected relation
graph** are legitimate; hierarchy node-to-node references are not. Save/load
preserves active IDs/counters, frozen memberships and evidence during a session.

Parent geometry is diagnostic only: current count-weighted Euclidean centroid
and maximum child distance. It is derived from current leaves, does not influence
q and need not be duplicated in the checkpoint.

## Runtime and limits

A contiguous cached novel region reuses the Phase 3C normalized matrix; evidence
does not rebuild Python lists of all prototype vectors per observation. Distances
are vectorized in blocks. Probe raw scores use bounded broadcast GEMV batches,
with per-query gated hierarchy scoring. Temporal relations are degree <=8
(E<=4N), parent size <=8, and complete-link candidates come from graph neighbors.
The optimized builder uses a lazy heap of affected group candidates, not an N-by-N
matrix. Small scalar/brute-force references live in tests.

Per-observation work is O(ND + N log N + E log E) for distances/top-neighbors and
exact evidence pruning; memory is O(ND+8N) plus bounded distance/probe scratch.
The sparse temporal rebuild uses bounded-degree/group neighborhoods and stale-heap
validation. Geometry ablation requires quadratic distance work and row rescans
can be worse in adversarial cases, but retains only O(N) best partners/selected
bounded edges, never a production full pair matrix. External geometry calibration
has quadratic **known-base-only** pair diagnostics/candidate records; large-base
scratch/artifact cost must be assessed before real execution. No live class-count
cap is introduced; unbounded leaf growth is explicitly retained/logged.

Report assignment, evidence, hierarchy, probe and metric times separately.
Structural-similarity time is a subset of hierarchy-update time; do not sum those
as disjoint phases. State/vector/graph JSON payload sizes are logged, not peak
RSS or a total Python memory estimate. Baseline/control runners retain existing
tracemalloc instrumentation; their end-to-end times are not directly comparable
as speedups against the new untraced runner. All runners restore thread settings.

## Fixture evidence only

The simple synthetic CLI fixture reaches 100% final/cumulative/probe accuracy,
13 leaves, zero parents, and 37 retained relation edges; zero forgetting/collapse.
This is a sanity check, not an empirical acceptance test or research result.

Hard fake-cache fixture (same metadata/features for every arm; overlapping novel
modes and dispersion):

| Arm | Final stream All/Old/New (%) | Cumulative All/Old/New (%) | Probe All/Old/New (%) | Leaves | Nodes/edges |
|---|---|---|---|---:|---|
| Baseline | 58.90/100/55.88 | 58.33/100/50.00 | 66.67/100/53.33 | 16 | 0/0 |
| Consolidation | 65.75/100/63.24 | 66.11/100/59.33 | 61.90/100/46.67 | 5 | 0/0 |
| Euclidean temporal | 58.90/100/55.88 | 58.33/100/50.00 | 66.67/100/53.33 | 16 | 5/52 |

Hard-fixture raw historical excess clusters: 9/17/9 (baseline/control/method);
probe excess: 3/0/3. Control makes 19 leaf merges and has one impure probe cluster
(class-merge excess one), versus zero explicit leaf merges and zero impure probe
clusters for baseline/method. Forgetting is 0 / 0.11111111111111112 / 0.
Thus low fragmentation alone can hide incorrect collapse or rejection. The main
method does not improve this fixture's leaf predictions; parents are associations,
not class relabelling.

The separate noisy fake cache distinguishes the main inference rule: baseline and
control Final All 93.15068493150684%, method 91.78082191780823%; all have cumulative
All 86.11111111111111% and Probe All 80.95238095238095%. Leaves remain 13, but the
method has four parents/40 edges; probe excess increases from two to three.
No empirical benefit is claimed. Full All/Old/New/collapse/fragmentation/forgetting
and payload fields are generated in `comparison.json`, not committed artifacts.

Observed persistent JSON sizes (baseline/control/method): hard fixture
165605/55364/175153 bytes; noisy fixture 134891/135042/142366 bytes. Vector payload
for hard fixture: 98304/30720/98304 bytes. These include differing schemas and are
not peak-memory estimates. Approximate single-run elapsed values were hard
2.30/1.93/0.68 seconds and noisy 1.87/2.94/0.60 seconds; instrumentation differs,
so these are not fair speedup comparisons.

Synthetic kernel benchmark: Python 3.11.9, NumPy 1.24.3, one BLAS worker, 128 novel
leaves, 768 dimensions, 48 distinct observation IDs, block 256, three repetitions.
Nine initial leaf centroids/features coincide, producing eight eligible edges and
one parent. Median assignment+evidence: optimized 0.04037 s vs scalar reference
0.20760 s (5.14x); sparse hierarchy: 0.000251 s vs brute-force 0.016746 s (66.77x).
This sparse synthetic kernel is not a real CUB benchmark or a worst-case bound.
IDs, leaf vectors/counts and memberships match exactly; independent scalar evidence
sums agree to absolute 2e-13. Ordinary tests assert no wall-clock speed threshold.

## Commands and artifacts

From the repository root, use fresh output directories with an existing parent:

```bash
python scripts/run_selex_euclidean_fixtures.py --output-dir outputs/selex_synthetic_1
python scripts/run_selex_euclidean_fixtures.py --output-dir outputs/selex_synthetic_2
python scripts/run_selex_euclidean_fixtures.py --fixture hard-cache --output-dir outputs/selex_hard
python scripts/run_selex_euclidean_fixtures.py --fixture noisy-cache --output-dir outputs/selex_noisy
python scripts/run_selex_euclidean_fixtures.py --variant geometry --output-dir outputs/selex_geometry
python scripts/run_selex_euclidean_fixtures.py --variant lambda-zero --output-dir outputs/selex_lambda0
python scripts/run_selex_euclidean_fixtures.py --variant no-hierarchy --output-dir outputs/selex_nohierarchy
python scripts/benchmark_selex_euclidean_cpu.py
python -m unittest discover -s tests -p 'test_online_selex*.py' -v
python -m unittest discover -s tests -v
```

The dedicated opt-in CLI accepts only generated fixtures, never user real-cache,
dataset or checkpoint paths. Its fake-cache API rejects non-fixture metadata
before feature loading, and rechecks loaded provenance. Existing real CUB CLIs
have no new method switch in this phase.

Semantic `euclidean.json` and schema-3 `euclidean_state.json` are byte-stable across
identical runs; nondeterministic clocks live in `euclidean_performance.json`.
Fake-cache runs also produce baseline/control files and a three-way comparison,
with a separate complete timing envelope. Save/load preserves exact serialized
state and future outputs in the same numerical environment. Block sizes are
recorded execution settings, so compare behavior/leaf/graph state after excluding
those settings and their provenance hashes when testing different block sizes.
No cross-hardware/library bitwise guarantee is made.

All output/cache/state/log artifacts belong under ignored `outputs/` or temporary
directories. This commit contains source, fixture builders/tests and documentation
only. The original Phase 3B/4A real result records are unchanged.

## Deferred before real CUB Phase 4B-2

Obtain/audit original versus recalibrated RunPod calibration JSONs, field-level
diffs and exact runtime/input hashes; the cause remains unresolved. Freeze the
implementation/configs and verify real feature/protocol identities. Check known
base coverage/calibration diagnostics and geometry-calibration memory cost without
using novel/probe labels. Benchmark actual-scale CPU state/growth/time and archive
all thresholds, raw/probe/collapse/fragmentation metrics and lifecycle audits.
Only a separately authorized real run may evaluate the frozen empirical success
criteria. Synthetic correctness, these negative fixture outcomes and the kernel
speedup do not establish research improvement. No real CUB/GPU/extraction/training,
compute rental, download or hyperbolic work ran during Phase 4B-1.

## Validation completed for this delivery

- Amendment check: the original `2ba7f40` decisions are unchanged; A1-A3 are an
  appended clarification in `d7acbf08338acd0af00d140d935df6c20bce1984`. The implementation
  commit appends status only, without altering those decisions.
- Targeted: **39 tests passed**, including seven fixture categories, A1 all-birth
  diagnostics and exact baseline pseudo-lifecycle, A2 separate geometry calibration,
  A3 active-node retirement/reappearance, independent references, strict schema,
  label/future invariance, RNG/thread/block tests and workload guards.
- Full suite: **167 tests passed** (128 existing +39 new; final run 120.858 seconds,
  Python 3.11.9 / NumPy 1.24.3). No native Python 3.13 equivalence is claimed.
- Two independent synthetic CLI runs produce byte-identical semantic result and
  state files. All four mechanism CLI variants run. Mid-session and boundary
  learner save/load match continuous inference, leaf updates and hierarchy exactly.
- Compilation passed for all Python files in `online_gcd`, `scripts`, `tests` and
  `project_utils`; seven relevant CLI help checks passed. Reproduction comparison
  returned `differences: {}`. `git diff --check` passed.
- Protected configs, split/data readers, offline models/losses/utilities, baseline
  and consolidation algorithms/runners/evaluators, DINO extraction/cache contract
  and authoritative real results are unchanged. Common state code adds only lazy
  schema-3 dispatch; old schema-1/2 output/loading is regression tested.
- The enabled synthetic CLI passes existing runtime guards against torch/torchvision,
  project data/models/methods/SSB imports, dataset/checkpoint opens, sockets and
  external subprocess workloads. New code performs no downloads or GPU work.
- Staging contains only source, tests/fixture builders and documentation. Generated
  cache/state/result/performance artifacts are ignored; no real data or checkpoint
  is staged. The amendment and implementation are separate commits and are pushed
  together only after these checks.

The CLI writes final checkpoints; the learner/state API supports mid-session
resume. Complete evaluator/orchestrator resume is not a CLI feature here: its
mapping/history/cursor must be separately persisted outside learner state before
a later production evaluation-resume adapter is introduced.
