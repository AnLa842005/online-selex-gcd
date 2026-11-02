# Phase 4B-1 implementation mapping

Base: `2ba7f40`; approved A1-A3 amendment: `d7acbf0` (2026-11-02).
This mapping is prepared before implementation. Only synthetic/fake caches are
in scope; remote calibration diagnosis and real execution remain Phase 4B-2 gates.

| Frozen requirement | Implementation | Acceptance tests |
|---|---|---|
| Frozen encoder, zero replay, leaf IDs, hard assignment gate, normalized sequential EMA | `selex_euclidean.py` | Birth/count/EMA, pre-update decisions, no label/schedule arguments, lambda=0 baseline leaf-state equality |
| Euclidean logits, group-balanced u, mixture q, singleton/disabled shortcut | `selex_euclidean.py`, `selex_math.py` | Hand-derived probabilities, independent scalar reference, accepted-set restriction, known singleton |
| Top novel plus eight alternatives; independent evidence neighborhood; means/counts; degree pruning | `selex_graph.py`, learner | Pre-update opportunity audit, exact degree bound, forgotten/recreated edges, RNG independence |
| Complete-link groups <=8, score/age/ID ties, no class-count oracle | `selex_graph.py` | Anti-chain, distinct-class relations without leaf collapse, heap/full reference equivalence |
| A1: one singleton pseudo-session, all births, external immutable origin labels, eligible diagnostics, fail without +/- | `selex_calibration.py` | Known-base-only projection, birth diagnostics, no rebuild feedback, insufficient samples/diagnostics, permutation/order isolation |
| A2: instantaneous normalized Euclidean pair score, separate known-only tau_geometry, no opportunities | calibration, graph, learner geometry variant | Score equation, zero temporal state, bounded exact greedy geometry edges, reference agreement |
| A3: implicit singleton, active-only composite ID reuse, retirement outside learner | learner, `selex_state.py`, runner | Reappearance gets new ID, missing/cyclic/duplicate nodes rejected, monotone counter, no retired registry |
| New strict state version, preserve schema 1/2, no sample/evaluator history | `selex_state.py`, lazy dispatch in `state.py` | Malformed/nonfinite/dimension/counter tests; mid-session and boundary save/load equality |
| Raw arrival metrics before rebuild; current probes read-only | `selex_runner.py`, evaluator hook | Ordering trace, immutable repeated/reordered probes; historical IDs unchanged |
| Baseline, consolidation and proposed method opt-in separated | dedicated fixture CLI/config; existing default runners unchanged | Existing suite; both old modes match; fake-cache three-way comparison |
| Seven specified fixtures, label/future invariance, hard negative cases | `tests/selex_fixtures.py`, `test_online_selex_*.py` | No synthetic research claim; report collapse separately from fragmentation |
| Bounded O(8N) relations, block 256, no full production pair matrix; separate timing/memory | math/graph/runner, synthetic benchmark | Bound instrumentation, block/thread equivalence; benchmark has no wall-clock assertion |
| Deterministic stable artifacts and independent reference | reference tests, CLI | Two runs excluding nondeterministic performance envelopes byte-identical; learner state exact |
| Frozen real-run acceptance criteria unchanged | docs only | Not evaluated on synthetic; no real cache/compute/training/download |

No Phase 4A merge algorithm is renamed or used as this learner. Composite parents
are structural experts, not class predictions. Known leaves can receive baseline
unlabelled EMA updates but cannot join composite nodes. All thresholds are
calibrated from authorized known base; real novel/probe labels are evaluator-only.

## Delivery validation

Implemented under the mapped module boundaries. All 39 new tests and the full
167-test suite pass. Mechanism identities remain separate. Reference probability/
evidence checks use strict numeric tolerances where independent scalar summation
rounds differently; leaf IDs/vectors/counts, tested memberships and metrics agree.
Same-implementation serialized resume and repeated semantic CLI artifacts are exact.
No empirical real-CUB acceptance criterion is evaluated by these fixture checks.
