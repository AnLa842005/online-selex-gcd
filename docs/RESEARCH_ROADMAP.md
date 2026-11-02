# Online SelEx research roadmap

## Preserved foundations

- Offline SelEx reproduction: 73.95% result, losses, training defaults and baseline
  branch/tag remain unchanged. Different exposure/protocol prevents a direct
  comparison with clean Online GCD.
- Phases 1–3: fixed clean protocol, CPU fixtures, frozen-DINO infrastructure and
  authoritative Phase 3B result. Its historical JSON remains unchanged; Phase 3C
  hardens paths, compatibility and CPU evaluation.

## Phase 4A: closed negative/control ablation

Implementation: `64fc515894f30a0d48ce3f90294945408a5dd9c7`, opt-in
`frozen-dino-prototype-consolidation-v0`. See the [retained method](NOVEL_PROTOTYPE_CONSOLIDATION.md),
[real CUB closure/audit](CUB_PHASE4A_REAL_CONTROL_REPORT.md) and
[curated result](../results/cub_frozen_dino_consolidation_phase4a.json).

User-reported CPU execution is complete; no Pod remains running. One merge in
session 4 changed the final active count from 2067 to 2065, with Probe New unchanged
at 2.51%. Base-known merge calibration did not transfer effectively to novel
prototypes in this run. Phase 4A did not solve fragmentation. Keep it as a negative
control, not Online SelEx. Cumulative All is unchanged and New decreases; do not
call the Old/New redistribution an improvement or retune thresholds on novel/probe
labels. The original disabled prototype baseline remains the default.

Artifacts remain at `/workspace/runs/cub_phase4a_control_20261102_062837` on the
persisted RunPod Network Volume. The current task records supplied execution
figures and performs local source/schema tests only; raw files were not supplied
or independently inspected.

## Outstanding calibration reproducibility issue

The user reports original assignment-artifact structural import passed, strict
runtime equality failed under Python 3.13, and known-base-only recalibration
reproduced Phase 3B metrics exactly. Record all three facts separately. The cause
is an **unresolved numerical/runtime compatibility issue**, not a diagnosed Python,
BLAS or numerical bug. Original/recalibrated artifacts, field-level differences
and exact runtime versions remain to be audited. Do not weaken strict matching.
The old Phase 3B JSON retains pending local import; the new record carries the
reported runtime thresholds/audit without rewriting historical metrics.

## Phase 4B: Online SelEx Euclidean, specification frozen

[Specification v0](ONLINE_SELEX_EUCLIDEAN_PHASE4B_SPEC.md) reserves a separate method
identity and freezes the input/state/output contract, causal updates, frozen
encoder, zero replay, known-base calibration, temporal hierarchy evidence,
mechanism ablations, resource accounting and acceptance criteria. It proposes
hierarchy-conditioned assignment without retiring leaf IDs. No algorithm, head,
loss, CLI method or Phase 4B workload is implemented by this task. The method is
an untested adaptation, not renamed consolidation or an offline USE/SSE claim.

Current gates and future work:

- Complete: Phase 4A implementation/fixture validation and reported real negative
  control; baseline/offline behavior preserved.
- Complete: Phase 4B specification/plan and evaluator-only success criteria frozen
  before implementation; no performance promise.
- Pending: obtain and audit original versus recalibrated calibration and full
  runtime/input provenance, without inventing a mismatch cause.
- Pending, separate implementation task: isolated primitives/state and reference
  tests, causal hooks and required mechanism ablations on synthetic/fake caches.
- Pending, separately authorized real execution only: freeze implementation and
  configs, verify identical input identities, archive all metrics/calibrations,
  evaluate every predeclared gate and report failures/tradeoffs without probe tuning.

This task authorizes no real CUB run, GPU work, extraction, training or download.

## Phase 5: hyperbolic extension

Begin after a tested, versioned Euclidean method and fair baseline/control
comparisons. Specify geometry, hierarchy and numerical stability separately with
the same clean exposure/protocol. No hyperbolic code belongs in the Phase 4A
control or the first frozen-feature Phase 4B design.
