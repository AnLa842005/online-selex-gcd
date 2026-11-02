# Online SelEx research roadmap

## Preserved foundations

- Offline SelEx reproduction: 73.95% result, algorithm, losses and training defaults
  remain unchanged; it is not a clean online initialization or a comparable score.
- Phases 1–3: fixed clean protocol, synthetic CPU baseline, frozen-DINO CUB
  infrastructure and the authoritative Phase 3B run. Phase 3C hardens paths,
  compatibility and CPU evaluation. The Phase 3B calibration artifact is pending.

## Phase 4A: consolidation control

Implemented as the opt-in `frozen-dino-prototype-consolidation-v0` ablation.
See [method, limitations, commands and validation](NOVEL_PROTOTYPE_CONSOLIDATION.md).
It tests novel-only greedy geometry, base-known merge calibration and cluster
lineage without encoder learning, replay or an oracle class count. The baseline
stays the default. Synthetic/fake-cache evidence validates behavior; no real CUB
consolidation result is claimed. This is a control, not Online SelEx.

## Phase 4B: Online SelEx Euclidean

Entry criteria before research training or real-data performance claims:

1. Phase 4A regression/equivalence, label-boundary, deterministic merge, lineage
   and resume checks pass; default clean baseline and offline defaults remain
   unchanged and the consolidation control is versioned independently.
2. Obtain/import/audit the original RunPod Phase 3B calibration using the existing
   validator, preserving authoritative metrics and provenance. Keep it pending
   until the real artifact is available.
3. With separate authorization, compare baseline and consolidation on the same
   persisted frozen-DINO cache/manifests/seed. Archive both threshold records,
   raw versus consolidated metrics, lineage, fragmentation, rejection, forgetting
   and runtime. Record negative results rather than tuning on probes.
4. Predeclare a separate Euclidean Online SelEx design and experiment identity:
   label-access boundaries, online adaptation of the SelEx objectives/hierarchy,
   update timing, state/identity rules, resource budgets and fair controls.
   Specify any future USE/SSE/HSSK or replay adaptation explicitly before coding
   it; Phase 4A implements none of them. Define tests on synthetic inputs first.

Real training, extraction or dataset access requires a new authorized workload.
The current phase does not automatically authorize any of it.

## Phase 5: hyperbolic extension

Begin after a tested, versioned Euclidean Online SelEx implementation and fair
Euclidean baseline/control comparisons exist. Specify geometry, hierarchy and
numerical stability separately; preserve the same clean protocol and exposure.
Hyperbolic geometry must not be mixed into the consolidation control.
