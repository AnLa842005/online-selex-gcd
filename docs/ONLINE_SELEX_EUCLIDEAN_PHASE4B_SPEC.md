# Phase 4B Online SelEx Euclidean: frozen specification v0

**Status: specification/plan only, frozen 2026-11-02. No implementation or
performance result exists.** Reserved experiment identity:
`online-selex-euclidean-frozen-evidence-v0`. This task neither implements the
algorithm nor authorizes training, extraction, downloads or real-data execution.
Changing a decision below requires a recorded spec revision before evaluation,
not an undocumented change after seeing probe metrics.

## Hypothesis and method boundary

The proposed method uses temporally accumulated, label-free relational evidence
and a Euclidean prototype hierarchy to shape future online assignment. Whether
this reduces fragmentation is an untested hypothesis. The Phase 4A negative
result motivates studying evidence beyond a single cosine threshold; it does not
prove that a hierarchy will help.

The implementation motifs are the repository's geometry-derived soft evidence
and expertise structure, visible in `info_nce_logits`, `LabelSmoothingLoss` and
`train` in the [offline source](../methods/contrastive_training/contrastive_training.py). This design is
an **online adaptation proposal**, not a claim to reproduce offline USE/SSE/HSSK
or the offline 73.95% result. V0 has no gradient training, learned projection head,
HSSK, replay or encoder updates. Its state changes through current-observation
EMA, evidence accumulation and structural rebuilds only. A future loss/head
training variant needs its own specification and experiment identity.

Normalized DINO features make squared Euclidean distance and cosine monotone:
`d(x, p)^2 = 2 - 2 * cosine(x, p)`. Renaming this distance is not a contribution.
The proposed difference is explicit hierarchy-conditioned soft evidence and its
causal effect on later assignments/EMA, not replacing the Phase 4A name.

## Fixed protocol, inputs, state and outputs

Retain the clean DINOv1 backbone/checkpoint/CLS/normalization contract, SSB split,
base selection, manifest order, five sessions and existing probe eligibility.
Use the baseline/control's actual archived cache and manifest identities for a
later comparison; do not infer those hashes from matching metrics.

| Boundary | Allowed content | Forbidden content |
|---|---|---|
| Initialization | Labelled known-base feature groups, dimension, frozen settings and known-only calibration records | Novel/test labels, future stream features, evaluator maps, true novel class count |
| `observe` | Current opaque sample ID, session ID, one frozen feature vector | True class, true known/novel flag, class-bearing path, future batch or probe |
| Learner state | Active leaf prototypes/counts/origin/session metadata, ID counter, bounded aggregate relation statistics, internal parent nodes/memberships, calibration/config identities, progress counters | Stored stream images/embeddings, sample replay buffer, evaluator labels/mappings or historical prediction list |
| `predict` / `predict_batch` | Current feature(s), read-only leaf/hierarchy state | Any state, threshold, evidence or graph update |
| Output | Persistent leaf ID, or rejection `None` for probes; structural/uncertainty diagnostics without true labels | True class estimate obtained from evaluator mapping, parent ID substituted for historical leaf IDs |

Known leaf IDs are authorized by known-base labels. Novel-origin flags mean
"created without a base label", not a claim that the observation is truly novel.
Do not give the learner the known/novel class schedule or an expected cluster K.
All prototypes remain in frozen 768-dimensional feature space. No full offline
SelEx checkpoint is used. Retain float64 learner numerics and momentum 0.9.

## Update timing and causal contract

1. Before the first stream item, compute assignment calibration from labelled
   known base exactly as the baseline. Independently calibrate structural evidence
   from known-base pseudo-streams described below. Initialize known leaves; their
   hierarchy membership is singleton. Discard transient calibration feature views.
2. For observation t, snapshot the pre-update bank and the hierarchy membership
   frozen at the preceding session boundary. Compute scores and emit a leaf ID.
   Below the assignment gate, create a monotonically numbered novel leaf as in
   the baseline; its initial vector is the current normalized observation.
3. Only after emission, update the selected leaf with the baseline normalized
   EMA and increment its count once. Accumulate bounded relation statistics using
   **pre-update** scores from this observation; do not re-score the observation
   against its own updated centroid. Discard the observation feature afterward.
4. At session end, record raw stream metrics using arrival-time IDs. Rebuild the
   structural hierarchy using only current leaf summaries and accumulated
   evidence. It affects the next session's assignments and the current eligible
   probes, with no retroactive re-assignment of any stream observation.
5. Probe inference is read-only and must preserve an exact learner state digest.
   Return statistics and persist versioned learner state at the boundary.

New leaves stay singleton until the boundary rebuild. Parent nodes are structural
experts, never emitted cluster IDs. V0 does not merge, split, delete or alias leaf
IDs, including known leaves; it cannot reduce prototype count by relabelling old
outputs. It can change future birth/attachment behavior through hierarchy scores.
Known leaves may receive unlabeled stream EMA updates as in the baseline; a true
known-membership flag is never available. Prototype count is unbounded and logged,
not stopped at a desired number of classes.

## Separate assignment and hierarchy decisions

Let `s_j = cosine(x, leaf_j)` and let `tau_assignment` be the unchanged known-base
assignment threshold. The accepted candidate set is
`C = {j: s_j >= tau_assignment}`. If C is empty, follow baseline birth/rejection
behavior. Structural evidence cannot accept a leaf below this hard gate.

For accepted candidates, use Euclidean logits
`z_j = -||x - leaf_j||^2 / T` and `p = softmax(z over C)` with stable reductions.
For each frozen parent group g, restrict its members to C, let their count be m_g,
and compute `h_g = logsumexp(z_j for j in g intersect C) - log(m_g)`.
Assign hierarchy distribution mass proportional to `exp(h_g)` per group and
spread that mass equally over its m_g accepted members. Call the resulting leaf
distribution u. Use the SelEx-inspired soft-evidence mixture
`q = (1 - lambda) * p + lambda * u`, choose largest q, and break ties by persistent
leaf ID. Singleton groups reproduce p. This is a declared inference/update rule,
not an offline contrastive-loss formula or a ground-truth pseudo-label.

V0 fixes lambda=0.25 and T=0.1 before implementation; no novel/probe tuning is
allowed. Lambda=0 must take an explicit baseline inference shortcut to preserve
its exact numerical ties and EMA/state behavior, not rely on floating-point
softmax to recreate them. A no-hierarchy arm also uses this shortcut.

## Temporal relational evidence and structural hierarchy

Use two levels: leaf prototypes and disjoint local parent groups. Known leaves
stay singleton. For each current stream observation, find the highest-scoring
novel-origin leaf and up to eight nearest alternative novel-origin leaves using
pre-update Euclidean distances. This evidence neighborhood is independent of C:
assignment acceptance and structural opportunity collection are separate rules.
With fewer than two novel-origin leaves, collect no pair evidence.

Compute a stable softmax r over that at-most-nine-leaf neighborhood. For each pair
of the top novel leaf i and an alternative j, accumulate
`a_ij = 2 * min(r_i, r_j) / (r_i + r_j)` (positive denominator), its sum and an
opportunity count. The edge score is the running mean of a_ij. Each current
observation contributes to a pair at most once; exactly-once delivery remains the
runner's responsibility. No past feature or sample list is retained. Coactivation
means model ambiguity/relatedness, **not proven class identity**.

Keep at most eight undirected incident relations per leaf. Enforce this bound by
sorting edges by descending running-mean evidence then sorted ID pair, greedily
retaining an edge only if both degrees remain below eight; apply after each
observation. Dropped aggregate relations are forgotten, not replayed. Record
opportunity counts and pruning counts so evidence starvation is visible.

At a session boundary, start with singleton groups. An eligible leaf pair needs
at least three observation opportunities and mean evidence >= `tau_structure`.
Greedily combine two groups only if **every** cross-pair has an eligible retained
edge and the combined group has at most eight leaves. Choose the highest minimum
cross-pair evidence, then earliest member creation session, then sorted member
IDs. Recompute affected candidates; stop when no eligible combination remains.
There is no fixed number of parents/classes and no connected-component chain
shortcut. Parent centroid/radius summaries are Euclidean diagnostics; the scoring
rule above uses their frozen membership and current pre-update leaf logits.

Reuse an internal node ID only for the identical sorted leaf membership;
otherwise allocate a new monotone `node-...` ID and audit the structural change.
Node retirement is not leaf retirement and cannot change old predictions or the
leaf counter. Save membership, node counter and aggregate edge statistics so
resuming does not silently rebuild a different hierarchy.

Ambiguity can connect different true classes, known stream items can contaminate
novel relations, and missing edges can leave all leaves singleton. Known-base
transfer can fail again. Report these failures; do not add hidden-label logic,
relax the assignment gate or silently substitute the Phase 4A merge threshold.

## Known-base-only structural calibration

The structural calibrator is privileged but receives only labelled known-base
features and known-base identities. It never reads evaluator manifests. For each
known class, deterministically order base sample IDs and initialize two opaque,
unlabelled pseudo-novel leaves from the first two samples. Shuffle the remaining
known-base observations with local seed 1 and feed them once to the pseudo-learner;
true known labels stay in the calibrator for diagnostics, never in `observe`.
Use the same evidence-neighborhood, pruning and opportunity rules as live v0.

At the end, positive diagnostic edges connect pseudo-leaves originating from the
same known class; negatives connect different known classes. Require the minimum
three opportunities. Calibrate `tau_structure` over their mean evidence with
balanced positive acceptance/negative rejection, testing endpoints 0/1, observed
scores and next representable values above scores; choose the largest threshold
on objective ties. Record positive/negative counts, coverage, candidates, split,
feature/config hashes and objective. Do not use a target number of clusters.

Any known class with fewer than five base samples, or no eligible positive or
negative edge overall, fails calibration explicitly. No stream/probe fallback,
class skipping, threshold borrowing or silent parameter search is allowed.
The assignment calibration remains independent and unchanged. Fixed settings
(lambda 0.25, T 0.1, degree eight, group size eight, three opportunities) are design
constants rather than values selected on real CUB scores. Structural calibration
quality on known classes is not a performance guarantee on novel classes.

## Ablation boundaries and resources

| Arm | Assignment / update | Structure | Identity policy |
|---|---|---|---|
| Prototype baseline | Existing hard gate, nearest prototype, sequential EMA | None | Existing leaf IDs; no merges |
| Phase 4A control | Existing hard gate and sequential EMA | Base-calibrated greedy cosine novel merge after raw metrics | Retired leaf aliases; frozen raw history |
| Proposed Online SelEx Euclidean v0 | Same hard gate; hierarchy-conditioned soft evidence before sequential EMA | Temporal evidence and complete-link local parents | Leaf IDs retained; parents never predictions |
| Required mechanism ablations | lambda=0 shortcut; hierarchy disabled; instantaneous geometry without temporal evidence | Isolate mixture, hierarchy and temporal support | Same protocol and evaluator rules |

All arms use identical data exposure and archived inputs. Do not combine the
consolidation control into v0 or borrow its measured threshold. A later head,
USE/SSE/HSSK adaptation, leaf merge/reconciliation, replay budget or encoder update
is a new version/ablation, not an undocumented extension of v0. Phase 5 alone
covers hyperbolic geometry after Euclidean validation.

Use the existing CPU prototype matrix, bounded relation state O(8N), block size
256 for distance candidates and bounded group size eight; do not allocate a full
unbounded pair matrix. No global prototype/class cap is imposed. Report bytes,
leaf/edge/node counts and calibration, assignment, evidence, rebuild, probe,
metrics and total session times separately. Pin/reduce numerics deterministically
and restore thread settings. No GPU or training is required by this v0 design.

## Acceptance criteria fixed before implementation

**Engineering/leakage gates (mandatory):**

- All existing baseline/control tests pass; default and lambda=0 leaf predictions,
  counts, vectors, raw metrics and saved leaf state match the reference exactly.
- Hand-check the equations, deterministic ties, temporal opportunity counting,
  pruning, complete-link anti-chain behavior and internal-node identity handling.
- Replacing future stream/probe data or permuting hidden evaluator labels cannot
  affect an earlier learner state or any known-base calibration. Instrument the
  API boundary; no label-bearing object crosses it.
- Probe calls preserve the full state; save/load/resume preserves later outputs,
  evidence and structures. Thread/block choices preserve IDs and metrics in the
  same numerical stack. Invalid/cancelling/nonfinite state fails explicitly.
- Synthetic/fake-cache tests include distinct-class coactivation, evidence
  starvation, noisy known/novel ambiguity, growth/memory and negative examples.
  Do not require an artificial fixture to show improvement in every case.

**Predeclared empirical decision rule for a later authorized matched run:**

Primary metric is final eligible Probe New. Call v0 a promising result only if it
improves Probe New by at least **one percentage point** versus both baseline and
Phase 4A control, does not reduce Probe All, and reduces Probe Old by at most one
percentage point. Also require no decrease in cumulative All or cumulative New
and at least **10% fewer final active novel-origin leaf prototypes** versus the
matched disabled baseline. These are evaluator-only reporting criteria, not
learner targets or threshold-tuning signals. Report all failed criteria and
tradeoffs; passing an engineering gate alone is not a research success.

Always report raw session/average/cumulative All/Old/New with counts, eligible
fixed-mapping Probe All/Old/New, forgetting/churn, NMI/ARI, raw historical and
current probe fragmentation, rejections, birth rates, relation coverage/pruning,
parent/leaf counts, state size and runtime. Never interpret cumulative Old/New
redistribution with unchanged All as an improvement. A single seed is descriptive;
no significance or generalization claim without separately predeclared additional
runs. Never directly compare these scores with offline SelEx 73.95%.

## Implementation entry and work order (future task only)

1. Audit the original/recalibrated RunPod calibration artifacts and versions;
   distinguish structural validity, strict equality and metric reproduction.
   Do not weaken guards before diagnosing the mismatch.
2. Implement the mathematical primitives and versioned state in an isolated new
   method, with hand-derived reference tests. No default switch is changed.
3. Implement causal runner hooks and required mechanism ablations, then validate
   on synthetic/fake caches with runtime workload guards and saved state checks.
4. Freeze code/config/calibration policies and run all engineering gates. Only
   with separate authorization, perform the matched real frozen-cache comparison;
   archive provenance, all metrics and negative outcomes. This is not training.

Current deliverable stops at this specification. No Phase 4B source algorithm,
loss, state schema, CLI switch, real workload or trained model is added now.
