# Phase 2: synthetic cosine prototype baseline

Version: **euclidean-prototype-v0**. This is infrastructure validation, **not a
research result**. Synthetic performance must not be compared with SelEx paper
results. The full SelEx checkpoint is not loaded or used. No CUB images, CUB
features, SSB files, network, training, feature extraction or CUDA are needed.

The offline SelEx baseline remains unchanged at tag `selex-cub-baseline-v1`:
73.95 / 71.45 / 75.21 All/Old/New-v2, PASS WITH QUALIFICATION. It was exposed to
unlabelled novel images, so it is an offline upper-bound reference or an explicitly
labelled warm-start ablation only. It is **not clean Online GCD initialization**.
"Upper bound" here names a comparison role, not a proven numerical bound across
different tasks. Phase 2 supersedes Phase 1's checkpoint-base choice for this
synthetic experiment; it preserves the Phase 1 stream generator and CUB config.

## Baseline and lifecycle

Despite its version name, assignment uses **cosine similarity on normalized
vectors**. No hyperbolic representation or learned encoder is involved.

```text
labelled known base features -> normalized mean -> known prototype bank
current safe observation -> normalize -> nearest active prototype by cosine
    similarity >= threshold: select existing persistent ID
    otherwise: create next persistent novel ID
    record prediction against the pre-update bank -> normalized EMA
hidden stream label + prediction -> isolated evaluator (no feedback to bank)
eligible held-out probe features -> read-only prediction -> isolated evaluator
```

Known initialization accepts only labelled base features grouped by known class.
Normalize each base feature, take the arithmetic mean per class and normalize
that mean. IDs are `known-<known_class_id>`; known labels are authorized here.
The base observation count equals the number of labelled features. There are no
novel base samples. Creation and last-update session are initially zero.

For each online observation, normalize the feature, compute dot products against
all prototypes, and select the maximum. Exact similarity ties choose the
lexicographically smallest stable ID. The attach threshold is inclusive.
Assignment does not use whether the observation is actually known or novel.
A known observation can therefore create a novel-origin cluster, or a novel
observation can attach incorrectly to a known-origin prototype.

For an existing prototype, use:

`p_new = normalize(momentum * p_old + (1 - momentum) * normalize(x))`.

Increment its observation count and update its last-session metadata, preserving
origin and creation session. For a new cluster, use `novel-000001`, then
`novel-000002`, etc., from an internal counter independent of hidden labels.
Initialize its unit vector to the observation and count to one; an EMA with
old=observation leaves that vector unchanged. No double counting occurs.
All prototypes stay active. There is no merge, split, deletion, ID recycling,
replay, hierarchy, USE/SSE, optimizer or encoder update.

Threshold must be finite in [-1,1]; momentum finite in [0,1]. Momentum 0 replaces
the selected vector; momentum 1 freezes it while counts still advance. Wrong
dimensions, nonnumeric/zero/NaN/infinite vectors, invalid metadata and backwards
sessions fail explicitly. An exactly cancelling normalized mean or EMA also
fails; it is not replaced with a fallback vector. Updates validate before mutation.
The bank permits nondecreasing positive session IDs without knowing future T;
the runner/evaluator enforce consecutive sessions. Exactly-once observation IDs
are enforced by stream construction and evaluator checks. The bank retains no
sample-ID history or past features, only aggregate prototype state.

Read-only `predict(feature)` follows the same similarity rule but returns `None`
below threshold. It never creates a cluster or updates counts/vectors/counters.
Probe rejections count as incorrect; empty subsets are reported as null, not zero.

## Synthetic populations and configuration

Configuration: [synthetic_prototype_v0.yaml](../configs/online_selex/synthetic_prototype_v0.yaml),
loaded through Phase 1's YAML loader with `SyntheticConfig`. The default loader
still uses `ProtocolConfig`, so existing Phase 1 calls and manifests are unchanged.
The synthetic config's `protocol_name` identifies the **reused scheduling rules**;
`dataset: synthetic` distinguishes it from the CUB/SSB experiment.

| Setting | Default |
|---|---:|
| Seed / feature dimension | 1 / 32 |
| Known / novel classes | 3 / 10 |
| Sessions / new classes per session | 5 / 2 |
| Labelled base samples per known class | 20 |
| Stream samples per class, across all sessions | 30 |
| Held-out probes per class | 12 |
| Isotropic Gaussian noise standard deviation | 0.025 |
| Cosine attach threshold / EMA momentum | 0.85 / 0.9 |

A local NumPy Generator creates a Gaussian matrix and QR orthogonalizes its
columns; deterministic diagonal-sign adjustment defines random orthogonal class
centers. Dimension must be at least known+novel class count. Add independent
Gaussian noise to centers and normalize every observation. Noise is configurable
in [0,1]; these easy defaults intentionally isolate infrastructure correctness.

Base, stream and probes use independent draws with disjoint opaque random IDs.
IDs contain a population prefix and random hex token, never a class ID. All base
labels are known. The generator retains hidden stream/probe labels only in its
privileged container. `learner_session(t)` exposes just sample ID, session ID and
feature vector. The bank receives only those three values, and never receives
`SyntheticData`, the full configuration, schedule or total novel count.

The Phase 1 `build_stream` function assigns novel classes one introduction session
using its local seeded shuffle. It allocates each original stream sample once,
with per-class round-robin allocation from introduction onward and seeded
within-session shuffling. Known classes occur throughout; seen novel classes
reappear via different observations. Counts must be divisible across sessions;
known samples per class must cover all sessions. No synthetic path reads CUB or
imports the SSB reader. Python and NumPy global RNG states are not modified.

## Stream metrics, probe metrics and identity

Phase 2 records each stream prediction at arrival, before its EMA update; a new
cluster can be created for that observation before its ID is emitted. At session
end, the evaluator scores these recorded predictions without re-predicting the
stream under the final bank. This differs explicitly from Phase 1's proposed
post-session transductive scoring. Probes use the bank **after** the session.

Session-local Hungarian accuracy fits one maximum-count one-to-one mapping on
all that session's recorded predictions. Old and New accuracy reuse the same
mapping. Cumulative Hungarian accuracy fits one mapping on all recorded
predictions through the current session, retaining their persistent IDs; it does
not concatenate independently remapped session labels or recompute old predictions.
Unmatched clusters and rejection `None` are incorrect. SciPy matching uses
sorted true labels and predicted IDs for deterministic solver tie resolution;
dependency versions are recorded. This is a concrete tie policy, not a promise
of lexicographically minimal optimal assignment across solver versions.

Final-session accuracy means the last session's local All accuracy. Average
online accuracy is the arithmetic mean of session-local All accuracies, not the
sample-weighted cumulative accuracy. NMI (arithmetic normalization) and ARI are
computed on novel observations, both locally and cumulatively; fewer than two
novel observations or fewer than two true novel classes gives null. If a caller
supplies rejections, clustering diagnostics treat all `None` as one rejection
category; accuracy still counts each rejected observation incorrect.

Independent Hungarian matching can hide ID permutations. In addition to
cumulative matching, the evaluator maintains a **frozen identity mapping**:

- At base time, `known-<c>` maps to authorized known label c.
- At each session, only unmapped predicted IDs and as-yet-unmapped eligible
  classes can be paired, using positive-support Hungarian matches on that
  session's stream labels. Existing mappings are never revised.
- Calibration uses no probes or future samples. No mapping is returned to the
  learner. Cumulative fixed-mapping accuracy is separately reported using the
  current frozen mapping and historical raw predictions. A newly established
  mapping can explain earlier unmapped predictions in that diagnostic; past
  session reports and past probe scores are never rewritten.

Probes are fixed, disjoint held-out observations. At base session 0, only known
probes are eligible. After online session t, evaluate known classes plus novel
classes with introduction <=t. No future-class feature is passed to prediction.
Probe accuracy uses the frozen identity mapping, never probe Hungarian rematching.
The evaluator checks that the entire learner JSON state remains unchanged by
probe inference. Probe labels/results do not tune thresholds, select models,
update prototypes or determine the stream.

For each previously eligible class c, forgetting at t is exactly
`max_{u < t, c eligible at u} probe_accuracy(c,u) - probe_accuracy(c,t)`.
Average equally across previously eligible classes; newly introduced classes
are excluded until they have a previous score. Known histories start at session 0.
Negative values represent improvement and are retained in `average_forgetting`;
`average_forgetting_clipped` separately averages max(0, difference), maintaining
the Phase 1 nonnegative diagnostic. No eligible previous class means null.
Identity churn is the class-macro fraction of fixed probes whose raw persistent
ID changes since the last session; rejections are included as `None`.

For true class c, let f(c) be the number of distinct non-rejected persistent IDs
assigned to its observations. Fragmentation is mean_c max(0,f(c)-1), alongside
the fraction of classes with f>1 and per-class counts. `duplicate_cluster_count`
is sum_c max(0,f(c)-1): **class-cluster excess incidences**, not a count of unique
bank entries (an impure cluster may contribute to several classes). Report these
locally and cumulatively. They detect fragmentation, not impurity alone; use
NMI/ARI, accuracy and count diagnostics together.

Active/known/discovered-novel counts describe bank entries and their origin,
not the number of correctly discovered true classes. All created entries remain
active. Evaluator histories may be retained for metrics; they are never replayed
to the bank. The evaluator and orchestrator are trusted components, not a
process-level security sandbox.

## Learner state and result schemas

Learner state is strict JSON, schema version 1. Top-level keys are
`schema_version`, `baseline_version`, `feature_dim`, `threshold`, `momentum`,
`next_novel_id`, `last_session`, and `prototypes`. Each prototype contains
`cluster_id`, `vector`, `observation_count`, `originated_known`,
`creation_session`, and `last_updated_session`.

State loading validates exact keys, versions, dimension, finite unit vectors,
unique/canonical IDs, origin/session consistency, counts, continuous novel ID
sequence and next-counter consistency. It rejects duplicate JSON keys and
nonfinite JSON numbers. It does not renormalize saved vectors or silently repair
metadata. Known labels may appear in known IDs; no novel labels, evaluator maps,
schedule, total novel count, probe history or sample features are serialized.
JSON writes are sorted UTF-8 with LF, reject NaN, and refuse existing files.

Save/load preserves exact vector values, metadata and future ID allocation.
`load_state(path, expected_feature_dim=...)` can enforce an external dimension
contract. Midstream learner continuation and final read-only inference are
supported. Full evaluator/run resume is **not** implemented: it would also need
the privileged evaluator history and delivery cursor, kept separate from learner
state. Loading the final bank and replaying the whole stream is not an equivalent
run and is intentionally not a CLI mode.

Result JSON is evaluator-only, schema version 1:

| Key | Meaning |
|---|---|
| `schema_version`, `audience`, `baseline_version`, `dataset` | Version 1, evaluator_only, euclidean-prototype-v0, synthetic. |
| `configuration` | Full synthetic config including builder-only class counts. |
| `synthetic_checksum`, `stream_manifest_checksum` | SHA256 of generated populations/features and the Phase 1-format stream payload. |
| `environment` | Python, NumPy, SciPy, scikit-learn, PyYAML versions. |
| `base_probe` | Known-only base accuracy and history initialization. |
| `sessions` | Ordered per-session objects described below. |
| `summary` | Final-session and average-online accuracy, final cumulative/probe accuracy, forgetting, churn, cumulative novel NMI/ARI and fragmentation, bank origin counts. |

Each session has `session_id`, `session_local`, `cumulative`,
`cumulative_fixed_mapping_accuracy`, `probe`, and the three prototype counts.
Local/cumulative reports contain `hungarian_accuracy` (all/old/new), novel NMI,
ARI and sample count, predicted cluster count, rejection count and fragmentation.
Every accuracy slice is `{value: fraction_or_null, count: integer}`.
Probe reports contain fixed-mapping accuracy slices, per-class accuracies,
signed/clipped average forgetting, previous-eligible class count, identity churn
and eligible class count. Class keys in JSON are strings. No field uses a
percentage except the human-readable console table; forgetting there is in
percentage points. Results intentionally omit nondeterministic timestamps and
wall-clock timings. Matching environment/config yields byte-identical artifacts;
cross-library/platform floating-point equality is not guaranteed.
Archive the code commit alongside these outputs for the full reproducibility record.

## Running and validation

Use the existing environment; no new dependencies were added. From the repo
root on Windows:

```powershell
.\.venv\Scripts\python.exe scripts/run_synthetic_online_baseline.py --config configs/online_selex/synthetic_prototype_v0.yaml

# outputs/ is already Git-ignored. Create it once if needed.
New-Item -ItemType Directory -Force outputs
.\.venv\Scripts\python.exe scripts/run_synthetic_online_baseline.py --seed 1 --output-json outputs/synthetic_seed1.json --save-state outputs/synthetic_seed1_state.json

.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_online_*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/compare_reproduction_config.py
```

Use `python` from the configured environment on other systems. Output paths must
be distinct, new files in existing directories. Test artifacts use temporary
directories and are removed; generated results/states are never tracked.

## Limitations and exact Phase 3 decisions

This intentionally easy, orthogonal Gaussian setup has no semantic domain shift
or representation learning. Unbounded prototype growth is possible; no pruning,
merging, split handling, replay or resource-budget mechanism is implemented.
Thresholds/momentum are synthetic settings, not validated CUB hyperparameters.
Frozen mapping and churn make identity failures visible but do not solve them.

Before real CUB evaluation, Phase 3 must resolve:

1. **Clean feature initialization:** choose a generic pretrained encoder with
   documented pretraining exposure, or a known-only fitted encoder. Do not use
   full offline SelEx as clean initialization. Any warm-start ablation is named
   separately and records novel-image/count exposure.
2. **Populations:** freeze labelled-known base IDs, online image pool and disjoint
   held-out probe IDs; archive overlaps and split hashes without reusing probes
   for tuning. Decide whether labelled-base images may reappear online.
3. **Feature contract:** specify backbone/layer, feature dimension, normalization,
   preprocessing, frozen weights hash and class-independent feature lookup IDs.
4. **Label-safe tuning:** select threshold/momentum using known validation or
   separately held-out development classes; no novel test labels or probe scores.
5. **Unknown-K and resources:** retain no oracle novel count by default; declare
   prototype growth limits, replay budget if introduced, and runtime/memory
   measurements before comparing methods.
6. **Evaluation:** retain or explicitly version arrival-time stream scoring,
   frozen probe mappings, ID lifecycle and calibration/tie policy; choose seeds,
   stream variants and a comparable clean real-feature baseline.
