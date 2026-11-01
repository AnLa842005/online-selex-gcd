# Online SelEx Phase 1: online-selex-cub-v0

Status: protocol and metadata stream infrastructure only. Online SelEx has **not**
been implemented or evaluated. No model, training, feature extraction, replay,
prototype learning, hyperbolic mapping, new losses or GPU evaluation is included.

## Frozen offline reference

- Baseline commit: `a52cf64dcbc43a57069c708ce0ce616cdd045221`.
- Annotated tag: `selex-cub-baseline-v1`.
- Tag message: `Reproduced SelEx CUB DINOv1 baseline: 73.95 All-v2`.
- Final checkpoint, seed 1: All-v2 **73.95%**, Old-v2 **71.45%**, New-v2
  **75.21%**; **PASS WITH QUALIFICATION**.
- Evidence and qualifications: [audit](../REPRODUCTION_AUDIT.md),
  [setup](../BASELINE_SETUP.md), [tracked results](../results/cub_dinov1_seed1.json).
  These recorded metrics are not online results and have not been rerun here.

The base state comes from this reproduced offline model. Original labelled-known
examples may initialize known-class information. Novel ground-truth labels must
never initialize or update the learner. Offline SelEx already used unlabelled
novel training images and a configured total class count. Consequently, v0 is an
**offline-preexposed warm-start protocol**, not evidence of discovery by a model
that has never encountered these classes. "Unseen" below refers only to stream
history. Before a real experiment, record exactly which checkpoint components
are reused; discard offline novel cluster assignments, cluster banks, hierarchy
cardinalities and evaluation mappings from the learner interface. A backbone
still retains representation-level preexposure, which cannot be undone here.
An unknown-K claim concerns new online inputs, not the offline model's provenance.

## Fixed protocol decisions and terminology

Dataset: CUB-200-2011 with the repository's existing SSB split in
`data/ssb_splits/cub_osr_splits.pkl`, inherited from upstream SelEx at
`c3d3b42a0aa7d7c9ea748a8305a6890a34100f73`. SHA256:
`48bdcaa05603d932037738d3df776ab500945e9b7aeff85f4668305bf71bd1e7`.
The metadata reader preserves `known_classes` and novel `Hard + Medium + Easy`
ordering exactly as `data/get_datasets.py`. There is no new random known/novel
partition. Assert 100 known and 100 novel IDs, disjoint, covering IDs 0..199.
These are original zero-based CUB IDs (official labels minus one), **not** the
training loader's known-first target remapping. Synthetic tests may use other
counts; the CUB validation command always enforces the real split.

| Term | Definition |
|---|---|
| Known / old class | A member of the fixed 100-class SSB known set K, regardless of whether a particular observation is labelled. |
| Novel / new class | A member of the disjoint 100-class SSB novel set U; it remains novel even after online introduction. |
| Introduction session i(c) | The single online session assigned to novel class c by the seeded schedule. |
| Newly introduced novel at t | Classes with i(c) = t. Each must have at least one observation in t. |
| Seen novel at t | Previously introduced classes with i(c) < t, excluding newly introduced classes. After processing t, the seen set includes i(c) <= t. |
| Unseen novel at t | Classes with i(c) > t; none may be supplied in or before session t. |
| Base session | Session 0, initialization from the offline model and permitted original known-labelled subset; it is not generated or counted in online averages. |
| Online session | One chronological finite batch of observations, delivered once, followed by evaluation of the resulting state. No next-session lookahead. |
| Session ID | Integer 1..T for online sessions; 0 reserved for the base state. |
| Persistent predicted cluster ID | Learner-issued identifier with stable meaning across sessions, never silently renumbered or recycled. Merges/splits must record lineage and retired IDs. It is not a true class label or Hungarian index. |

The default configuration is [cub_5sessions_seed1.yaml](../configs/online_selex/cub_5sessions_seed1.yaml):
T=5, seed=1, exactly 20 newly introduced novel classes per session. A local
`random.Random(seed)` shuffles the existing novel list, then assigns consecutive
groups to sessions. Known classes occur in every session; introduced novel
classes may reappear. Each session can therefore mix known, previously seen
novel and newly introduced novel observations.

## Deterministic sample allocation

`online_gcd/stream.py` accepts `SampleRecord(sample_id, class_id, path=None,
is_labelled_known=False)`. IDs must be unique, stable, opaque strings independent
of class names (for CUB, a string derived from official image ID is appropriate).
Input paths and labels belong to the trusted builder/evaluator. The caller must
correctly identify the original labelled-known subset; do not resample that subset.

The default `sample_allocation: round_robin` policy is:

1. Validate class lists, configuration, records, unique IDs and known-only labels.
2. Create one local allocation RNG seeded from the configured seed, separate
   from the schedule RNG. Iterate classes in supplied known order followed by
   supplied novel order. Within each class, sort samples lexicographically by
   sample ID and shuffle using this allocation RNG. Metadata changes cannot
   alter the class introduction schedule.
3. Known samples are eligible for sessions 1..T. Novel samples are eligible for
   i(c)..T, or only i(c) when seen-novel reappearance is disabled.
4. Assign shuffled sample j to eligible session `j % number_of_eligible_sessions`.
   A declared class must have at least one sample. When known coverage is enabled,
   every known class needs at least T samples; fail instead of duplicating them.
   With fewer novel samples than eligible sessions, fill the earliest sessions;
   recurrence is allowed, not guaranteed when metadata is sparse.
5. Shuffle each completed session in ascending session order using the same
   allocation RNG, then freeze the result into tuples.

Each original input sample appears **exactly once**. Reappearance means another
sample of an already introduced class, not repeated delivery of an image. No
implicit replay, oversampling, balancing between classes or sample dropping is
performed. Natural per-class totals are retained, with near-even allocation
across each class's eligible sessions. This induces changing novel-class sample
densities toward the end and must be considered when interpreting scores.

An implemented configurable alternative, `contiguous_chunks`, splits each
shuffled per-class list into consecutive chunks over eligible sessions: size
floor(n/m), with one extra for the first n mod m sessions. It preserves the same
coverage and no-duplication rules. Explicit repeated observations with separate
observation IDs are a future protocol extension, not supported by v0.

The full manifest is evaluator-only. It includes configuration, seed, protocol
version, introduction schedule, session sample IDs and evaluator records with
labels and paths. The checksum is SHA256 over UTF-8 JSON of the entire payload
excluding `manifest_checksum`, with sorted keys, separators `(',', ':')` and
`ensure_ascii=True`. Export uses sorted, indented JSON with LF and a trailing
newline, and refuses to overwrite files. Input iteration order does not affect
the result; supplied class list order, seed, config and metadata do. Record the
Python version as well as the checksum for cross-environment reproducibility.

## Information boundary and state

Only `manifest.learner_session(t)` may be handed to the learner, after the trusted
orchestrator has reached t. It contains only `sample_id` and `session_id` by
default. An explicit `expose_labelled_known_membership: true` adds a boolean for
membership in the original labelled-known subset; it never exposes a class ID.
Future initialization may obtain known labels from a separate authorized base
subset, not from the online manifest.

Images or precomputed features must be resolved by a trusted adapter from opaque
IDs. Paths are retained only in evaluator records because CUB filenames and
directories encode class names. A future path interface must use sanitized,
class-independent paths. Arbitrary input IDs can also encode labels; the builder
cannot infer their semantics, so opaque-ID provenance is a required caller check.

Do not pass the full manifest, builder configuration, labels, paths, future
records, introduction schedule or cluster-to-class evaluation mappings to the
learner. The builder needs the total novel count, but the learner must not receive
it unless `reveal_total_novel_class_count: true` is explicitly declared as an
oracle experiment. Even then, only a trusted future runner may pass the scalar
count; the learner view never expands to include the schedule or labels.
This Python data projection is not a process sandbox: access control and
chronological delivery are responsibilities of the future runner.

Across sessions, a future method may carry encoder/model parameters, projection
or classification head, prototype/cluster bank, hierarchy state, persistent
predicted IDs, budgeted replay memory, and optimizer state if online training is
enabled. Hidden true labels and future data are never learner state. Unlimited
historical samples require an explicitly named offline/oracle comparison;
they are prohibited in default online runs. The current budget is **0 stored
historical observations**, encoder frozen, replay disabled. Persistent model
and cluster state must still be measured in bytes separately from replay count.
Evaluator-only diagnostic storage is reported separately and never read back by
the learner. Phase 1 implements none of this learner state.

## Evaluation contract (definitions, not implemented evaluators)

Default reporting is after processing session t, on that session's observations
S_t: a post-session transductive score. Do not mix it with pre-update/prequential
scores. Let y be the evaluator label and z the persistent predicted ID. Fit one
maximum-weight one-to-one Hungarian assignment H_t between z and y on **all**
S_t, with unmatched predicted IDs counted incorrect. Reuse that same assignment
for every accuracy slice, matching the offline v2 convention. No per-subset
rematching for the primary report. If predictions can reject samples, count
rejections incorrect and report their rate; never silently exclude them.

| Measure | Definition |
|---|---|
| All accuracy A_t | Correct predictions under H_t divided by number of observations in S_t. |
| Old / known accuracy | Same numerator/denominator restricted to y in K. |
| New / novel accuracy | Restricted to y in U, including new and previously introduced classes. |
| Newly introduced accuracy | Restricted to i(y)=t. |
| Seen novel accuracy | Restricted to i(y)<t. |
| Average online accuracy | Arithmetic mean of A_t across T sessions; report sample-weighted aggregate separately with its name. |
| Final-session accuracy | A_T on S_T; not cumulative accuracy on all historical observations. |
| Novel NMI / ARI | Normalized mutual information (arithmetic entropy normalization) and adjusted Rand index between raw z and y on novel observations in S_t. No Hungarian mapping needed. |
| Predicted cluster count | Number of distinct IDs emitted in S_t, and separately active bank IDs and cumulative distinct emitted IDs. Report all and novel-subset counts. |
| Fragmentation / duplicate clusters | For each true class c, f(c,t) = number of predicted IDs assigned at least one observation of c in the evaluation population. Report class-macro mean max(0,f(c,t)-1), fraction with f>1 and per-class counts, on S_t and cumulatively. This detects fragmentation but not impure merged clusters; accompany it with NMI/ARI and cluster count. |
| Memory | Peak host RSS, accelerator allocation if used, and persistent learner state bytes split into model/optimizer, bank/hierarchy and replay. Report stored observation count and evaluator memory separately. |
| Processing time | Wall-clock seconds from session delivery through final state/predictions; separately report input/feature time and evaluator time, hardware and batch size. Synchronize accelerators if a future experiment uses them. |

An empty accuracy subset (e.g. seen novel at t=1) is N/A with count 0, never zero
accuracy. For NMI/ARI require at least two novel observations and at least two
true novel classes; otherwise report N/A. Online averages exclude N/A slices and
report the number of contributing sessions. No such metrics are computed in Phase 1.

**Per-session Hungarian matching alone cannot demonstrate stable discovery.** A
learner can exchange every predicted ID each session without lowering those
scores. Preserve raw IDs and report the following evaluator-only diagnostics:

- Frozen identity mapping: initialize mappings for known predicted IDs using
  authorized base-known anchors at t=0; at t assign only previously unmapped
  predicted IDs to as-yet-unmapped classes introduced by t, using current labels
  in the evaluator. Use one-to-one matching with deterministic tie-breaking by
  true class ID then predicted ID. Never revise a historical mapping. Later
  fragmentation IDs remain unmatched when their class already has an ID. Report
  accuracy under this frozen mapping separately from H_t. No future samples or
  unseen-class labels may participate in calibration.
- Identity churn: on a fixed evaluator-only probe set Q_c for each previously
  learned class, measure the fraction whose raw predicted ID differs between
  states t-1 and t. Macro-average over classes, retain class-level values, and
  record merges/splits and lineage separately; do not erase these changes by
  rematching IDs. Define no-history comparisons as N/A.
- Forgetting: fix disjoint held-out probes Q_c before running. Known classes are
  eligible from t=0, novel classes from i(c). Let a(c,t) be probe accuracy under
  the frozen identity mapping. For previously learned c at t, compute
  `max(0, max_{u in [start(c), t-1]} a(c,u) - a(c,t))`, then macro-average across
  those classes. Newly introduced classes are excluded. Also report the signed
  difference if improvement is of interest. Never estimate forgetting by
  comparing different session samples or separately rematched accuracies.

The probe population, checkpoint preexposure overlap, base anchors and calibration
rules must be archived before evaluation. Probes are read only by the evaluator,
never used for optimization, thresholds or early stopping. Evaluator inference
must not update learner state. Until a disjoint probe population is specified,
forgetting and probe churn are **not available**, not inferred from stream scores.

## Configurable experiments and unresolved research questions

These choices are experimental factors, not claims about the final method:

| Factor | Phase 1 default / future variation |
|---|---|
| Sessions | 5 x 20; 10 x 10 can be configured, with matching counts. |
| Stream composition | Mixed known/seen/new; strict class-incremental needs a separately declared policy. Disabling novel recurrence alone still mixes known classes. |
| Sample ordering | Deterministic seeded shuffle; alternative chronological/adversarial orderings require explicit implementation and config. |
| Sample allocation | Round robin; contiguous chunks implemented as an alternative. |
| Class balance | Preserve natural counts; class-balanced sampling is future work and must document discarded/repeated observations. |
| Memory / replay | 0 observations / disabled; future declared budgets and replay policies. |
| Encoder | Frozen; partial training is deferred. |
| Novel count | Hidden online; declared oracle count may be a comparison. |
| Prototype threshold | Not defined or used; requires synthetic validation and label-safe tuning later. |

The configuration parser rejects unsupported method settings rather than
pretending replay or encoder updates are implemented. Changing supported stream
choices creates a distinct experiment recorded by config/hash, even under v0.

Open decisions before real CUB experiments: which image population supplies the
stream (e.g. training unlabelled versus official held-out images), whether original
labelled samples appear online, how to reserve disjoint probes, checkpoint hash
and component sanitization, adaptation versus prequential reporting, unknown-K
initialization without offline count hints, merge/split lineage semantics, and
memory/threshold selection. The generator accepts explicitly curated metadata;
it does not silently choose a CUB image partition. Do not describe training-pool
reuse as first exposure. A strict no-preexposure experiment needs a different
base-training design, outside this phase.

## No data leakage checklist

- [ ] Use the pinned SSB split, original zero-based IDs and original labelled subset.
- [ ] Validate disjoint 100/100 classes, one introduction per novel class and no early samples.
- [ ] Validate unique opaque sample IDs; no class names in learner IDs or paths.
- [ ] Deliver only the current learner view plus sanitized images/features, never the full manifest/config.
- [ ] Novel labels, introduction schedule, future observations and evaluation mappings remain privileged.
- [ ] Document online K oracle status and offline checkpoint exposure/count information separately.
- [ ] Do not retain hidden labels or unlimited historical observations in learner state.
- [ ] Declare replay budget, state bytes and any explicitly repeated observations.
- [ ] Keep probes disjoint from learner inputs; no evaluator feedback for tuning.
- [ ] Preserve predicted IDs and report frozen-mapping accuracy, churn and fragmentation.

These are experiment-run checks; Phase 1 mechanically enforces metadata and
projection invariants, but cannot attest to an unimplemented learner's behavior.

## Reproducibility record

Every future run must archive dataset/version and image partition; SSB source,
file hash and ID convention; protocol version; seed; session count; full class
introduction schedule; sample ordering/manifest SHA256; checkpoint SHA256 and
reused components (currently unavailable, **not fabricated**); offline exposure
provenance; exact configuration snapshot; code commit; Python and dependency
versions; metadata/probe hashes and population overlap; hardware; ordering and
allocation policy; oracle settings; and evaluation/mapping version. Pin exported
manifests before execution. The builder JSON supplies the construction subset
of this record, not checkpoint/environment provenance for a future training run.

## Commands

From the repository root, using the existing environment (Python 3.11 and
PyYAML 6.0.3 at Phase 1 validation). On Windows use
`.\.venv\Scripts\python.exe` in place of `python` below. The stream module itself
is standard-library-only; loading YAML requires the already available PyYAML.

```bash
# Split and schedule only: no images, synthetic samples or model execution.
python scripts/validate_online_protocol.py

# Use an existing output directory; outputs/ is already ignored by Git.
python scripts/validate_online_protocol.py --output outputs/cub_schedule.json

# Full stream from evaluator-supplied metadata; never reads the listed images.
python scripts/validate_online_protocol.py --metadata /path/to/metadata.json --output outputs/cub_stream.json

python -m unittest discover -s tests -p "test_online_*.py" -v
python -m unittest discover -s tests -v
python scripts/compare_reproduction_config.py
```

Metadata is a JSON array of objects with `sample_id` (opaque string), `class_id`
(original zero-based integer), optional `path` (string/null), and optional
`is_labelled_known` (boolean, default false). All declared classes need metadata;
known classes need at least one distinct sample per session. Without `--metadata`,
the JSON has `manifest_kind: schedule_only` and empty sample lists: it is **not**
a CUB sample stream and does not claim sample coverage. The CLI reports this mode.
Generated artifacts belong in ignored `outputs/` or outside the repository;
temporary test artifacts use automatically cleaned temporary directories.

## Explicitly deferred phases

Phase 2 will implement a **Euclidean frozen-feature prototype baseline on
synthetic embeddings before using real CUB features**. Prototype updates,
threshold selection, persistent cluster lifecycle, metric implementations and a
chronological runner are future work. Replay, hierarchical self-expertise,
forgetting interventions, encoder updates, new losses, hyperbolic representations,
training, feature extraction and GPU/real-CUB evaluation remain out of scope.
