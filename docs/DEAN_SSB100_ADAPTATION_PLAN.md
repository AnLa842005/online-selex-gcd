# Future DEAN SSB 100/100 adaptation

Status: **NOT RUN / execution disabled**. This is a separate experiment from
official DEAN CUB. Neither SSB compliance nor performance is established.
Draft contract: [cub_ssb100_v0.yaml](../configs/dean/cub_ssb100_v0.yaml).

## Three independent layers

1. **External DEAN core** owns offline/online model and head, PET/LoRA, EGD, VFA
   and DEAN losses. DEAN-0 executes the original entrypoint externally. Future
   extraction into callable core requires license clarification/permission first;
   do not copy the monolith into this repository. Preserve official semantics in
   official mode; each adaptation difference receives a new config version.
2. **Dataset/protocol adapter** owns official loader invocation or, separately,
   causal SSB manifest consumption. `OfficialProtocolAdapter` supplies audited
   launcher flags. `SSB100ProtocolAdapter` is deliberately non-executable until
   the gates below pass. The future `LearnerBatch` contract contains opaque IDs,
   resolved tensors and current session only. It contains no labels, class-bearing
   paths, total novel K, probe membership or future-session metadata. Base known
   labels require a separate explicitly known-only initialization interface.
3. **Evaluation adapter** receives prediction IDs and sample IDs, while keeping
   labels and Hungarian mappings private. `EvaluationAdapter` is a contract only,
   not a completed official-to-project metric conversion. Official console
   metrics stay in the official namespace. Project All/Old/New, fragmentation
   and signed/clipped forgetting use current evaluator definitions and persistent
   prediction-ID semantics, without changing learner decisions.

## Protocol gates

- Reuse the immutable SSB split/hash, global 50% known-labelled subset, five-session
  20-class introduction schedule, causal ordering and current manifest integrity.
  Use `manifest.learner_session(t)`; never pass the full evaluator manifest to core.
- Remove true-label diagnostics from any learner-side interface, move them to
  evaluator storage and prove shuffled labels leave learner updates unchanged.
  Do not pass `dataset.new_inc`, `whole_test_set` or `online_seen_classes` derived
  from labels into the learner. The small data contract alone is not such a proof.
- AP may estimate novel K from current permitted features. No true novel count,
  unseen future categories, fine-split oracle or probe labels are allowed.
- Replace test-based base checkpoint selection with a predeclared known-only
  training validation split for the adaptation. Disclose this difference;
  official mode keeps its own selection rule unchanged.
- Decide batch size/current-batch 15-epoch reuse/VFA budget explicitly before any
  results. These differ from the frozen single-sample one-pass control. Do not
  label the resulting comparison as matched exposure without accounting for it.
- Define tiny/empty GMM splits, K estimates exceeding sample count and numerical
  loss behavior through separate reviewed changes. Do not silently repair official.
- Persist online model/head/PET, optimizer, RNG and stream cursor if exact resume
  is required. Preserve stable discovered IDs and evaluator state isolation.
- Probe is evaluator-only. No threshold/hyperparameter tuning on probes, novel
  GT, future sessions or test metrics. State hashes before/after probes must agree.

## Acceptance before implementation/real execution

Require license decision for extraction, completed dataflow audit, separate
official/SSB artifact provenance, checkpoint key equivalence, deterministic fixture
tests for core/adapter isolation and label permutation, resource measurements,
frozen protocol budgets, and unchanged lower-bound/Phase 4A/offline regression suite.
Do not infer a learner's unknown-K status merely because the CLI omits true K.

The main path is official DEAN reproduction, then a separately reviewed SSB
adaptation. Frozen hierarchy-only Phase 4B and hyperbolic Phase 5 are deferred
ablations. No hierarchy, USE, SSE, modified losses or tuning belongs in DEAN-0.
