# Real SelEx integration into DEAN: source audit and proposed online design

Audit date: 2026-11-05. Status: **REVIEWED V1 DESIGN / APPROVED FOR IMPLEMENTATION**.
The current `dean_integration/selex_online.py` is synthetic scaffolding. Its
nearest-prototype parent CE/cosine losses are not the research USE/SSE and must
not be used to claim a SelEx result.

The infrastructure was reviewed, all 232 tests passed again in 161.009 seconds,
and exactly 12 previously reported DEAN infrastructure files were committed as
`d1c7a030490d16fc9648ea94953e6f123771f6f4` with message
`feat(dean): add deterministic paired online experiments`. The branch
`experiment/dean-selex-online` was pushed to origin; the remote ref was checked
against that SHA. No merge occurred. This audit document was created afterward; the reviewed v1 revision is committed separately before implementation. No implementation, dataset, checkpoint, baseline
artifact or upstream source was edited during the audit. No GPU training ran.

## 1. Authority, revision and reproducible evidence

Authoritative repository: [SarahRastegar/SelEx][repo]. GitHub's repository metadata
identifies it as the authors' official ECCV 2024 repository. Its `main` was
verified through the connected GitHub API at
`c3d3b42a0aa7d7c9ea748a8305a6890a34100f73`. Local `upstream/main` matches.
The audit reads **Git blobs at that revision**, not the locally modified SelEx
training entrypoint. The remote recursive Git tree's blob IDs matched the local
objects below; no checkout, fetch, source copy or modification was needed.

| Authoritative file | Git blob ID |
|---|---|
| `methods/contrastive_training/contrastive_training.py` | `07a0df7fed61d5ad688e2307701e0e0e92d80bf0` |
| `methods/clustering/faster_mix_k_means_pytorch.py` | `0444e1887c5773f3494a5a665bb195677a74f56e` |
| `models/vision_transformer.py` | `5d4e14caaa380a5dc5d26affbd8ee145d2623a63` |
| `data/data_utils.py` | `d05d7a10af60cfdb42ae606ffefa28a87e81bed8` |
| `data/get_datasets.py` | `74fac7038019f662346b846b904a67e082ad022b` |
| `data/augmentations/__init__.py` | `3988aadb4f6390d10f15fdbc161063089df660dc` |
| `README.md` | `6f51814fd6b6aa09a142cc068b7059ab2fb984fa` |

DEAN source remains the verified external checkout at
`28b9384cd7ea9b123b03d045bac2e92d8d0b2870`. All DEAN line numbers below refer
to that revision. All SelEx line numbers refer to the authors' revision above,
and differ from the local working repository's modified offline trainer.

The [author README, lines 22–25][notice] says that order sensitivity is under
investigation. [Issue 11][order-issue] reports the coarse-label order mismatch.
This audit independently traces the relevant code; it does not independently
reproduce that issue's reported accuracy numbers.

The [paper, arXiv v1][paper] is a conceptual cross-check: equations 10–11 concern
hierarchical targets and smoothing, equation 12 describes BCE, equation 13
defines weighted prefix supervision, and equation 14 mixes USE/SSE. The runtime
uses soft-target cross entropy, radius-based targets and the implementation
details below. Therefore paper notation is not sufficient for code parity.

## 2. Authoritative implementation map

| Component | Source and exact lines | Runtime meaning |
|---|---|---|
| Two image views | [trainer 151–159][views]; [augmentations 14–23][augment]; trainer 765–766 | Two independent augmentations of each real image; views are concatenated in view-major order |
| DINO embedding | [ViT 211–219][vit-forward], 257–261 | ViT-B/16 normalized-transformer CLS token, dimension 768; no per-row L2 normalization in `forward` |
| Projection | [DINOHead 264–298][head]; trainer 737–741, 801–807 | MLP 768→2048→2048→256, bottleneck L2 normalization, weight-normalized linear 256→65536 |
| Hierarchy/HSSK orchestration | [trainer 502–592][hssk] | `extract_labeled_protos`; finest semi-supervised clustering, then progressively reduced known/novel counts |
| BSSK / semi-supervised K-means | [K_Means 174–303][bssk], 329–345; seeding 87–124 | Labelled center initialization, K-means++ novel initialization, capacity reassignment, final unconstrained refinement |
| USE geometry/targets | [trainer 287–352][use-targets] | Detached backbone features, hierarchy-dependent radii/confusion, smoothing |
| USE logits and loss | [trainer 162–204][use-logits], [42–50][smooth] | Euclidean contrastive logits, self removal, paired positive first, soft-target CE |
| SSE kernel | [trainer 55–148][supcon] | Euclidean supervised contrastive loss with an explicit positive mask |
| SSE inputs/slicing | [trainer 354–364][sse-compose] | Full projection on truly labelled examples, plus progressively shorter prefixes on all examples with hierarchy labels |
| Total loss/backprop | [trainer 366–380][se-total] | Convex USE/SSE mixture; one backward through projection and unfrozen backbone |
| Hierarchy update timing | [trainer 236–268][refresh]; defaults 674, 681 | Epoch start under `no_grad`, interval 1 by default; radius momentum default 1 |

There is no independent `USE()`/`SSE()` or named `HSSK` class in this trainer.
Those paper components are composed from the functions and call sites above.

## 3. Complete offline data flow and gradient boundary

1. `get_datasets` concatenates the genuinely labelled subset first and the
   unlabelled subset second ([dataset builder 73–91][datasets]). Membership
   produces `mask_lab`; it is not inferred from a predicted known-class ID
   ([MergedDataset 25–37][merged]). The weighted sampler draws training batches;
   the hierarchy loader iterates the complete merged dataset without shuffle
   (trainer 780–792).
2. At a refresh, `model.eval()` and `torch.no_grad()` extract the first augmented
   view of every training image, normalize the 768-D backbone feature and fit
   the hierarchy. This uses the whole offline training set, including examples
   that would be future observations in an online protocol. It must not be
   copied literally into a causal stream.
3. The training batch has B images and two views. `model(images)` produces
   `z` of shape `[2B,768]`. A detached copy supplies geometry only. The live
   tensor goes through `DINOHead`, then outer L2 normalization, yielding
   `q` of shape `[2B,65536]` (trainer 272–286).
4. Hierarchy IDs and detached `z` produce confusion weights. USE differentiates
   contrastive distances in `q`, not assignments, centers, radii or targets.
5. SSE differentiates distances between `q` or its coordinate prefixes. Truly
   labelled examples contribute full-dimensional GT supervision; all examples
   contribute hierarchy-based prefix supervision.
6. The combined scalar backpropagates through `q -> projection -> z -> trainable
   backbone blocks`. The original SGD optimizer includes both model and head
   parameters (trainer 208–214). Backbone parameters are frozen first, then
   blocks from `grad_from_block=10` onward are unfrozen (trainer 746–758).
   This is not DEAN's LoRA-only training policy.

## 4. Equations as executed, with source correspondence

Notation: B is the image count, M=2B the view count, `a+` the other view of
image a, H the number of stored hierarchy levels, and h_i(a) its detached
level-i assignment, repeated for both views. Level 0 is the finest stored
pseudo-label level. All distances below are Euclidean, not squared Euclidean,
unless explicitly identified as K-means distances. These equations audit the
default Euclidean path with two views and `is_code=False`.

### E1. Geometry and learned representation

```text
z_a = F_theta(T_a(x)),                         z_a in R^768
zbar_a = normalize(z_a)                      [hierarchy extraction only]
b_a = normalize(MLP_phi(z_a)),                b_a in R^256
q_a = normalize(W_weight_norm b_a),           q_a in R^65536
q_a^(i) = q_a[0 : floor(65536 / 2^(i+1))]
```

Sources: [ViT 211–219][vit-forward]; [head 294–298][head];
[trainer 278–286][refresh]; [trainer 518–519][hssk]; [trainer 361–364][sse-compose].
The hierarchy uses normalized backbone features; the USE batch geometry uses
detached **unnormalized** backbone features. These are different scales.
SSE prefixes are **not renormalized after slicing**. USE additionally calls
`F.normalize` again at trainer line 169. The supervised kernel does not do it.

### E2. BSSK centers, assignments and hierarchy counts

```text
mu_c^init = mean(zbar_j : labelled y_j=c)
D_jc = sum_d (zbar_j[d] - mu_c[d])^2
h_j = argmin_c D_jc                         [unlabelled rows]
mu_c = mean(zbar_j : current h_j=c)         [when that center is updated]
J = sum_unlabelled_j min_c D_jc + sum_labelled_j D_j,y_j
capacity_i = ceil(N / (K_known_i + K_novel_i))
K_known_(i+1) = max(floor(K_known_i/2), 1)
K_novel_(i+1) = max(floor(K_novel_i/2), 1)
```

Sources: [K-means distances 10–46][kdist]; [BSSK 181–226, 254–301][bssk];
[trainer 538–590][hssk]. K-means++ seeds additional centers from unlabelled
features using probabilities proportional to the squared distance to their
nearest existing center (K_Means 87–124). During capacity handling, farthest
excess unlabelled points are reassigned toward sparse centers (229–252).
The final loop (270–301) drops capacity enforcement and updates all centers.
Keep the distinction between fixed labelled assignments and centers that can
move during final refinement. Empty-center arithmetic is not guarded.

Fine known prototypes are clustered by sklearn KMeans with `random_state=0`;
their grouping creates higher-level anchor labels, then BSSK produces the
higher-level assignments (trainer 575–590). The source loop stops when the
known count reaches one, not when the novel count reaches one. `label_proto`
and `all_preds` remain the original finest tensors inside that loop; it does
not update them to each previous higher-level result. Thus do not assert that
the literal implementation produces a strictly nested tree.

### E3. Radii and hierarchy-dependent confusion

```text
d_ab = || stopgrad(z_a) - stopgrad(z_b) ||_2
Delta_i,cd = || mu_i,c - mu_i,d ||_2
r_i,c^init = 0.5 * min_d [Delta_i,cd + 1[c=d] * max(Delta_i)]
n_i,c = max(sum_a 1[h_i(a)=c], 1)
r_i,c <- (1-m) * sum_a 1[h_i(a)=c] ||z_a-mu_i,c||_2 / n_i,c
         + m*r_i,c
G_ab = sum_(i=0..H-1) 2^(-i) * g_i(a,b)
```

Sources: [trainer 254–263][refresh], [287–335][use-targets]. The count includes
both augmented views. Default `m=cluster_momentum=1` freezes radii between
refreshes. The default `zero_one` strategy uses
`g_i(a,b)=1[d_ab > 2*r_i,h_i(b)]`: the `[M]` threshold vector broadcasts over
columns. `pair_dist` instead uses `g_i(a,b)=d_ab`; `pair_cluster` uses
`g_i(a,b)=||z_a-mu_i,h_i(b)||_2`. Only the last/default forms depend on hierarchy
assignments in their target values. No gradient crosses any term in E3.

### E4. Literal confusion normalization (including its broadcasting)

```text
Gtilde_ab = (G_ab - min(G)) / (max(G)-min(G) + 1e-7)
s_j = sum_b Gtilde_jb
C_ab = Gtilde_ab / s_b                     [literal source behavior]
```

Source: [trainer 345–347][use-targets]. The code divides an `[M,M]` tensor by
an `[M]` vector without `keepdim=True`. Consequently the divisor for column b
is the sum of row b. It is **not** `Gtilde_ab/s_a`. There is no zero guard in
the final division. Changing this is a deliberate mathematical correction,
not an optimization that preserves reference outputs.

### E5. USE logits, smoothing target and scalar

Let pi_a list `a+` first and then all non-self, non-positive views. Then:

```text
ell_a,t = -||normalize(q_a)-normalize(q_pi_a(t))||_2 / tau_USE
S_a,t = C_a,pi_a(t)                        [self column removed]
T_a,t = (1-alpha)*1[t=0] + alpha*S_a,t
L_USE = -(1/M) * sum_a sum_t T_a,t * log_softmax(ell_a,:)[t]
```

Sources: [trainer 162–204][use-logits], [42–50][smooth],
[351–352][use-targets]. There is no re-normalization of S after diagonal
removal. Soft CE accepts the literal targets even if their row sum is not one;
do not silently reinterpret it as BCE or ordinary InfoNCE. With `alpha=0`,
it reduces to paired-view CE. Code default `tau_USE=1.0` and `alpha=1.0`;
the [README CUB instruction, line 72][notice] also selects alpha=1.0.

### E6. Supervised contrastive kernel

For a set A of n images with two views, P(a) contains every non-self view with
the same supplied label, including the other view of the same image.

```text
u_ab = -||f_a-f_b||_2 / tau_S
p_ab = exp(u_ab) / sum_(c != a) exp(u_ac),    b != a
SC(f,y) = -(tau_S/tau_0) * (1/(2n)) * sum_a
          [ sum_(b in P(a)) log(p_ab) / |P(a)| ]
```

Source: [trainer 89–146][supcon]. The detached row-max subtraction is a
numerically stable equivalent for these probabilities. Both temperatures
default to 0.07 in this kernel, independently of `args.temperature` for USE.
Zero positive counts and empty sets have no explicit guard.

### E7. SSE and actual offline total loss

```text
S_GT = SC(q[mask_lab], true_labels[mask_lab])
S_H = sum_(i=0..H-1) 2^(-(i+1)) * SC(q^(i), h_i)
L_SSE = 0.5 * (S_GT + S_H)
L_SE = (1-lambda) * L_USE + lambda * L_SSE
```

Source: [trainer 354–380][sse-compose], including the division by two at
line 368. Default lambda is `sup_con_weight=0.35` (line 669). The first
pseudo-labelled term already uses only half the projection dimensions and
weight 1/2; the finest GT-labelled term uses all dimensions. There is no DEAN
CE, EC, direct cosine-to-parent penalty or independent classifier-parent CE in
the original SelEx total loss. Applying a second outer 1/2 to an already
defined `L_SSE` would be an incorrect extra scaling.

### E8. Timing and assignment lifecycle

Source: [trainer 242–268][refresh], 287–335, 681.

```text
if epoch % prototype_extraction_interval == 0:
    hierarchy = HSSK(all offline training features)      [no_grad; model.eval]
for minibatch in epoch:
    update detached radius state using momentum m
    compute E5-E7 and update trainable parameters
```

Default refresh interval is 1, including epoch zero. Assignments/centers are
fixed for the epoch. Radii can change each minibatch when m<1. The hierarchy
extraction uses `images[0]` of the training augmentation, not the deterministic
test transform (trainer 514; merge loader 789).

## 5. Reference defects and semantic differences requiring explicit treatment

| Finding | Evidence | Integration consequence |
|---|---|---|
| Coarse-label row order | trainer 566–590 splits by predicted known/novel; K_Means 184 returns `[l;u]`; trainer 300/364 later reads as loader order | Restore labels to sample-ID order explicitly; retain a literal-reference fixture to expose the difference |
| Dense-ID assumption | trainer 300/364 indexes `argsort(uq_index)` by `uq_idxs` | DEAN IDs can be sparse/reordered; use an explicit ID→row mapping, never array position as class information |
| Confusion division | E4 | Decide literal versus corrected row normalization; do not call the latter bitwise parity |
| Zero confusion | E4 final denominator has no epsilon | Uniform/tied geometry can make USE NaN; needs a documented fallback or fail policy |
| Masked USE mode | trainer 338–351 subsets features but leaves confusion matrix full-sized | Default is all samples; `contrast_unlabel_only=True` is not a safe drop-in switch |
| Nonrecursive known prototypes | trainer 561, 575–590 | Choose literal grouping versus a recursive hierarchy; do not silently substitute a parent-neighbor graph |
| Squared vs unsquared metrics | K_Means 25–27 vs trainer 118/174 | Preserve K-means squared distances and loss unsquared distances separately |
| Optional cosine path | trainer 293/312 uses a whole-tensor norm; logits still use Euclidean cdist | Initial integration should explicitly use Euclidean; cosine would require a separate audit/ablation |
| Empty/singleton cluster updates | K_Means 255–257, 288–290 uses squeeze/index_select and unguarded mean; kpp 103 divides by sum(d2) | Small online groups need declared support limits and deterministic handling |
| CPU portability | K_Means 176 hard-codes CUDA; pairwise distance 31 allocates a CPU output | A CPU test port must disclose device plumbing changes without claiming unchanged whole-program execution |

Read-only CPU checks executed selected original `SupConLoss` AST nodes in memory:
one image/two views returned finite 0; an empty group raised `IndexError` at
`max`; two unique labels/one view returned NaN. An asymmetric 3×3 confusion
matrix using the literal division produced row sums approximately
`[0.666667, 1.833333, 0.833333]`; all-zero confusion produced nonfinite values.
No research implementation or test file was added to perform these probes.

## 6. Verified DEAN insertion points and invariants

| Stage | Pinned DEAN source | Required insertion/data contract |
|---|---|---|
| Stream | [entrypoint 389–395][dean-stream] | Retain the paired manifest and exact incoming sample batches |
| Frozen teacher vs student | entrypoint 403–425 | Teacher/model/head frozen; student copied and LoRA inserted at layers 7–11 |
| Base pseudo labels | [entrypoint 512–522][dean-labels] | Offline teacher argmax, not observed GT |
| Seen/new novel partition | entrypoint 527–565 | Save detection membership by sample ID before merge/shuffle; never infer seen/unseen from GT |
| VFA/discovery | [entrypoint 569–595][dean-vfa] | VFA affects AP-estimated K; final GMM pseudo labels are assigned to real unseen images |
| Final targets and IDs | [entrypoint 600–632][dean-merge] | `dset_tr_now_m.ys` and `.uq_idxs` after all offsets; retain final `on_y` alignment after shuffle |
| Dynamic classifier | entrypoint 636–646 | Class inventory is finalized here; build current-batch detached hierarchy afterward |
| Optimizer creation/expansion | [entrypoint 658–705][dean-optimizer] | Add any new projector group before the final classifier group; DEAN explicitly assumes the last group is the expandable head |
| Differentiable feature | [entrypoint 738–741][dean-update] | Live `feats=model_now(on_x)`, shape `[b,768]`, before classifier; not detached discovery features |
| CE/EC and backward | entrypoint 743–771 | Preserve the official expression, add auxiliary scalar before zero_grad/backward/step |

DEAN's feature is its CLS token after LayerNorm **and the learned 768→768
embedding layer**, with no L2 normalization in that return path
([DEAN ViT 423–447][dean-features], wrapper 590–592). It is not identical to
the original SelEx bare CLS representation. Proposed auxiliary geometry and
projection use this pre-classifier feature for a minimal insertion; tapping a
pre-embedding CLS tensor would be another architecture choice.

The actual DEAN loss is:

```text
I = nonzero(1 - 1[on_y >= 160])[0]           [literal first-index selection]
E_base(z) = -logsumexp(base-head logits)
E_novel(z) = -logsumexp(novel-head logits)
L_EC = mean(log(1 + E_novel(feats[I])/E_base(feats[I])))  [when branch active]
L_DEAN = CE(header_now(feats), on_y) + energy_hp * L_EC
```

Source: [entrypoint 731–767][dean-update]. Preserve its existing selection,
branching and numerical behavior; this plan does not authorize fixing the
misleading mask names, `[0]` indexing or log-ratio domain. `gt_y` is read for
upstream diagnostics at 719–727; it must never enter the new loss/hierarchy
API. No synthetic VFA feature is an additional real image, augmented view or
positive pair for USE/SSE.

## 7. Reviewed v1 decisions (normative; supersede the earlier candidate design)

This is **SelEx adapted to DEAN online discovery**, not a literal reproduction
of offline SelEx. Sections 1-6 describe the reference as audited, including
known defects; production must implement the corrected decisions below, never
an intentional bug-compatible mode. The old `selex_online.py` remains synthetic
scaffolding under its own explicit implementation identity.

1. **Representation.** Hierarchy/EMA prototypes use feature-dimension L2-normalized
   768-D live DEAN pre-classifier backbone features. Contrastive losses use the
   reference three-layer DINOHead: 768->2048->2048->256, normalized bottleneck,
   weight-normalized output, frozen weight-norm gain, and normalized output.
   Output dimension is configurable; production default is 65536. A 768-D output
   changes the projector output, not the existence of the three-layer head.
2. **Views/RNG.** Generate two independent real-image augmentations for EVERY
   online optimization update, including all 15 updates of an incoming batch.
   Re-read only currently available batch images by explicit IDs. Retain the
   original DEAN CE/EC image and original optimizer lifecycle. Isolate Python,
   NumPy, CPU Torch and used CUDA RNG states for SelEx initialization, views and
   stochastic work; the paired stream manifest and DEAN random progression must
   not be changed by auxiliary random draws. Do not duplicate one embedding and
   present it as independent image augmentations.
3. **Supervision.** USE and hierarchy SSE receive valid DEAN pseudo-labels for
   current real samples (base, seen novel and newly discovered novel). The full-
   dimensional true-label SSE term receives ONLY explicitly labelled replay
   samples. Pseudo-base labels are not GT. No replay is enabled by default; when
   none is supplied, omit only the GT term, not hierarchy SSE or USE. No automatic
   base replay or reads of deleted historical raw samples are allowed.
4. **Discovery/hierarchy IDs.** DEAN leaf IDs and pseudo-anchors remain authoritative.
   Do not rerun finest-level HSSK. Maintain bounded EMA prototypes indexed by
   DEAN class ID; build ONLY higher levels from these prototypes. Recursively
   merge nearest prototype groups within base/novel sides, halve supported group
   counts per level, then assign globally unique IDs across leaves and all
   higher levels. This prototype coarsening is an intentional online replacement
   for full-dataset higher-level BSSK and must be recorded in provenance.
5. **Correct numerics.** Normalize along the feature axis; remove self entries,
   use stable row-normalized confusion weights, epsilon-clamped denominators and
   explicit sample/class ID maps. Use stable non-self logsumexp for contrastive
   probabilities. There is no production literal-bug mode. A zero/invalid
   confusion row contributes zero for that USE anchor and increments counters;
   there is no unreviewed one-hot fallback.
6. **Degeneracy.** Empty, singleton and no-positive GROUPS contribute differentiable
   finite zero, without skipping the whole batch. A singleton semantic group is
   one real image even though it has two views. USE needs at least two real
   images and valid target mass; SSE skips semantic groups with fewer than two
   distinct real samples. Count every skip condition. Invalid shapes, duplicate
   IDs, unknown hierarchy IDs and nonfinite inputs remain hard errors.
7. **Online lifecycle.** Update bounded EMA class prototypes AFTER DEAN discovery
   once per incoming stream batch. Rebuild higher hierarchy once, snapshot it,
   and hold it fixed for that batch's 15 optimizer epochs. Store prototypes and
   counts only, with deterministic least-recently-updated eviction and capacity
   at least the current batch's class support. Never store a raw feature/image
   history bank. Rebuild on the next incoming batch, not after every update.
8. **Loss.** Preserve reference internal USE/SSE mixture, temperatures, CUB smoothing,
   hierarchy weighting and the SSE factor 1/2. Keep DEAN CE and its weighted EC
   unchanged. Add gamma*SelEx with default gamma=0.1. Log CE, weighted EC, USE,
   SSE, SelEx total and final total separately on EVERY update, plus skips.
9. **Optimization.** Add a persistent projector AdamW parameter group with lr=1e-4
   before the dynamically expanded DEAN classifier group. The classifier remains
   final throughout expansion; assert group order and parameter identity. Record
   trainable parameter counts and group metadata. Disabled or gamma=0 takes the
   exact original DEAN path: no projector initialization, views, hierarchy,
   RNG draws, optimizer mutation or loss arithmetic.

The real configuration is `configs/dean/selex_online_reference_v1.yaml`. Reviewed
reference defaults are output_dim=65536, mixture/SSE weight=0.35, smoothing=1.0,
USE temperature=1.0, SSE/base temperature=0.07, geometric hierarchy weights
1/2 per level, radius momentum=1, and gamma=0.1. Configurable operational defaults
are prototype momentum=0.9, bounded capacity=512, epsilon=1e-8, and projector
lr=1e-4. Counters and all these values are recorded. Prototype capacity exhaustion
for the current batch is a hard error rather than silently dropping a class.

## 8. Reviewed v1 loss and intentional departures

```text
S_GT = SC(full_projected_replay_views, true_replay_labels)  [zero if none]
S_H = sum_(i=0..H-1) 2^(-(i+1)) * SC(projected_views[:D/2^(i+1)], h_i)
L_SSE = 0.5 * (S_GT + S_H)
L_SelEx = (1-lambda) * L_USE + lambda * L_SSE
L_online = L_CE + L_EC_weighted + gamma * L_SelEx
L_EC_weighted = args.energy_hp * upstream_energy_loss
```

Differences from SelEx commit c3d3b42 are intentional and versioned:

- DEAN discovery supplies leaf assignments instead of offline finest BSSK.
- Bounded EMA prototypes replace whole-dataset feature extraction; recursive
  prototype-only higher grouping replaces higher BSSK. No future/raw history.
- Hierarchy refresh is per incoming stream batch, fixed during its 15 updates.
- Auxiliary views are regenerated per optimizer update; DEAN CE/EC retain their
  existing view and update policy. Auxiliary RNG is isolated.
- Normalized geometry/prototypes share a feature-axis scale; correct ID mapping,
  row-normalized self-excluded targets and stable denominators replace defects.
- Unsupported terms are finite zeros with counters, not NaN, fabricated positive
  pairs or an implicit InfoNCE fallback.
- True supervised SSE is restricted to explicit labelled replay; current stream
  pseudo-labels contribute only hierarchy grouping. Default replay is absent.
- Backbone adaptation uses DEAN LoRA, not SelEx full-block fine-tuning; AdamW and
  DEAN CE/EC are retained. The projector has a separate group before classifier.
- Production uses the 65536-D head; small configurable output/hidden dimensions
  are for CPU tests and declared ablations, never silently selected in production.

The live loss path is auxiliary views -> model_now -> DINOHead -> USE/SSE ->
LoRA/projector gradients. Prototypes, assignments, radii and targets are detached.
This does not authorize changes to DEAN EC masks, discovery/VFA, diagnostics,
stream ordering, checkpoint validation or strict replay.

## 9. Reviewed file-level implementation and test plan

| Proposed file/change | Scope |
|---|---|
| `dean_integration/selex_reference.py` (new) | CPU-capable corrected Euclidean USE/SupCon/SSE kernels; source parity fixtures are test-only, no bug-compatible production profile |
| `dean_integration/selex_hierarchy.py` (new) | Detached HSSK adapter, compact active IDs, ID-order restoration, support/empty-cluster rules, per-batch state |
| `dean_integration/selex_projector.py` (new) | Reference DINOHead profile; optional lightweight profile explicitly labelled an ablation |
| `dean_integration/selex_real_online.py` (new) | Current-batch inputs/two fresh views per update, explicit labelled replay versus pseudo hierarchy provenance, detached targets, loss composition and telemetry |
| `dean_integration/runtime.py`, `external.py` | Checked in-memory hooks after discovery, during second-view preparation, before online loss/backward, and at optimizer group creation; upstream stays pristine |
| `scripts/run_dean.py`, `experiment_config.py`, new versioned YAML | Explicit real-vs-synthetic implementation choice; real and synthetic flags are explicit, with default-off/gamma=0 exact bypass; record profile, weights, head dimensions, support policy, RNG and source hashes |
| `dean_integration/provenance.py` | Record selected real profile, fallback counts, hierarchy support/refresh counts and parameter counts; retain paired manifest identity |
| Existing `selex_online.py` | Keep clearly synthetic; no silent replacement under old experiment identity |
| `tests/test_dean_selex_reference.py` (new) | Literal extracted-reference equality for USE logits/targets/loss and SSE loss/gradients; all weights, temperatures, prefix lengths and factors of two; corrected target row sums after self removal |
| `tests/test_dean_selex_hierarchy.py` (new) | Reordering/sparse-ID/duplicate-ID tests; compare co-membership after ID permutations; empty sides, singletons, all-known, all-novel, ties, K>N and nonfinite inputs |
| `tests/test_dean_selex_real_online.py` (new) | Same paired batches; no future/GT/test-label access; two distinct views; 15-step frozen hierarchy; LoRA/projector/head updates and frozen-parameter invariants |
| Existing CLI/checkpoint/provenance suites | Default-off and gamma=0 exact no-op, strict replay unchanged, synthetic identity retained, classifier last-group expansion and manifest hashes unchanged |

Required stages before any full experiment:

1. Use the reviewed decisions in sections 7-8 and pinned source hashes. Write pure CPU
   reference fixtures without importing/running the whole CUDA trainer.
2. Validate finite scalar and gradient parity against literal source on supported
   fixtures; explicitly assert expected differences for corrected math/guards.
3. Verify ID alignment and causal access, including tests that poisoned `gt_y`,
   future labels and future images cannot change the learner's auxiliary loss.
4. Test two consecutive incoming batches with discovery/head expansion. A new
   projector optimizer group must persist; expansion must replace only the last
   DEAN classifier group and preserve all appropriate AdamW state.
5. Exercise original pinned online statements on tiny CPU modules with real
   auxiliary loss; check finite/nonzero group gradients, frozen tensors, view
   correspondence, loss decomposition and isolated RNG use.
6. Run the full regression suite. Later, only after a separate execution
   decision, profile bounded GPU forward/backward memory and timing, then run
   paired control/treatment. None of these future GPU actions occurred here.

## 10. Extra parameters, VRAM and runtime: analytic estimates

Assumptions: b=64 real images, two views M=128, FP32, DEAN AdamW with two moment
buffers, no history bank, no new backbone copy, no AMP/checkpointing. Values
below are calculations, not measured GPU performance or a guarantee of fitting
on a specific device.

| Item | Reference-head profile | Direct 768-D ablation |
|---|---|---|
| Extra trainable parameters | 23,073,024 | 0 if projection is omitted |
| Extra frozen parameters | 65,536 weight-norm gain entries | 0 |
| Head parameter+gradient+Adam moments | 352.32 MiB, including frozen gains but excluding transient weight-norm/activation buffers | 0 |
| One `[128,D]` projected FP32 tensor | 32 MiB at D=65536 | 0.375 MiB at D=768 |
| One `[128,128]` FP32 pair matrix | 0.0625 MiB | Same |
| Pairwise distance dimensional work | Proportional to 1,073,741,824 coordinate-pair dimensions per full distance matrix | 12,582,912, about 85.3x less |
| Expected extra encoder exposure | One extra view forward/backward per update, plus at most one detached hierarchy forward per incoming batch | Same two-view lifecycle |

Parameter arithmetic from [DINOHead 264–298][head]: the MLP has 6,295,808
parameters and the last layer's trainable weight has 256*65536=16,777,216.
The 65,536 weight-norm gain values are frozen by default. AdamW trainable
state is approximately 16 bytes per parameter (weight, gradient, two moments).
Weight normalization also materializes a roughly 64 MiB effective final weight;
its graph and other buffers are additional. This reference head is much larger
than DEAN's existing 76,810 LoRA parameters across five layers (two rank-5
KV paths and two learned scales per layer; [DEAN PET 61–118][dean-pet]).
Using a direct 768-D tensor preserves neither the original projection space
nor its prefix geometry, so it must be reported as a resource ablation.

For current-batch active HSSK, N<=64 and K<=64: one naive broadcast distance
temporary `[N,K,768]` is at most 12 MiB. Retaining K=200 would make it 37.5 MiB,
but fitting unsupported centers is not the proposed policy. Scanning a large
bank with a 1024-point chunk and K=200 creates a 600 MiB temporary per broadcast
buffer; the original subtraction/square can require more than one buffer.
These are reasons not to copy full-dataset HSSK into the online loop.

Planning allowance for a reference head is roughly **0.5–1 GiB of head/loss
state and transient memory**, plus an additional view's ViT backward activations.
A coarse **2–5 GiB total extra VRAM** budget is a profiling starting point,
not an upper bound. Activations can dominate even though pair matrices are small;
cdist autograd algorithm, simultaneous graphs, attention kernels and allocator
behavior matter. No extra full model weights are required.

Backbone update cost is approximately doubled by the second view. A rough
initial runtime budget is **1.5–2.5x the online update section**, plus hierarchy
and data/augmentation overhead. One detached hierarchy forward per 15 update
steps adds around 1/15 of a forward per original update, not another full
backward. HSSK clustering cost is roughly O(I*N*K*768) across levels; USE/SSE
distance work is O(M^2*D), with the sum of prefix dimensions below D. These
budgets are not a forecast of end-to-end wall time or accuracy.

## 11. Reviewed status and implementation gates

All nine design choices in sections 7-8 are reviewed v1 requirements. There is
no pending permission request to implement them. No GPU training is authorized.
First commit only this reviewed design document as
`docs(dean): finalize real SelEx online integration design`. Then implement and
run the complete CPU test suite. Leave implementation uncommitted/unpushed until
the diff and full results have been reported. The documentation commit is local
only; do not push it in this task.

The historical DEAN result remains All=0.547808, Old=0.631773, New=0.219492.
The paired control and real treatment both require new runs on the same paired
manifest. Synthetic scaffolding results cannot be renamed real SelEx results.

[repo]: https://github.com/SarahRastegar/SelEx
[paper]: https://arxiv.org/html/2408.14371v1#S4
[notice]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/README.md#L22-L74
[order-issue]: https://github.com/SarahRastegar/SelEx/issues/11
[views]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/methods/contrastive_training/contrastive_training.py#L151-L159
[augment]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/data/augmentations/__init__.py#L14-L23
[vit-forward]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/models/vision_transformer.py#L211-L261
[head]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/models/vision_transformer.py#L264-L298
[hssk]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/methods/contrastive_training/contrastive_training.py#L502-L592
[bssk]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/methods/clustering/faster_mix_k_means_pytorch.py#L174-L303
[kdist]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/methods/clustering/faster_mix_k_means_pytorch.py#L10-L46
[use-targets]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/methods/contrastive_training/contrastive_training.py#L287-L352
[use-logits]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/methods/contrastive_training/contrastive_training.py#L162-L204
[smooth]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/methods/contrastive_training/contrastive_training.py#L42-L50
[supcon]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/methods/contrastive_training/contrastive_training.py#L55-L148
[sse-compose]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/methods/contrastive_training/contrastive_training.py#L354-L380
[se-total]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/methods/contrastive_training/contrastive_training.py#L366-L380
[refresh]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/methods/contrastive_training/contrastive_training.py#L236-L286
[datasets]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/data/get_datasets.py#L73-L91
[merged]: https://github.com/SarahRastegar/SelEx/blob/c3d3b42a0aa7d7c9ea748a8305a6890a34100f73/data/data_utils.py#L25-L37
[dean-stream]: https://github.com/KU-VGI/OCGCD/blob/28b9384cd7ea9b123b03d045bac2e92d8d0b2870/Ours_train_online.py#L389-L425
[dean-labels]: https://github.com/KU-VGI/OCGCD/blob/28b9384cd7ea9b123b03d045bac2e92d8d0b2870/Ours_train_online.py#L512-L565
[dean-vfa]: https://github.com/KU-VGI/OCGCD/blob/28b9384cd7ea9b123b03d045bac2e92d8d0b2870/Ours_train_online.py#L569-L595
[dean-merge]: https://github.com/KU-VGI/OCGCD/blob/28b9384cd7ea9b123b03d045bac2e92d8d0b2870/Ours_train_online.py#L600-L646
[dean-optimizer]: https://github.com/KU-VGI/OCGCD/blob/28b9384cd7ea9b123b03d045bac2e92d8d0b2870/Ours_train_online.py#L658-L705
[dean-update]: https://github.com/KU-VGI/OCGCD/blob/28b9384cd7ea9b123b03d045bac2e92d8d0b2870/Ours_train_online.py#L709-L771
[dean-features]: https://github.com/KU-VGI/OCGCD/blob/28b9384cd7ea9b123b03d045bac2e92d8d0b2870/net/vision_transformer.py#L423-L447
[dean-pet]: https://github.com/KU-VGI/OCGCD/blob/28b9384cd7ea9b123b03d045bac2e92d8d0b2870/net/pet_modules.py#L61-L118
