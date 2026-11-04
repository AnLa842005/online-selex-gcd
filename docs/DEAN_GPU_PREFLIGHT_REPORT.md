# DEAN-1A GPU preflight report

Date: 2026-11-04 (Asia/Bangkok). **BLOCKED before GPU smoke/training**.
This report records actual local observations, not a Linux reproduction result.
No new DEAN accuracy, full epoch, GPU forward/backward or optimizer step ran.
Machine-readable evidence: [local JSON report](../results/dean_1a_local_preflight.json).

## A. Status and base

Initial branch `research/dean-selex`, HEAD
`a5a40a105b8fefe0103bd270da9d603b3d75dc78`, working tree clean: PASS.
The existing external clone `../OCGCD-dean-audit` was reused and detached at
`28b9384cd7ea9b123b03d045bac2e92d8d0b2870`. Origin is
`https://github.com/KU-VGI/OCGCD.git`. SHA, clean status and all 42 locked Git blob
hashes pass. No upstream source was copied or modified; no license change found.

DEAN-1A acceptance is **not met**: no suitable Linux environment/full teacher
checkpoint is available, so real GPU smoke and parameter-update measurements are
NOT RUN. Successful synthetic tests cannot discharge these gates.

## B. Environment

| Item | Observed |
|---|---|
| Host | Windows, not Linux |
| Python | Existing `.venv`, 3.11.9; required separate candidate is 3.9.18 |
| Torch / runtime CUDA / cuDNN | 2.0.0+cu118 / 11.8 / 8700 |
| Local GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| Device memory | 4,294,443,008 bytes reported by torch; nvidia-smi reports 4096 MiB |
| Driver | 566.07 from nvidia-smi |
| Linux access | WSL inventory contains only docker-desktop; Docker Linux daemon pipe unavailable; no Linux SSH host supplied |
| Candidate environment | `environments/dean/environment.yml`, unchanged; Linux installation NOT RUN |

`conda` is not available on PATH. Current timm=.9.10, sklearn=1.3.0 and other
package versions differ from the candidate; einops and pytorch-metric-learning
are absent. Exact installed package versions/errors are in JSON. No packages
were installed/upgraded and the existing SelEx environment was not changed.
GPU availability was queried only; no GPU tensor workload was launched.

## C. CUB audit

Actual dataset parent: `C:/Users/ASUS/Desktop/Code/DA/data`.
Root: `CUB_200_2011`, with `images/` and official metadata in the expected layout.
All 11,788 unique referenced image paths exist. Metadata IDs align across all
three tables, 200 classes cover official labels 1..200, train/test counts are
5,994/5,794. Official DEAN uses zero-based known 0..159 and novel 160..199.
Per-class floor(80%) gives 3,836 labelled base samples. This is the official
160/40 protocol, not SSB 100/100. Metadata/layout checks do not decode all images
or compute a full pixel-content digest.

| Metadata | SHA256 |
|---|---|
| images.txt | `e1cfc5ed1f4bd1b838040784b6f75c44f7a25d9b31707a3adab49ec694618c1f` |
| image_class_labels.txt | `e9df8c96d897994a245746fe2c98914b9b31f5342bddc6e8b06bdfbf45e49b02` |
| train_test_split.txt | `8bbbfcc09b4d3c9c6e3e40c9874bd863eb6658f4f456e472df1b33d3403d2600` |

## D–E. Actual checkpoint structure and matching

Input: `C:/Users/ASUS/Desktop/Code/DA/pretrained/dino_vitbase16_pretrain.pth`.
Size: 343,242,485 bytes. SHA256:
`bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e`.
Loaded on CPU with `torch.load(..., map_location='cpu', weights_only=True)`.

**FAIL: bare state dict, 150 top-level keys, no nonempty `teacher` mapping.**
It is the retained frozen-backbone asset; it was not silently wrapped into a
teacher dict or substituted for the full checkpoint expected by official DEAN.

The audit constructs actual external `ViT_Dino(sz_embedding=768)` at the exact
teacher-load point, before the upstream embedding/head replacements. Its target
has 150 parameter tensors / 85,798,656 elements. It checks names, shapes, finite
values and every critical group: patch_embed, cls_token, pos_embed, blocks 0..11,
final norm. No model download or state_dict loading is invoked by the audit.

For this rejected input the effective teacher mapping is empty: 0 matched,
150 missing, 0 unexpected teacher keys and no comparable shape pairs. Eligible
parameter tensor/numel ratios are both 0%. These figures do not mean the raw
bare tensor dimensions were incompatible; the required wrapper structure itself
failed, so it is ineligible for the official load call. All mandatory groups fail.
Full missing-key/group lists and raw top-level prefix counts are in JSON.

`teacher`, `module`, `backbone` prefixes are reported separately. Possible prefix
strips/additions are diagnostic counts only. **No key transformations are applied.**
The audit permits unexpected projection-head keys only when the entire reference
backbone matches. Any missing backbone tensor, shape mismatch, non-tensor matching
entry or nonfinite matched value fails closed. This gate is stricter than a
majority threshold. Official `strict=False` is unchanged; the separate audit
does not call or modify that upstream load operation.

`run_dean.py` now requires this compatibility result (plus Linux/environment
checks), preventing a teacher-shaped but incompatible checkpoint from passing
the previous schema-only preflight.

## F–G. GPU and frozen/trainable parameter audit

GPU real-image forward: NOT RUN. Online model/loss minibatch: NOT RUN.
Optimizer steps: **0**. Offline model/head state preservation and online/PET-only
updates have **not** been dynamically verified on GPU. This is a blocker, not a pass.

Static upstream evidence: Ours_train_online.py:403–431 copies offline model/head,
freezes the original pair, freezes online backbone parameters, then inserts KVLoRA
at layers 7..11. net/vision_transformer.py:317–339 and 369–375 define PET freeze
rules/insertion. Head expansion occurs at 645. Actual trainable parameter names,
gradients and before/after values still need measurement with eligible assets.
No base-trained classifier exists for this smoke yet; any future untrained-head
path test must disclose that limitation and cannot count as reproduction accuracy.

## H. Leakage observations and evidence

All references below are to the pinned upstream.

| Tensor/field | Meaning and use |
|---|---|
| Initial incremental `on_y`, `prev_labels` | True labels from ordinary dataset samples; seen-class list and discovery diagnostics |
| `preds_lb_o`, `pseudo_seen`, `pseudo_unseen`, `preds_lb_new` | Model/clustering pseudo labels; replace dataset `.ys` at 604,621/623 |
| Binary CUB `target` / training `on_y` | Pseudo label from `.ys[idx]`, because binary=True at 631 |
| Binary `real_target` / training `gt_y` | True label from original metadata; diagnostic Hungarian/pseudo accuracy and `lab_mask` at 719–726 |
| `logits`, `pred`, `preds_lb` | Classifier scores and prediction IDs; test evaluator mapping |
| `y_n`, energy masks | Derived from pseudo `on_y` at 731, not `gt_y` |

The online CE at 765 is exactly `F.cross_entropy(logits, on_y)`. Energy loss at
741–767 depends on features, heads and pseudo-label masks. The external-fixture
test executes original CUB `__getitem__` binary code: changing the metadata label
changes returned GT while the pseudo target stays fixed. It then executes the
original selected online loss statements and permutes `gt_y`: both loss and head
gradient remain identical. This proves the tested binary-loader/loss path uses
pseudo targets; it is not an end-to-end GPU or global no-leakage proof.

True labels remain in discovery diagnostics, pseudo accuracy, class-list logging
and seen-class test filtering. Base checkpoint selection still uses official test
accuracy. Nothing here removes or changes that selection. No SSB-compliance claim
is made, and probes were not used for tuning.

## I. Resources

Peak GPU VRAM: **not measured**, JSON null. Device capacity is not peak usage.
Forward/backward runtime and real-image tensor dimensions: **not measured**, JSON
null. CPU component fixtures pass but are not substituted for these measurements.

## J. Validation

DEAN checkpoint/component tests pass, including exact-reference construction,
prefix/shape/partial/nonfinite/empty-teacher rejection, CPU load/hash, unchanged
model state, loader pseudo/GT separation and loss/gradient invariance to GT.
Final full suite: **187 tests PASS**, 132.647 seconds, with external checkout
enabled. Targeted DEAN suite: **20 tests PASS**, 15.252 seconds. Both CLIs' help,
compilation, report-overwrite refusal and `git diff --check` pass. Offline config
comparison remains `differences: {}`. Tests preserve offline SelEx, frozen lower bound, Phase 4A, baseline
tag and official config semantics. The official config and environment file
are byte-identical to DEAN-0.

## K–L. Changes and Git

Added `dean_integration/checkpoint_audit.py`, prerequisite-report CLI,
checkpoint/leakage tests, this report and small JSON evidence. Updated only the
DEAN preflight and its runner call to enforce matching/Linux gates. No dataset,
checkpoint, checkout, model or large log is included in Git. Audit JSON records
base HEAD and dirty status honestly because new audit code was under development.
Commit/push outcome is reported in the task response.

## M. Conditional DEAN-1B seed 1 command

**Do not execute yet.** First provide an accessible Linux CUDA host and trusted
full teacher checkpoint, create the isolated candidate environment, pass the
new checkpoint audit, then perform and review a bounded real-image GPU one-batch
smoke with state/gradient/VRAM checks. This prerequisite-report CLI intentionally
does not implement or claim that GPU stage.

Linux environment setup and prerequisite audit (not run here):

```bash
conda env create -f environments/dean/environment.yml
conda activate dean-official-v0
python /path/to/OCGCD/Ours_train_online.py --help
python scripts/audit_dean_preflight.py --upstream /path/to/OCGCD --dataset-parent /path/to/datasets --checkpoint /path/to/dino_vitbase16_pretrain_full_checkpoint.pth --output /path/to/runs/dean_1a --report /path/to/runs/dean_1a_prerequisites.json
```

That command exits nonzero on prerequisite failure, and reports
`PREREQUISITES_PASS_GPU_SMOKE_PENDING` on success, never a completed DEAN-1A pass.
Only after the separate GPU acceptance passes, the proposed full DEAN-1B command is:

```bash
python scripts/run_dean.py --action train --config configs/dean/cub_official_v0.yaml --upstream /path/to/OCGCD --dataset-parent /path/to/datasets --checkpoint /path/to/dino_vitbase16_pretrain_full_checkpoint.pth --output /path/to/runs/dean_1b_seed1 --seed 1
```

CUB full training remains **NOT RUN**.
