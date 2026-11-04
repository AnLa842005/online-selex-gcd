# DEAN-0 upstream and protocol audit

Status: preparation only, **NOT RUN** on CUB. No new accuracy result. Audit date:
2026-11-04. DEAN is the strong baseline/main integration path. Frozen-DINO is
**lower bound, frozen**; Phase 4A is **negative control, closed**. Phase 4B is a
deferred hierarchy-only ablation; its existing implementation is preserved.

## Source and preservation

- Official URL: https://github.com/KU-VGI/OCGCD (KU-VGI authors, not project code).
- Pinned main SHA: `28b9384cd7ea9b123b03d045bac2e92d8d0b2870`.
- External audit clone: sibling `../OCGCD-dean-audit`; no code vendored.
- No root LICENSE/COPYING or explicit repository license found in the 42 tracked
  files. Individual files have attribution/reference headers, which do not grant
  a blanket redistribution license. Redistribution is blocked pending clarification.
  Private execution through a pinned external checkout is the integration route.
- Full tracked tree and SHA256 of every Git blob: [upstream lock](../configs/dean/upstream_lock.json).
  Hashes use Git blobs to avoid Windows CRLF conversion differences. Config
  digests are canonical JSON SHA256; source revisions and dirty checkouts fail closed.
- Requested starting HEAD was `2ba7f40f3f2f817530557ba3a0721dbe9c3983d2`;
  observed clean HEAD was `031bd87a7c7636036c6ccb7e5540739c721b5398` on
  `research/online-selex`. `git pull --ff-only` returned already up to date.
  New `research/dean-selex` starts at the observed HEAD, including existing Phase 4B.
- Remotes remain origin `AnLa842005/online-selex-gcd`, upstream `SarahRastegar/SelEx`.
  Tag object `450c6ba5db51812a6c8873bd6cc925a99a0ccb2f`, baseline target
  `a52cf64dcbc43a57069c708ce0ce616cdd045221` are unchanged. All pre-existing
  Python/YAML/JSON and requirements files are compared to the task base in tests.

## Evidence map at pinned revision

All line references below refer to the pinned upstream, not project code.

| Source | Audit evidence |
|---|---|
| `README.md`, `run.sh` | Python 3.9.x, torch 2.0.0, timm 0.9.8, sklearn 1.3.2; RTX 3090; official sweep/flags |
| `Ours_train_online.py:76–139` | CLI defaults; no resume flag; checkpoint path; prop_samples=0.8 |
| `Ours_train_online.py:172–213` | CUB parent hard-coded as `Your data path..`; DINO teacher load `strict=False`; 768-D model/head; unconditional CUDA |
| `Ours_train_online.py:255–350` | 30 base epochs via launcher; CE; warmup; test evaluation every epoch; best base model/head save and reload |
| `Ours_train_online.py:389–490` | shuffled incremental batches; frozen offline copy, online PET copy; true-label diagnostics; energy/GMM split |
| `Ours_train_online.py:525–645` | online novel-head second split; VFA; Affinity Propagation estimates K; GMM pseudo-labels; head expansion |
| `Ours_train_online.py:710–779` | diagnostic GT read; 15 update epochs on each current batch; CE + energy ratio loss |
| `Ours_train_online.py:787–934` | GT-seen-class test filtering; global Hungarian mapping; final console summary |
| `dataset/get_data_utils.py:5–10,52–61` | per-call numpy seed 0 subsampling, first 160/last 40 classes |
| `dataset/cub.py:198–321`, `dataset/__init__.py` | actual invoked CUB loader, per-class 80% base labels, withheld known + novel stream, test construction, `new_inc` from labels |
| `net/vision_transformer.py`, `net/pet_modules.py`, `net/Decoder.py` | custom ViT-B/16, KVLoRA rank 5, expanding linear Head |

`OCGCD_DEAN_ORIGINAL_CODE.py` is an alternative source file; **run.sh invokes
Ours_train_online.py**. The wrapper follows that entrypoint, not the alternative.

## Official architecture and training

Base: DINOv1 ImageNet ViT-B/16, custom 12-block/768-D backbone plus learned embedding
and linear `Head(768,160)`. Base supervised CE trains according to official warmup,
AdamW lr=1e-4, weight decay=1e-4 and StepLR(step=5,gamma=.5). The best test Old
Hungarian accuracy selects the base checkpoint. The offline model/head are then
frozen. Online model/head are copied; KVLoRA rank 5 is inserted into blocks
7,8,9,10,11; the expanding online head and PET parameters are trained.

EGD: energy is negative logsumexp of offline logits, two-component GMM separates
base/novel with the more negative-energy component assigned base. In later batches,
online novel-head energy separates previously discovered/new novel samples.
VFA adds five Gaussian samples per unseen feature using its per-dimension batch
standard deviation. Affinity Propagation over augmented features estimates K;
a K-component GMM on original unseen features assigns pseudo-labels. This is
estimated K, not a true-40-class oracle in the clustering call. The dataset builder
does know the complete 200-class inventory; do not conflate these facts.

The invoked loss is CE on pseudo-labels plus energy_hp=1 times
`log(1 + energy_n_unlab / energy_n_lab).mean()`. Preserve the actual upstream mask
indexing and loss, including possible numerical/selection defects. `--alpha 32`
and `--mrg 0.1` are launcher arguments but the active base loss is CE. Do not
substitute Proxy Anchor because of the CLI description.

## Official CUB versus current clean frozen SSB protocol

| Dimension | Official DEAN CUB | Frozen project SSB 100/100 |
|---|---|---|
| Known/novel | contiguous zero-based 0..159 / 160..199 | fixed SSB 100/100 file/hash retained |
| Labelled fraction | floor(80% of each known class), helper resets numpy seed 0 | global known-train 50%, RandomState(0) |
| Sessions | base + 1 incremental | base + 5 incremental |
| Introduction schedule | all 40 novel classes in session 1 | 20 novel classes introduced each session, causal allocation |
| Stream | shuffled batches of 64, one pass through incremental loader | single-sample causal updates within manifest sessions |
| Replay/reuse | current image batch reused for 15 optimizer epochs; 5 VFA feature replicas; no active raw cross-batch replay buffer in this entrypoint | zero replay, one update per observation, unbounded logged prototype bank |
| Epochs | base 30, incremental 15 **per batch** | frozen extraction; no encoder/head training epochs |
| Novel count | AP estimates discovery K; builder knows full label inventory | novel count/future metadata unavailable to learner |
| Fine split | offline energy GMM then online novel-head GMM | calibrated known-only prototype assignment |
| Model roles | frozen base offline teacher + trainable PET online student/head | one frozen backbone + prototype state |
| Evaluator mapping | one Hungarian projection on evaluated test subset, shared across subsets at that evaluation | existing evaluator's fixed/cumulative/probe mappings and persistent predicted IDs |
| Forgetting | no project-equivalent signed/clipped forgetting implementation | signed/clipped forgetting from project evaluator |
| Probe semantics | no immutable project probe contract | eligible disjoint test probes, state invariant before/after prediction |
| Test access | base best checkpoint selected by test accuracy; anytime test filtered by true seen classes | probes evaluator-only, never select/tune learner |
| Checkpoint provenance | full DINO teacher dict expected, strict=False, absolute default path | known hashed bare DINO backbone, clean frozen feature cache |

**Official reproduction and SSB adaptation are different experiments. Official
CUB scores cannot be compared directly with frozen SSB 100/100.** No SSB compliance
claim is made for DEAN-0. The legacy Phase 1 offline-preexposed protocol is not the
clean frozen protocol used in this comparison.

## Environment, paths, artifacts and operational limitations

The README supplies only Python minor, torch, timm and sklearn pins. No complete
upstream requirements/lock exists. [Environment](../environments/dean/environment.yml)
is a project-selected reproducible **candidate**, not an authors' complete environment:
Python 3.9.18, torchvision .15.1/CUDA 11.8 and all remaining package pins were selected
for compatibility. Fresh Linux solve/import/full run remains NOT RUN. Existing
SelEx environments/requirements are untouched; no packages installed in this task.

Active data literal is in the entrypoint, not config.py. config.py also contains
`/local_datasets`, `MetaGCD/pretrained_weight/...`, `/MetaGCD/...`, `/data/pgh2874/...`.
The wrapper binds the single dataset-parent literal **in memory**, passes an
explicit trusted checkpoint path, and runs in a separate seed-specific output cwd.
The external source remains clean. This infrastructure-only delta, path/log/exp
arguments and process/result isolation are all differences; no learning change.

Output: relative `Ours/cub/VIT_sp_True_gm_False_DEAN_Repro/cub_VIT_best_step_0.pth`,
`Inc_0/`, `report_dict/Latents/DEAN_Repro/`, `report_dict/grads/DEAN_Repro/`,
console diagnostics/final metrics. Some report directories are created but not
populated. Wrapper adds provenance.json, execution.json and console.log per seed.
No final online model checkpoint or optimizer/RNG/scheduler resume state is saved
by this invoked entrypoint. “Resuming from checkpoint” means intra-run base-best
reload, not interrupted-run resume. Stop/restart means a fresh output root/run.

CUDA is assumed throughout `.cuda()`, `device='cuda'`, optimizer state expansion
and embedding extraction. `--gpu-id -1` is not a CPU execution mode. CPU fixtures
execute selected external AST nodes and import PET only; they do not import/run
the complete training entrypoint.

## Leakage and correctness gates for DEAN-1

1. True incremental `on_y`/`prev_labels`/`gt_y` reach orchestration diagnostics;
   `online_seen_classes` filters test data. Training dataset carries both pseudo
   and real labels. Current audit finds CE uses pseudo-labels, but a complete
   dataflow isolation proof is still required for strict SSB.
2. Base test-based checkpoint selection is actual official behavior. Preserve and
   disclose it for reproduction; a known-train validation rule is a separately
   versioned adaptation, never a silent official fix.
3. Default checkpoint expects `torch.load(...)["teacher"]` then `strict=False`.
   Confirm key prefixes, matching keys and effective initialized weights before
   GPU execution; a bare frozen checkpoint cannot be silently substituted.
4. EGD/GMM may receive tiny/empty/degenerate partitions; AP-estimated K can exceed
   original unseen sample count after VFA. Smoke does not establish robustness.
5. Energy masks use `torch.nonzero` and `[0]`, GT diagnostic boundary uses `<=`,
   evaluation reuses `batch_idx`, and end-of-run confidence concatenation assumes
   populated lists. These are review risks, not patched algorithms.
6. No probe-driven thresholds/tuning, future SSB labels/counts or hierarchy/USE/SSE
   changes are authorized. A separate DEAN-1 execution decision must freeze the
   exact assets/environment and resolve checkpoint/entrypoint feasibility first.
