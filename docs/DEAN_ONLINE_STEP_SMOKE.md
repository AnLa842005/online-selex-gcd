# DEAN-1A opt-in GPU one-step smoke

Status: implementation and CPU tests only; **CUDA execution NOT RUN**. Base:
`d795cf98ad70d343a62c68ad4d05dc072b20c143`, branch `research/dean-selex`, initial
working tree clean. The existing full teacher checkpoint has raw-official PASS;
its [audit evidence](../results/dean_1a_full_checkpoint_audit.json) is unchanged.
This task changes no upstream source, official config or DEAN algorithm.

## Two modes and strict limits

`scripts/smoke_dean_gpu_forward.py` keeps its default frozen forward-only mode:
one real CUB training image, official test transform, one inference forward, no
backward, zero optimizer steps and zero epochs. The opt-in `--online-step` uses
two images by default, official train transforms, offline forward, online
forward/loss, one backward and exactly one AdamW step. `--batch-size` accepts 1..4;
online mode requires at least two samples. There is no epoch or dataset-loader
loop, resume, model saving, tuning or accuracy calculation.

The script is Linux/CUDA-only and retains the isolated environment and full
checkpoint gates. Missing prerequisites exit before GPU execution. Reports cannot
overwrite existing files. CPU tests call the shared boundary/step helper directly
with synthetic tensors; they do not represent a CUDA pass or a real CUB step.

## Official components and fixture boundary

The actual external `ViT_Dino` loads `checkpoint['teacher']` using unchanged
`strict=False`, before the official post-load embedding/head replacement. Full
raw loading is audited first; no prefix conversion is applied. The offline
model and native `Head(768,160)` are copied to the online pair before freezing
the original model/head. All online model parameters are frozen, then native
KVLoRA is attached at blocks **7,8,9,10,11** with official defaults (rank 5).
The native online Head expands to 162 using its official `update` method.

The embedding and base head are **untrained** in this initialization-path smoke.
Two extra head slots are fixture pseudo classes, **not estimated K or the true
novel count**. Even batch rows take offline-head argmax (0..159); odd rows take
fixture slots 160/161. No ground-truth class label is an input to this helper.
The image adapter reads images.txt and train_test_split.txt, not class labels.
The separate trusted asset preflight reads labels only to validate CUB layout.
This fixture replaces discovery for the smoke; EGD/VFA pseudo-labelling and
base-trained model behavior are not validated by this run. No SSB claim is made.

The online pseudo masks, feature indexing, CE and energy loss are read at runtime
from the **locked original AST statements** at Ours_train_online.py:731–767.
The epoch-loop header is not executed; its loss body is executed once. Original
mask indexing (including `[0]`) and the unclamped energy log-ratio are retained.
No copied DEAN core or modified loss is distributed. Nonfinite loss fails rather
than being clamped, retried, tuned or repaired. Optimizer settings/groups follow
the initial official online AdamW: lr=1e-4, weight_decay=1e-4, betas=(.9,.999).

## Update boundary and reporting

Before forward, exact parameter snapshots are cloned to **CPU memory only**;
they are never serialized. Expect additional host RAM for both model snapshots.
The checker rejects shared parameter storage, wrong requires_grad flags, changed
parameter names/identities/shapes and nonfinite initial values. After backward,
it requires finite nonzero gradients in both KVLoRA and online-head groups and
rejects any gradient in an offline/frozen group before permitting the step.

After the step it requires:

- Offline model and head parameters remain exactly equal to their snapshots.
- Every frozen online parameter remains exactly equal to its snapshot.
- KVLoRA and online head each contain a parameter with a nonzero finite gradient
  **and** an observed update. Weight decay alone cannot satisfy this gate.
- All parameter values, gradients, optimizer state, features, logits and loss are
  finite. Model structure and trainability have not changed.

LoRA B starts at zero, so not every A/B/scale tensor must have a nonzero gradient
or change on step one. Requiring a gradient-driven update in each trainable group
is intentional. Failures return nonzero with status FAIL; no retry/second step
or checkpoint is made. `optimizer_step_attempts` distinguishes an attempted step
that raises from a successfully completed step. A post-step boundary failure
retains `optimizer_steps=1` and FAIL, not a misleading success or rollback claim.

The JSON report includes checkpoint SHA, environment, input/features/logits/target
shapes, pseudo target origin, trainable/frozen tensor and numel counts, changed
parameter/gradient names by group, finite gates, forward/backward/step timings,
peak allocated/reserved VRAM, `epochs=0`, `full_training=NOT RUN`, and exact
optimizer step count. Offline and online forward timings are separate. Snapshot,
gradient-copy and diff overhead is outside the three GPU compute timings, while
peak VRAM covers setup/forward/backward/update. Runtime failures and CUDA OOM are
FAIL. No model or dataset enters Git.

## Exact Linux commands

Activate the isolated `dean-official-v0` environment before these commands. Paths
are explicit placeholders for the Linux host, not Windows paths. Use fresh report
paths and the pinned external checkout SHA
`28b9384cd7ea9b123b03d045bac2e92d8d0b2870`.

Environment/assets/checkpoint verification (must exit 0):

```bash
python scripts/run_dean.py --action verify --upstream /path/to/OCGCD --dataset-parent /path/to/datasets --checkpoint /path/to/dino_vitbase16_pretrain_full_checkpoint.pth --output /path/to/runs/dean_1a_verify
```

CPU components (no dataset or checkpoint required):

```bash
python scripts/run_dean.py --action smoke --upstream /path/to/OCGCD
```

Default one-image GPU forward smoke:

```bash
python scripts/smoke_dean_gpu_forward.py --upstream /path/to/OCGCD --checkpoint /path/to/dino_vitbase16_pretrain_full_checkpoint.pth --trusted-local --dataset-parent /path/to/datasets --report /path/to/runs/dean_1a_gpu_forward.json
```

Opt-in one-step GPU smoke (two real images, no epoch):

```bash
python scripts/smoke_dean_gpu_forward.py --upstream /path/to/OCGCD --checkpoint /path/to/dino_vitbase16_pretrain_full_checkpoint.pth --trusted-local --dataset-parent /path/to/datasets --report /path/to/runs/dean_1a_gpu_online_step.json --online-step --batch-size 2
```

No full-training command is proposed until an actual Linux online-step report
passes all gates. Full DEAN reproduction remains NOT RUN.

## Local validation record

- Full suite with external checkout enabled: **202 tests PASS**, 124.640 seconds.
- DEAN suite: **35 tests PASS**, 18.975 seconds. The new boundary suite has eight
  tests, including one CPU integration fixture using the original small 12-block
  ViT, KVLoRA, Head and selected loss statements. It observes one offline forward,
  one online forward, one backward, and exactly one AdamW call on CPU.
- Negative fixtures reject frozen mutations, invalid trainability, shared storage,
  missing/zero-only PET gradients, absent updates, changed parameter structure,
  NaN gradients and Inf parameters. Snapshots remain independent CPU copies.
- Compilation, CLI help, batch-size-1 online refusal and diff checks pass.
- Official/draft configs, environment, upstream lock and historical JSON evidence
  remain byte-identical to the task base. Offline comparison: `differences: {}`.
  Existing frozen lower bound, Phase 4A and offline SelEx regression tests pass.
- Upstream remains clean at the pinned SHA. **No CUDA execution**, real-CUB
  optimizer update, GPU peak memory measurement or GPU timing was performed here.
