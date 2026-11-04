# DEAN official CUB reproduction runbook v0

Status: **NOT RUN**. This task performs local/static/component validation only.
No dataset/model download, full training, RunPod or GPU job has been executed.
See the [audit](DEAN_UPSTREAM_AUDIT.md) for deviations, leakage and source attribution.

## Isolated environment and source

Use a separate Linux CUDA environment; never upgrade the SelEx environment in place.
The candidate pins are fully listed in `environments/dean/environment.yml`:
Python 3.9.18; PyTorch 2.0.0; torchvision 0.15.1; CUDA runtime 11.8;
timm 0.9.8; sklearn 1.3.2; numpy 1.24.4; scipy 1.10.1; pandas 2.0.3;
Pillow 9.5.0; matplotlib 3.7.3; tqdm 4.66.1; einops 0.7.0;
pytorch-metric-learning 2.3.0; PyYAML 6.0.1.
Only Python 3.9.x/torch/timm/sklearn originate in the upstream README. The remaining
pins are project choices and the environment solve/import on Linux is NOT RUN.
Archive `conda list --explicit`, `pip freeze`, driver and GPU details on the real host.

```bash
conda env create -f environments/dean/environment.yml
conda activate dean-official-v0
git clone https://github.com/KU-VGI/OCGCD.git /path/to/OCGCD
git -C /path/to/OCGCD checkout --detach 28b9384cd7ea9b123b03d045bac2e92d8d0b2870
```

No upstream sources are distributed by this project. Checkout must be clean and
origin must match the official URL. The lock verifies the entire Git tree/blobs.

## Assets and preflight

- `/path/to/datasets/CUB_200_2011`: official `images/`, `images.txt`,
  `image_class_labels.txt`, `train_test_split.txt` (11,788 rows; 5,994 train/5,794 test).
- `/path/to/dino_vitbase16_pretrain_full_checkpoint.pth`: trusted local full
  DINOv1 teacher checkpoint. Record source URL/acquisition, license, SHA256 and
  effective matched teacher keys. The current project bare-backbone SHA is not
  evidence that this full checkpoint is equivalent.
- `/path/to/runs/dean_RUN_ID`: new writable output root; avoid prior run roots.
  Inside this repository, only `runs/dean/...` is accepted.

```bash
python scripts/run_dean.py --action plan --upstream /path/to/OCGCD
python scripts/run_dean.py --action verify --upstream /path/to/OCGCD --dataset-parent /path/to/datasets --checkpoint /path/to/dino_vitbase16_pretrain_full_checkpoint.pth --output /path/to/runs/dean_RUN_ID
```

Verify checks exact Python/package pins, CUDA 11.8/GPU availability, teacher dict
schema/SHA, aligned full CUB metadata/image existence, expected contiguous 160/40
class split, writable output and upstream/config/environment provenance. Errors
return nonzero. This is necessary but not sufficient: effective checkpoint/model
key matching and full entrypoint imports must still pass on the target environment.
Metadata checks use actual assets; plan/smoke never require or download assets.

On the target environment, check complete entrypoint imports without training:

```bash
python /path/to/OCGCD/Ours_train_online.py --help
```

## CPU component smoke

```bash
python scripts/run_dean.py --action smoke --upstream /path/to/OCGCD
DEAN_UPSTREAM_CHECKOUT=/path/to/OCGCD python -m unittest discover -s tests -p test_dean_integration.py -v
```

Smoke imports the external PET module, constructs official offline/online Heads,
expands the online head, forwards tiny CPU features, checks LoRA, EGD GMM separation,
deterministic VFA, a finite loss/backward fixture and minimal state save/load.
Original AST statements are read from the locked external source at runtime;
no copied learner implementation is included here. No real dataset, full ViT,
complete entrypoint import or optimizer resume is claimed by this smoke.

## Exact official reproduction command (future execution only)

```bash
python scripts/run_dean.py --action train --config configs/dean/cub_official_v0.yaml --upstream /path/to/OCGCD --dataset-parent /path/to/datasets --checkpoint /path/to/dino_vitbase16_pretrain_full_checkpoint.pth --output /path/to/runs/dean_RUN_ID
```

Default sweep is sequential `[1,3,8,2,4]`; `--seed 1` selects one official seed.
The effective upstream command for each seed, executed in its separate cwd, is:

```bash
python Ours_train_online.py --model VIT --dataset cub --alpha 32 --mrg 0.1 --lr 1e-4 --warm 5 --base_epochs 30 --inc_epochs 15 --batch-size 64 --inc_batch-size 64 --num_base_classes 160 --num_inc_sessions 1 --lora_layers 7 8 9 10 11 --energy_hp 1. --n_replay 5 --use_EC --use_VFA --seed 1 --exp DEAN_Repro --vit_pretrained_dino /path/to/dino_vitbase16_pretrain_full_checkpoint.pth --LOG_DIR ./logs
```

The wrapper substitutes only the hard-coded dataset-parent literal at runtime.
All remaining defaults, transforms, labels/diagnostics, losses and evaluator
behavior remain official. There is no hierarchy, Self-Expertise, USE or SSE.

## Artifacts, resources and stop behavior

Expected output tree:

```text
dean_RUN_ID/official/cub_v0/
  provenance.json                # config, SHA, effective args, environment spec and assets
  seed1/                        # likewise seed3, seed8, seed2, seed4
    execution.json              # STARTED / COMPLETE / FAILED / INTERRUPTED
    console.log                 # official metrics/traceback, not project-schema results
    Ours/cub/VIT_sp_True_gm_False_DEAN_Repro/
      cub_VIT_best_step_0.pth    # base model/head only
      Inc_0/
    report_dict/Latents/DEAN_Repro/
    report_dict/grads/DEAN_Repro/
```

The output root cannot reuse an existing `official/cub_v0` directory. On nonzero
child exit, mark FAILED, retain logs/partial artifacts and stop the sweep. Ctrl+C
stops the foreground process group; verify the child process has exited on the
target shell before releasing a GPU. Hard kill/power loss can leave STARTED status;
do not interpret it as completion. No exact training resume exists upstream:
restart into a new root and do not report a partial run as reproduction.

Upstream reports RTX 3090. Plan one CUDA GPU of the same 24-GB class for initial
validation; this is a planning estimate, not a measured minimum. Dual ViT copies,
optimizer state and per-batch repeated training need actual profiling. Runtime,
host RAM and disk requirements have not been benchmarked. Run seeds sequentially.

Before a DEAN-1 real run: resolve full-checkpoint provenance/key compatibility,
fresh candidate environment solve/full imports, real metadata split checks,
source edge-case risks and explicit acceptance of official test selection. Any
algorithm bug fix requires a separate version and stated deviation. SSB remains
NOT RUN and cannot be launched by this CLI.

## DEAN-0 local validation record

Local `.venv`: Python 3.11.9, torch 2.0.0+cu118, sklearn 1.3.0. This is the
existing project environment, not the candidate official environment. No packages
were installed/upgraded. All component tensors ran on CPU; no GPU training ran.

- Full suite with external checkout enabled: **175 tests PASS**, 130.287 seconds.
- After adding source-rejection/metadata-negative tests: **10 DEAN tests PASS**,
  10.753 seconds; the earlier full suite included eight of these ten tests.
- Standalone CPU smoke, plan, project CLI help and compilation: PASS.
- Byte-equivalence tests for prior code/config/results/requirements and the
  baseline tag: PASS. Offline reproduction comparison: `differences: {}`.
- Negative preflight using intentionally missing placeholder assets: exits 1,
  correctly reports candidate-environment mismatches and absent CUB/checkpoint.
- Complete upstream `--help` import in the current environment: **BLOCKED** by
  missing `pytorch_metric_learning`; `einops` is also absent by package inspection.
  These are included in the separate candidate environment, whose install is NOT RUN.
- Source license, effective full-checkpoint key matching, real assets, complete
  target-environment imports and Linux environment solve remain unverified gates.
