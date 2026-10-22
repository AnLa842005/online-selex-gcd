# Local validation — 2026-10-22

Official upstream: `c3d3b42a0aa7d7c9ea748a8305a6890a34100f73`.

## Machine and environment

- Windows 11; PowerShell 5.1.26100.9711.
- Git 2.51.0.windows.1; GitHub CLI 2.98.0, authenticated.
- Python 3.11 and 3.12 installations found; no Conda/Mamba in PATH or the common
  user installation locations checked. A separate repository venv uses Python 3.11.9.
- Approximately 33 GiB free before setup; the destination repository did not exist.
- Actual GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB; driver 566.07.
- torch 2.0.0+cu118, torchvision 0.15.1+cu118, CUDA runtime 11.8, CUDA available.
- timm 0.9.10, scikit-learn 1.3.0. Editable kmeans source pinned as documented.
- `pip check`: PASS. The parent Python 3.12 environment was not modified.

## Data and backbone

- Official Caltech CUB archive MD5: PASS.
- All 11,788 images verified with PIL, all required metadata and attributes present.
- 200 classes; official SSB split: 100 old / 100 novel, disjoint and exhaustive.
- Official loader counts: 1,498 labelled / 4,496 unlabelled / 5,794 test.
- DINOv1 ViT-B/16 backbone-only checkpoint from Meta: strict load PASS.
- Dataset, checkpoint, run artifacts and logs are outside this repository.

## Smoke outcome

```powershell
powershell -ExecutionPolicy Bypass -File scripts/smoke_cub_local.ps1 -Device cpu -BatchSize 2
```

The Python smoke completed with a PASS report. The PowerShell wrapper's explicit
success exit was then normalized and tested with success/stderr and failure
processes: success returns 0, failure returns nonzero. This wrapper-only adjustment
does not change the validated Python smoke computation.

| Check | Result |
| --- | --- |
| Real dataset loader and SSB split | PASS |
| Strict pretrained backbone load | PASS |
| HSSK and all hierarchical levels | PASS |
| Forward / finite gradients / backward / optimizer step | PASS |
| USE | 1.2040894031524658 |
| SSE | 1.7597784996032715 |
| Total loss | 1.0906193256378174 |
| Evaluation on 200 held-out images | PASS |
| Model and projection-head save/reload, finite parameters | PASS |
| Epoch training-state checkpoint save | PASS |
| Runtime | 267.468 seconds |
| Device | CPU |
| Peak CUDA VRAM for successful run | Not applicable |

The debug subset retains all 200 SSB classes, with 100 labelled and 300 unlabelled
training images. One batch and one epoch executed. Accuracy from this tiny smoke
is intentionally omitted: it is not a baseline result.

GPU attempts at batch sizes 4 and 2 ran out of memory in HSSK. The first CPU
attempt exposed Windows NumPy int32 cluster indices; explicit int64 indices fixed
that portability problem. The successful CPU smoke includes this fix. No full
training was run locally.

## Checks and remaining scope

- Six unit/regression tests PASS: paths, debug limit, checkpoint/RNG restoration,
  finite loss/backward, CPU HSSK integer portability, upstream algorithm parity.
- Parity covers the loss classes, USE logits, HSSK extraction, evaluation function,
  training loss block, clustering implementation and metric definitions, allowing
  only documented device/import/integer-width portability substitutions.
- Bash server script syntax PASS; Python compilation and Git whitespace checks PASS.
- New/modified file audit: no model/data/log artifacts, user absolute paths, or
  common credential patterns. LICENSE and authors' attribution retained.
- Full Linux/A6000 training, full final-report subprocess execution and paper
  metric reproduction remain unexecuted. They must be validated on the server.
- GitHub visibility is private, following the request's initial default because
  its later public/anonymous-clone instructions conflict. Anonymous cloning of a
  private repository is unavailable; authenticated server access is required.

No algorithmic changes. The upstream order-bias caveat in BASELINE_SETUP.md applies.
