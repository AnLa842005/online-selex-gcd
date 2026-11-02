# Clean frozen-DINO CUB: authoritative Phase 3B result and Phase 3C hardening

This is the **first clean Online GCD baseline** for this project. The values below
are a transcription of the authoritative RunPod console record supplied by the
user, not a new run or an inferred result. Machine-readable source:
[`results/cub_frozen_dino_online_seed1.json`](../results/cub_frozen_dino_online_seed1.json).
The raw console log, full result JSON, manifests and persisted calibration were
not supplied. The run commit and timestamp remain unknown; the Phase 3C starting
commit is `85a9bca86a89d97f07b97d9dbec6874ca7543286`.

## Exposure and environment

The full offline SelEx checkpoint was **not used**. The frozen DINO backbone had
no CUB-specific training or novel-stream exposure. Generic ImageNet pretraining
is not a guarantee of no semantically related bird images. Do not directly compare
these metrics with offline SelEx **73.95%**: protocols, training exposure and
evaluation populations differ. The offline algorithm and reproduced result stay
unchanged.

- RunPod; Python 3.12; torch 2.8.0+cu128; torchvision 0.23.0+cu128.
- NVIDIA RTX PRO 4000 Blackwell, 24 GB.
- DINOv1 ViT-B/16, backbone-only checkpoint SHA-256:
  `bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e`.
- CUB-200-2011; seed 1; base 1498, stream 4496, probe 5794; known/novel 100/100.
- Complete DINO feature cache checksum:
  `c575eda6f7914828a8a77b032581d427da006982742f6b92d1f6cfe956434d3f`.
- Extraction completed 11788/11788 images; extraction and baseline exit codes 0.

## Recorded results

Session percentages are rounded console values. Missing exact per-session scores
and counts are not reconstructed from those rounded values. Processing seconds
are the original pre-optimization stream times, not Phase 3C benchmark results.

| Session | All (%) | Old (%) | New (%) | Active prototypes | Processing (s) |
|---|---:|---:|---:|---:|---:|
| 1 | 46.64 | 54.84 | 23.33 | 315 | 3.097560290945694 |
| 2 | 41.36 | 48.76 | 32.86 | 597 | 8.515799070941284 |
| 3 | 36.33 | 47.49 | 29.38 | 938 | 18.227227735798806 |
| 4 | 31.15 | 44.04 | 26.45 | 1375 | 36.412236660020426 |
| 5 | 29.13 | 41.54 | 26.75 | 2067 | 85.15443701297045 |

| Metric | All (value; count) | Old (value; count) | New (value; count) |
|---|---|---|---|
| Final session | 0.2912801484230056; 1617 | 0.4153846153846154; 260 | 0.2675018422991894; 1357 |
| Cumulative | 0.23309608540925267; 4496 | 0.30820547031354234; 1499 | 0.19552886219552887; 2997 |
| Held-out probe | 0.19295823265447015; 5794 | 0.36234396671289876; 2884 | 0.025085910652920963; 2910 |

Average online accuracy: **0.36920810510124125**. Average forgetting:
**0.06604838911435613**. Final active prototypes: **2067**; discovered novel
clusters: **1967**. The baseline severely over-fragments 100 novel classes into
1967 clusters. **Probe New accuracy is only 2.51%**. Processing time grows rapidly
with prototype count (3.10 → 8.52 → 18.23 → 36.41 → 85.15 seconds).

## Calibration artifact still required

The exact selected threshold is absent from the supplied console record.
`calibration.threshold` is `null` and status is `pending_import`. Do not substitute
a fixture threshold, a default or a threshold inferred from accuracy.

Import the original persisted RunPod artifact into a **new** record:

```bash
python scripts/import_cub_phase3b_calibration.py \
  --calibration /path/to/persisted/RunPod/calibration.json \
  --output /path/to/new/cub_frozen_dino_online_seed1_calibrated.json
```

The importer verifies the canonical artifact checksum, schema/method, pinned cache
identity, 768 dimensions, 1498 positive/negative samples, 100 known classes,
provenance hash syntax, candidate counts/objective, selected threshold and largest
threshold tie policy. It changes only the calibration field and retains the
entire artifact. It refuses a second import and existing output files. This is
structural/provenance validation, not a recomputation from features: the original
base-manifest hash was not supplied independently. The normal CPU runner remains
the stronger path for recomputing/validating calibration against actual base
features when a future real workload is authorized. No such workload ran here.

## Phase 3C implementation

Image paths are validated lexically before Path normalization. Absolute paths,
Windows drives/UNC, backslashes, NUL, colons and empty/dot/parent components are
rejected. Default extraction preflight does not resolve or stat every image.
Missing files raise FileNotFoundError with their path when opened. Optional
`--verify-images` scans existence explicitly. Checkpoint provenance uses lexical
`abspath`, not remote `resolve`. The configured dataset root and filesystem
symlinks are trusted; lexical validation prevents metadata traversal, not hostile
filesystem symlink replacement.

Import guards now distinguish absolute workload imports from unrelated relative
modules called `models`; explicit negative guard tests retain workload blocks.
DINO pretrained tensors use `weights_only=True`. Only trusted, project-generated
training resume uses `weights_only=False`, with the pickle trust boundary stated
at the loading site. Optimizer, scheduler and Torch/NumPy/Python RNG restoration
are covered without external checkpoints.

The bank caches sorted persistent IDs, row indices and a contiguous float64
normalized matrix with amortized capacity growth. Every stream sample still
queries the pre-update bank and applies its EMA immediately. Only the changed
row is replaced/appended; lexical insertion preserves tie order for long IDs.
Cache data is derived state and is rebuilt after JSON loading, never serialized.
Probe batches and calibration negatives use broadcast NumPy matrix-vector
operations. These preserve the scalar GEMV reduction order rather than introducing
GEMM rounding changes at exact candidate thresholds or ties. Calibration LOO
normalization and positive dot products remain unchanged.

The runner prints validation, calibration, initialization, session and probe
start/end messages with counts/times; the CLI prints final output paths. No hidden
evaluator labels are printed. Session performance adds stream assignment, probe
inference, metric computation and total-session seconds. Probe time measures
inference; other evaluator work, including state immutability checks, is counted
in metric computation. Total session time also includes reporting/state size
measurement. Legacy processing/evaluation fields are retained. `--blas-threads`
exposes CPU BLAS concurrency; default 1 avoids launch overhead for small GEMV
calls and the prior pool settings are restored afterward. More vCPUs alone do not
remove the sequential observation dependency; benchmark thread settings before
choosing a larger value. No GPU dependency is added to the runner.

## Equivalence and synthetic CPU benchmark

`tests/reference_online_v0.py` freezes bank, calibration and evaluator source from
starting commit `85a9bca` as an independent pre-optimization oracle. Tests compare
arrival cluster IDs, observation counts, vectors (absolute tolerance 1e-15), exact
saved state JSON, probe predictions/immutability, resume, candidate diagnostics,
all per-session metrics and final summaries. Synthetic and fake-cache CUB runs
match the reference exactly in this environment, excluding performance fields.
This establishes tested equivalence, not a claim of bitwise identity across every
BLAS implementation or hardware. No real feature cache was used for validation.

Reproduce the opt-in benchmark:

```bash
python scripts/benchmark_online_cpu.py
```

Observed on Windows, Python 3.11.9, NumPy 1.24.3; Gaussian seed 123; 256 initial
prototypes, 768 dimensions, 192 sequential observations, 256 probes; one BLAS
thread; median of 3 runs, without tracemalloc:

| Work | Reference seconds | Optimized seconds | Speedup |
|---|---:|---:|---:|
| Sequential stream | 2.560247099987464 | 0.03815700000268407 | 67.10x |
| Probe inference | 4.425987800001167 | 0.030264200002420694 | 146.24x |

Each benchmark repetition asserts exact IDs, probe predictions and serialized
learner state. Ordinary tests have no wall-clock speed threshold. These are
synthetic kernel timings, not a prediction of full CUB session speed: metric
matching, tracing and serialization still cost time.

## Validation and next task

See the Phase 3C validation notes below for executed checks. Configs, clean
protocol/manifest builders, offline methods/losses, dataset split and DINO feature
definition are unchanged. No real CUB workload, training, extraction, dataset
download, CUDA workload or network asset fetch is part of this change. Only the
user-supplied summary is tracked, not real logs, caches, datasets, checkpoints or
learner state.

**Exact next Phase 4 task:** import and audit the persisted Phase 3B
`calibration.json` first, then implement a separately versioned, CPU-testable
novel-prototype consolidation ablation to address over-fragmentation. Keep this
baseline unchanged; define merge criteria and persistent-ID handling, validate on
synthetic/fake-cache inputs, and predeclare an evaluation plan with identical
clean frozen-DINO exposure and protocol. Do not tune on probe labels. Real CUB
ablations require a separate authorized workload.

## Executed Phase 3C validation (2026-11-02)

- Python 3.11.9 / torch 2.0.0+cu118 local environment: **92 tests passed**
  (`python -m unittest discover -s tests -v`, 29.316 seconds on the final run).
  Includes all existing tests, fake-cache five-session CUB baseline, synthetic
  baseline, Network Volume regressions, calibration importer, reference
  equivalence, checkpoint/RNG and import guards.
- Python 3.12.10: `py -3.12 -m unittest discover -s tests -p
  test_online_import_guards.py -v` passed, including actual relative
  `charset_normalizer.models` and forbidden absolute workload imports.
- Python 3.12.10 / native torch 2.6.0+cu124: `py -3.12 -m unittest discover -s
  tests -p test_checkpoint_compatibility.py -v` passed. This exercises a native
  newer weights-only default and explicit-argument assertions. **Torch 2.8 is not
  installed locally**, so no native 2.8 result is claimed. The Python 3.12
  environment lacks SciPy/scikit-learn; the full suite ran in the project venv,
  while the two independent compatibility tests ran under Python 3.12.
- All Python files in `online_gcd`, `scripts`, `tests`, `project_utils` compiled.
- CLI `--help` passed for manifest builder, extractor, CUB runner, synthetic
  runner, calibration importer and CPU benchmark.
- `scripts/compare_reproduction_config.py`: **`differences: {}`**.
- `git diff --exit-code 85a9bca --` protected configs, protocol, manifest builder,
  SSB reader/data, models, methods/losses, offline evaluator, server launcher,
  `BASELINE_SETUP.md` and `REPRODUCTION_AUDIT.md`: unchanged.
- `git diff --check`: passed. The commit contains only source, tests,
  documentation and the explicitly requested console-summary JSON. No real cache,
  dataset, checkpoint, raw log or learner state is included.
