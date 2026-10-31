#!/usr/bin/env bash
# Evaluate existing CUB checkpoints only. No training or asset downloads.
set -euo pipefail
: "${MODEL:?Set absolute path to model.pt}"
: "${SELEX_DATA_ROOT:?Set parent of CUB_200_2011}"
: "${SELEX_PRETRAINED_PATH:?Set existing DINO checkpoint}"
: "${SELEX_OUTPUT_ROOT:?Set a new absolute output directory}"
PYTHON="${SELEX_PYTHON:-python}"
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# Fail on incomplete local data instead of letting the CUB loader fetch an archive.
export SELEX_CUB_LOCAL_ONLY=1
# Fast preflight checks metadata only. Set SELEX_VERIFY_ALL_IMAGES=1 to stat every image.
"$PYTHON" - <<'PY'
import os
from pathlib import Path
model = Path(os.environ['MODEL'])
output = Path(os.environ['SELEX_OUTPUT_ROOT'])
if not model.is_absolute() or model.name != 'model.pt' or not output.is_absolute():
    raise SystemExit('MODEL must be absolute model.pt; output must be absolute')
if output.exists():
    raise SystemExit('Output directory must be new to prevent stale feature reuse')
required = [model.parent / n for n in ('model.pt', 'model_best.pt', 'model_proj_head.pt', 'model_proj_head_best.pt')]
required.append(Path(os.environ['SELEX_PRETRAINED_PATH']))
root = Path(os.environ['SELEX_DATA_ROOT']) / 'CUB_200_2011'
required += [root / n for n in ('images.txt', 'image_class_labels.txt', 'train_test_split.txt', 'attributes/image_attribute_labels.txt')]
for directory in (root, root / 'images'):
    if not directory.is_dir():
        raise SystemExit('Missing required directory: ' + str(directory))
missing = [str(p) for p in required if not p.is_file()]
if missing:
    raise SystemExit('Missing required files: ' + ', '.join(missing))
if os.environ.get('SELEX_VERIFY_ALL_IMAGES') == '1':
    for row in (root / 'images.txt').read_text().splitlines():
        image = root / 'images' / row.split(maxsplit=1)[1]
        if not image.is_file():
            raise SystemExit('Missing dataset image: ' + str(image))
PY
RUN_ID="$(basename "$(dirname "$(dirname "$MODEL")")")"
mkdir -p "$SELEX_OUTPUT_ROOT"
cd "$SELEX_OUTPUT_ROOT"
printf 'Checkpoint\tAll (%%)\tOld (%%)\tNew (%%)\n' > summary.txt
for BEST in False True; do
  NAME=final
  if [[ "$BEST" == True ]]; then NAME=best; fi
  "$PYTHON" -u -m methods.clustering.extract_features --dataset cub --model_name vit_dino --warmup_model_dir "$MODEL" --use_best_model "$BEST" > "extract-$NAME.log" 2>&1
  "$PYTHON" -u -m methods.clustering.k_means --dataset_name cub --model_name vit_dino --warmup_model_exp_id "$RUN_ID" --use_best_model "$BEST" --use_ssb_splits True --prop_train_labels 0.5 --unbalanced False --eval_funcs v1 v2 --silent False > "evaluation-$NAME.log" 2>&1
  cp results.txt "results-$NAME.txt"
  "$PYTHON" "$REPO/scripts/summarize_cub_evaluation.py" "evaluation-$NAME.log" "$NAME" >> summary.txt
done
cat summary.txt
