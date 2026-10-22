#!/usr/bin/env bash
set -euo pipefail
: "${SELEX_DATA_ROOT:?Set the parent of CUB_200_2011}"
: "${SELEX_PRETRAINED_PATH:?Set the DINO ViT-B/16 checkpoint}"
: "${SELEX_OUTPUT_ROOT:?Set the output parent directory}"
export CUDA_VISIBLE_DEVICES="${SELEX_GPU:-0}"
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${SELEX_PYTHON:-python}"
mkdir -p "$SELEX_OUTPUT_ROOT"
RUN="$(mktemp -d "$SELEX_OUTPUT_ROOT/cub-full-XXXXXXXX")"
export SELEX_OUTPUT_ROOT="$RUN"
cd "$RUN"
mkdir Plots
EXTRA=()
if [[ -n "${SELEX_RESUME:-}" ]]; then EXTRA+=(--resume "$SELEX_RESUME"); fi
"$PYTHON" -u -m methods.contrastive_training.contrastive_training \
  --dataset_name cub --base_model vit_dino --batch_size 128 --epochs 200 \
  --grad_from_block 10 --num_workers 4 --use_ssb_splits True \
  --prop_train_labels 0.5 --sup_con_weight 0.35 --weight_decay 1e-4 \
  --contrast_unlabel_only False --transform imagenet --lr 0.1 \
  --momentum 0.9 --temperature 1.0 --n_views 2 --seed 1 \
  --strategy zero_one --cluster_momentum 1 --unsupervised_smoothing 1.0 \
  --prototype_extraction_interval 1 --train_report_interval 200 \
  --distance euclidean --eval_funcs v1 v2 --unbalanced False \
  --gpu_clustering True --report True --device cuda:0 "${EXTRA[@]}" \
  2>&1 | tee "$RUN/train.log"
