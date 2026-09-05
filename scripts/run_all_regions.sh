#!/usr/bin/env bash
set -euo pipefail

# Reproduce the four-region NGMM evaluation with region-specific settings.
# Data must be available under DATA_ROOT/<region>/obs and optional era5.
DATA_ROOT="${DATA_ROOT:-./processed_data}"
USE_ERA5="${USE_ERA5:-0}"
RUNS="${RUNS:-5}"
EPOCHS="${EPOCHS:-100}"
OUT_ROOT="${OUT_ROOT:-./results/all_regions}"

for dataset in France Australia Zhejiang Xinjiang; do
  batch=16
  lr=5e-4
  wd=1e-4
  kernel=3
  if [[ "$dataset" == "France" ]]; then kernel=5; fi
  if [[ "$dataset" == "Australia" || "$dataset" == "Zhejiang" ]]; then lr=8e-4; fi
  if [[ "$dataset" == "Xinjiang" ]]; then batch=1; wd=3e-4; fi

  python main.py \
    --dataset "$dataset" --data_root "$DATA_ROOT" --use_era5 "$USE_ERA5" \
    --hidden_dim 256 --num_layers 3 --num_heads 4 --kernel_size "$kernel" \
    --seq_len 24 --train_stride 12 --lr "$lr" --weight_decay "$wd" \
    --beta 0.1 --missing_penalty 2.0 --temporal_weight 0.1 \
    --wind_vector_weight 0.025 --adaptive_temporal_scale 1 \
    --pointwise_loss huber --train_missing_type point_mcar \
    --train_missing_ratio 0.2 --val_missing_ratio 0.1 \
    --batch_size "$batch" --epochs "$EPOCHS" --num_runs "$RUNS" \
    --output_dir "$OUT_ROOT/$dataset"
done
