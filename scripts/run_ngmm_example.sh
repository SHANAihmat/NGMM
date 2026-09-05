#!/usr/bin/env bash
set -euo pipefail

# Example: train NGMM without ERA5 on the prepared France split.
# Set DATA_ROOT to the directory containing France/obs/{train,val,test}.npy.
DATA_ROOT="${DATA_ROOT:-./processed_data}"
DATASET="${DATASET:-France}"
OUT_DIR="${OUT_DIR:-./results/example_${DATASET}}"

python main.py \
  --dataset "$DATASET" \
  --data_root "$DATA_ROOT" \
  --use_era5 0 \
  --hidden_dim 256 \
  --num_layers 3 \
  --num_heads 4 \
  --kernel_size 3 \
  --seq_len 24 \
  --train_stride 12 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --beta 0.1 \
  --missing_penalty 2.0 \
  --temporal_weight 0.1 \
  --wind_vector_weight 0.025 \
  --adaptive_temporal_scale 1 \
  --pointwise_loss huber \
  --train_missing_type point_mcar \
  --train_missing_ratio 0.2 \
  --val_missing_ratio 0.1 \
  --batch_size 16 \
  --epochs 100 \
  --num_runs 5 \
  --output_dir "$OUT_DIR"
