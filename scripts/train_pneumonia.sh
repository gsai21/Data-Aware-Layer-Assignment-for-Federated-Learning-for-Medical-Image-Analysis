#!/usr/bin/env bash
set -e

python -m dafl.run \
  --dataset_path "/kaggle/input/chest-xray-pneumonia/chest_xray" \
  --num_clients 10 \
  --rounds 50 \
  --epochs_per_client 2 \
  --batch_size 16 \
  --lr 1e-4 \
  --alpha 0.5 \
  --max_influence_batches 10 \
  --sample_count_weight 0.3 \
  --out_dir artifacts/pneumonia