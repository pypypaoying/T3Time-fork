#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

device="${DEVICE:-cuda}"
steps="${BENCHMARK_STEPS:-100}"
val_steps="${BENCHMARK_VAL_STEPS:-30}"
num_workers="${NUM_WORKERS:-4}"
embedding_mode="${EMBEDDING_MODE:-zeros}"
embed_root="${EMBED_ROOT:-./Embeddings}"

run_benchmark() {
  local horizon="$1"
  local learning_rate channel e_layer d_layer dropout batch_size

  case "$horizon" in
    96)
      learning_rate=1e-3; channel=64; e_layer=6; d_layer=2; dropout=0.1; batch_size=32 ;;
    192)
      learning_rate=1e-4; channel=32; e_layer=1; d_layer=2; dropout=0.1; batch_size=32 ;;
    336)
      learning_rate=1e-4; channel=64; e_layer=1; d_layer=6; dropout=0.5; batch_size=32 ;;
    720)
      learning_rate=1e-4; channel=128; e_layer=1; d_layer=1; dropout=0.25; batch_size=64 ;;
    *)
      echo "Unsupported horizon: $horizon" >&2
      return 2 ;;
  esac

  python -u train.py \
    --device "$device" \
    --root_path ./dataset \
    --embed_root "$embed_root" \
    --embedding_mode "$embedding_mode" \
    --data_path Weather \
    --num_nodes 21 \
    --seq_len 336 \
    --pred_len "$horizon" \
    --epochs 1 \
    --seed 2024 \
    --batch_size "$batch_size" \
    --channel "$channel" \
    --learning_rate "$learning_rate" \
    --dropout_n "$dropout" \
    --e_layer "$e_layer" \
    --d_layer "$d_layer" \
    --num_workers "$num_workers" \
    --max_train_steps "$steps" \
    --max_val_steps "$val_steps" \
    --skip_test \
    --save ./logs/benchmark-weather-336
}

for horizon in 96 192 336 720; do
  echo "===== Weather seq=336 horizon=${horizon} ====="
  run_benchmark "$horizon"
done
