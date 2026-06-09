#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACCELERATE="${ACCELERATE:-accelerate}"
DATA_FILE="${PROJECT_DIR}/dataset/ETT-small/ETTm1.csv"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TEST_BATCHES="${TEST_BATCHES:-0}"
MASTER_PORT="${MASTER_PORT:-29600}"

if ! command -v "${ACCELERATE}" &> /dev/null; then
  echo "accelerate command not found: ${ACCELERATE}" >&2
  exit 1
fi

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "dataset not found: ${DATA_FILE}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

for pred_len in 96 192 336 720; do
  CHECKPOINT="${PROJECT_DIR}/rts_ckpt/ETTm1/${pred_len}/checkpoint"
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
  fi

  echo "========================================="
  echo "Evaluating ETTm1 with pred_len=${pred_len}"
  echo "========================================="

  "${ACCELERATE}" launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    --main_process_port "${MASTER_PORT}" \
    run_main.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --checkpoint_path "${CHECKPOINT}" \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_512_${pred_len}_test \
    --model RTSLLM \
    --model_comment ETTm1-${pred_len}-test \
    --data ETTm1 \
    --features M \
    --seq_len 512 \
    --label_len 48 \
    --pred_len "${pred_len}" \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 32 \
    --d_ff 128 \
    --n_heads 8 \
    --patch_len 16 \
    --stride 8 \
    --llm_model BERT \
    --llm_dim 768 \
    --llm_layers 12 \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --test_batches "${TEST_BATCHES}" \
    --itr 1
done
