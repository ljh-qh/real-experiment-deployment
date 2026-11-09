#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

RUN_STATE_DIR="${RUN_STATE_DIR:-$PWD/outputs/openwam_alpha_libero_isaacsim_baseline_6ep_seed42_20260914_2358_ckpts/2026-09-14_23-38-07}"
export RESUME_CKPT_DIR="$RUN_STATE_DIR"
export RESUME_SOURCE_BATCH_SIZE="${RESUME_SOURCE_BATCH_SIZE:-1}"
export RESUME_SOURCE_GRAD_ACCUM="${RESUME_SOURCE_GRAD_ACCUM:-16}"
export RUN_NAME="${RUN_NAME:-openwam_alpha_libero_isaacsim_baseline_6ep_seed42_20260914_2358}"
export RUN_DIR="${RUN_DIR:-$PWD/outputs/$RUN_NAME}"
export CKPT_OUT="${CKPT_OUT:-$PWD/outputs/${RUN_NAME}_ckpts}"
export BASELINE_BATCH_SIZE=2
export BASELINE_GRAD_ACCUM=8

echo "[batch-migration] trying batch=2 accumulation=8 from $RUN_STATE_DIR"
set +e
bash scripts/train_openwam_alpha_libero_isaacsim_baseline.sh
status=$?
set -e

if (( status == 0 )); then
  exit 0
fi

echo "[batch-migration] batch=2 run exited with status $status; restoring batch=1 accumulation=16"
sleep 30
export BASELINE_BATCH_SIZE=1
export BASELINE_GRAD_ACCUM=16
exec bash scripts/train_openwam_alpha_libero_isaacsim_baseline.sh
