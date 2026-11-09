#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OPENWAM_VENV="${OPENWAM_VENV:-/home/nvme/lijiawang/.venvs/openwam-alpha}"
if [[ -f "$OPENWAM_VENV/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$OPENWAM_VENV/bin/activate"
else
  echo "OpenWAM venv not found: $OPENWAM_VENV" >&2
  exit 6
fi

RUN_NAME="${RUN_NAME:-openwam_alpha_libero_isaacsim_baseline}"
LOG_DIR="${LOG_DIR:-$PWD/outputs/${RUN_NAME}_launcher_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$(date +%Y-%m-%d_%H-%M-%S)_wait_and_train.log}"
LOCK_FILE="${LOCK_FILE:-/tmp/${RUN_NAME}.lock}"
MAX_WAIT_MINUTES="${MAX_WAIT_MINUTES:-720}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another $RUN_NAME launcher is already running (lock: $LOCK_FILE)." >&2
  exit 4
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -r -a _OPENWAM_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  export NPROC_PER_NODE="${#_OPENWAM_GPU_IDS[@]}"
fi

export MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-70000}"
export MAX_GPU_UTIL_PCT="${MAX_GPU_UTIL_PCT:-10}"
export GPU_STABLE_CHECKS="${GPU_STABLE_CHECKS:-3}"
export GPU_STABLE_SLEEP="${GPU_STABLE_SLEEP:-10}"
export BASELINE_MAX_STEPS="${BASELINE_MAX_STEPS:-null}"
export BASELINE_NUM_EPOCHS="${BASELINE_NUM_EPOCHS:-6}"
export BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-1}"
export BASELINE_GRAD_ACCUM="${BASELINE_GRAD_ACCUM:-16}"
export BASELINE_SAVE_STEPS="${BASELINE_SAVE_STEPS:-4000}"
export BASELINE_KEEP_LAST_K="${BASELINE_KEEP_LAST_K:-2}"
export BASELINE_SAVE_FULL_STATES="${BASELINE_SAVE_FULL_STATES:-true}"
export BASELINE_ZERO_STAGE="${BASELINE_ZERO_STAGE:-2}"
export BASELINE_OFFLOAD_OPTIMIZER_DEVICE="${BASELINE_OFFLOAD_OPTIMIZER_DEVICE:-none}"
export BASELINE_INIT_ON_CPU="${BASELINE_INIT_ON_CPU:-true}"
export BASELINE_LR="${BASELINE_LR:-1e-4}"
export BASELINE_NUM_WORKERS="${BASELINE_NUM_WORKERS:-4}"

{
  echo "[$(date --iso-8601=seconds)] OpenWAM-α LIBERO IsaacSim launcher starting"
  echo "Log: $LOG_FILE"
  echo "GPUs: $CUDA_VISIBLE_DEVICES | nproc=$NPROC_PER_NODE | min_free=${MIN_FREE_GPU_MB}MiB | max_util=${MAX_GPU_UTIL_PCT}%"
  echo "Baseline: max_steps=$BASELINE_MAX_STEPS num_epochs=$BASELINE_NUM_EPOCHS batch=$BASELINE_BATCH_SIZE grad_accum=$BASELINE_GRAD_ACCUM"

  deadline=$(( $(date +%s) + MAX_WAIT_MINUTES * 60 ))
  while true; do
    now=$(date +%s)
    if (( now > deadline )); then
      echo "[$(date --iso-8601=seconds)] Timed out waiting for free GPUs after ${MAX_WAIT_MINUTES} minutes."
      exit 5
    fi

    echo "[$(date --iso-8601=seconds)] Checking resources..."
    if scripts/check_openwam_alpha_resources.sh; then
      echo "[$(date --iso-8601=seconds)] Resources are free; running 1-step smoke."
      scripts/train_openwam_alpha_libero_isaacsim_smoke.sh
      echo "[$(date --iso-8601=seconds)] Smoke passed; launching baseline."
      scripts/train_openwam_alpha_libero_isaacsim_baseline.sh
      echo "[$(date --iso-8601=seconds)] Baseline command finished."
      exit 0
    fi

    echo "[$(date --iso-8601=seconds)] GPUs busy; sleeping ${CHECK_INTERVAL_SECONDS}s."
    sleep "$CHECK_INTERVAL_SECONDS"
  done
} >> "$LOG_FILE" 2>&1
