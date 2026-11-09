#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VAL_DATASET_DIR="/home/nvme/lijiawang/LIBERO_IsaacSim_Lerobot_openwam_eef10_trainval/val"
if [[ ! -f "$VAL_DATASET_DIR/meta/info.json" ]]; then
  echo "The draft train/val dataset was removed during data cleanup." >&2
  echo "This script computes offline held-out loss, not IsaacSim rollout success." >&2
  echo "Select the evaluation protocol before recreating the draft views with dataset_tools/create_openwam_trainval_views.py." >&2
  exit 7
fi

OPENWAM_VENV="${OPENWAM_VENV:-/home/nvme/lijiawang/.venvs/openwam-alpha}"
if [[ -f "$OPENWAM_VENV/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$OPENWAM_VENV/bin/activate"
else
  echo "OpenWAM venv not found: $OPENWAM_VENV" >&2
  exit 6
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

BASELINE_CKPT_ROOT="${BASELINE_CKPT_ROOT:-$PWD/outputs/openwam_alpha_libero_isaacsim_baseline_ckpts}"
PRETRAIN_CKPT_DIR="${PRETRAIN_CKPT_DIR:-$PWD/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
if [[ -z "${EVAL_CKPT_DIR:-}" ]]; then
  EVAL_CKPT_DIR=""
  if [[ -d "$BASELINE_CKPT_ROOT" ]]; then
    while IFS= read -r candidate; do
      if compgen -G "$candidate/checkpoint_step_*.safetensors" > /dev/null; then
        EVAL_CKPT_DIR="$candidate"
        break
      fi
    done < <(find "$BASELINE_CKPT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2-)
  fi
  if [[ -z "$EVAL_CKPT_DIR" ]]; then
    echo "No finetuned baseline checkpoint found; evaluating the pretrained OpenWAM-alpha checkpoint."
    EVAL_CKPT_DIR="$PRETRAIN_CKPT_DIR"
  fi
fi

EVAL_RUN_NAME="${EVAL_RUN_NAME:-openwam_alpha_libero_isaacsim_val_loss}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$PWD/outputs/$EVAL_RUN_NAME}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-$EVAL_OUTPUT_DIR/$(date +%Y-%m-%d_%H-%M-%S)_val_loss.json}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-null}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
EVAL_ZERO_STAGE="${EVAL_ZERO_STAGE:-2}"
EVAL_INIT_ON_CPU="${EVAL_INIT_ON_CPU:-true}"
EVAL_OFFLOAD_OPTIMIZER_DEVICE="${EVAL_OFFLOAD_OPTIMIZER_DEVICE:-none}"

mkdir -p "$EVAL_OUTPUT_DIR"

scripts/check_openwam_alpha_resources.sh

NNODES="${NNODES:-${HOST_NUM:-${WORLD_SIZE:-1}}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  scripts/eval_openwam_val_loss.py \
  model=dual_system \
  dataloader=libero_isaacsim_openwam_val \
  training.finetune_ckpt_path="$EVAL_CKPT_DIR" \
  training.batch_size="$EVAL_BATCH_SIZE" \
  training.dataset_num_workers="$EVAL_NUM_WORKERS" \
  training.zero_stage="$EVAL_ZERO_STAGE" \
  training.initialize_model_on_cpu="$EVAL_INIT_ON_CPU" \
  training.offload_optimizer_device="$EVAL_OFFLOAD_OPTIMIZER_DEVICE" \
  +evaluation.split=val \
  +evaluation.batch_size="$EVAL_BATCH_SIZE" \
  +evaluation.dataset_num_workers="$EVAL_NUM_WORKERS" \
  +evaluation.max_batches="$EVAL_MAX_BATCHES" \
  +evaluation.output_path="$EVAL_OUTPUT_PATH" \
  project.output_dir="$EVAL_OUTPUT_DIR" \
  project.wandb.run_name="$EVAL_RUN_NAME"

echo "Validation loss written to $EVAL_OUTPUT_PATH"
