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

CKPT_DIR="${CKPT_DIR:-$PWD/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
RUN_DIR="${RUN_DIR:-$PWD/outputs/openwam_alpha_libero_isaacsim_smoke}"
CKPT_OUT="${CKPT_OUT:-$PWD/outputs/openwam_alpha_libero_isaacsim_smoke_ckpts}"

scripts/check_openwam_alpha_resources.sh

bash scripts/train.sh \
  model=dual_system \
  dataloader=libero_isaacsim_openwam \
  training.finetune_ckpt_path="$CKPT_DIR" \
  training.max_steps=1 \
  training.num_epochs=null \
  training.batch_size=1 \
  training.gradient_accumulation_steps=1 \
  training.dataset_num_workers=0 \
  training.save_steps=0 \
  training.keep_last_k_ckpts=1 \
  training.zero_stage=2 \
  training.use_gradient_checkpointing=true \
  training.initialize_model_on_cpu=true \
  training.offload_optimizer_device=none \
  training.output_path="$CKPT_OUT" \
  project.output_dir="$RUN_DIR" \
  project.wandb.run_name=openwam_alpha_libero_isaacsim_smoke
