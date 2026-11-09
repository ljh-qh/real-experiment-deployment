#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OPENWAM_VENV="${OPENWAM_VENV:-$PWD/../venv}"
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
RESUME_CKPT_DIR="${RESUME_CKPT_DIR:-}"
RUN_NAME="${RUN_NAME:-openwam_alpha_libero_isaacsim_baseline}"
RUN_DIR="${RUN_DIR:-$PWD/outputs/$RUN_NAME}"
CKPT_OUT="${CKPT_OUT:-$PWD/outputs/${RUN_NAME}_ckpts}"

# One micro-batch already uses about 53 GiB on an H100. The initial resource-
# adapted run used batch 1 x accumulation 16; batch 2 x accumulation 8 keeps
# the same effective global batch of 128 on eight GPUs when it fits.
# Six epochs is the pre-declared baseline budget: the default five plus one.
BASELINE_MAX_STEPS="${BASELINE_MAX_STEPS:-null}"
BASELINE_NUM_EPOCHS="${BASELINE_NUM_EPOCHS:-6}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-1}"
BASELINE_GRAD_ACCUM="${BASELINE_GRAD_ACCUM:-16}"
# Keep no periodic checkpoints for this disk-constrained recovery. A positive
# value above the total step count is intentional: save_steps=0 also disables
# the trainer's final weight artifact. The final save is ~24.8 GB; the initial
# resume state is removed by finalize_keep_weights_only after it lands.
BASELINE_SAVE_STEPS="${BASELINE_SAVE_STEPS:-1000000000}"
BASELINE_KEEP_LAST_K="${BASELINE_KEEP_LAST_K:-1}"
BASELINE_SAVE_FULL_STATES="${BASELINE_SAVE_FULL_STATES:-false}"
BASELINE_ZERO_STAGE="${BASELINE_ZERO_STAGE:-2}"
BASELINE_OFFLOAD_OPTIMIZER_DEVICE="${BASELINE_OFFLOAD_OPTIMIZER_DEVICE:-none}"
BASELINE_INIT_ON_CPU="${BASELINE_INIT_ON_CPU:-true}"
BASELINE_LR="${BASELINE_LR:-1e-4}"
BASELINE_NUM_WORKERS="${BASELINE_NUM_WORKERS:-4}"
RESUME_SOURCE_BATCH_SIZE="${RESUME_SOURCE_BATCH_SIZE:-null}"
RESUME_SOURCE_GRAD_ACCUM="${RESUME_SOURCE_GRAD_ACCUM:-null}"

if [[ -n "$RESUME_CKPT_DIR" ]]; then
  CHECKPOINT_ARGS=(
    training.finetune_ckpt_path=null
    training.resume_ckpt_path="$RESUME_CKPT_DIR"
  )
else
  CHECKPOINT_ARGS=(
    training.finetune_ckpt_path="$CKPT_DIR"
    training.resume_ckpt_path=null
  )
fi

scripts/check_openwam_alpha_resources.sh

bash scripts/train.sh \
  model=dual_system \
  dataloader=libero_isaacsim_openwam \
  "${CHECKPOINT_ARGS[@]}" \
  training.max_steps="$BASELINE_MAX_STEPS" \
  training.num_epochs="$BASELINE_NUM_EPOCHS" \
  training.batch_size="$BASELINE_BATCH_SIZE" \
  training.gradient_accumulation_steps="$BASELINE_GRAD_ACCUM" \
  training.resume_source_batch_size="$RESUME_SOURCE_BATCH_SIZE" \
  training.resume_source_gradient_accumulation_steps="$RESUME_SOURCE_GRAD_ACCUM" \
  training.dataset_num_workers="$BASELINE_NUM_WORKERS" \
  training.learning_rate="$BASELINE_LR" \
  training.save_steps="$BASELINE_SAVE_STEPS" \
  training.keep_last_k_ckpts="$BASELINE_KEEP_LAST_K" \
  training.save_full_states_for_resume="$BASELINE_SAVE_FULL_STATES" \
  training.zero_stage="$BASELINE_ZERO_STAGE" \
  training.use_gradient_checkpointing=true \
  training.initialize_model_on_cpu="$BASELINE_INIT_ON_CPU" \
  training.offload_optimizer_device="$BASELINE_OFFLOAD_OPTIMIZER_DEVICE" \
  training.output_path="$CKPT_OUT" \
  project.output_dir="$RUN_DIR" \
  project.wandb.run_name="$RUN_NAME"
