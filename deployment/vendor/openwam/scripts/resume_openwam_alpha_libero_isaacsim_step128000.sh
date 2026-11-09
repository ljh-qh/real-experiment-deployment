#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/nvme/lijiawang/OpenWAM}"
RUN_STATE_DIR="${RUN_STATE_DIR:-$REPO_DIR/outputs/openwam_alpha_libero_isaacsim_baseline_6ep_seed42_20260914_2358_ckpts/2026-09-14_23-38-07}"
EXPECTED_STATE="$RUN_STATE_DIR/accel_state_step_128000/trainer_state.json"
SESSION_NAME="${SESSION_NAME:-openwam_alpha_baseline_6ep}"
LOG_DIR="$REPO_DIR/outputs/openwam_alpha_libero_isaacsim_baseline_6ep_seed42_20260914_2358_launcher_logs"
LOG_PATH="${LOG_PATH:-$LOG_DIR/train_after_reallocation.typescript}"

if [[ ! -s "$EXPECTED_STATE" ]]; then
  echo "Missing full resume state: $EXPECTED_STATE" >&2
  exit 1
fi

python3 - "$EXPECTED_STATE" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    state = json.load(handle)

expected = {
    "global_step": 128000,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "world_size": 8,
}
for key, value in expected.items():
    if state.get(key) != value:
        raise SystemExit(f"Unexpected {key}: {state.get(key)!r}; expected {value!r}")
print(f"Validated resume state: {state}")
PY

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

printf -v train_command \
  'exec env RUN_STATE_DIR=%q RESUME_CKPT_DIR=%q RESUME_SOURCE_BATCH_SIZE=2 RESUME_SOURCE_GRAD_ACCUM=8 RUN_NAME=openwam_alpha_libero_isaacsim_baseline_6ep_seed42_20260914_2358 RUN_DIR=%q CKPT_OUT=%q BASELINE_BATCH_SIZE=2 BASELINE_GRAD_ACCUM=8 bash scripts/train_openwam_alpha_libero_isaacsim_baseline.sh' \
  "$RUN_STATE_DIR" \
  "$RUN_STATE_DIR" \
  "$REPO_DIR/outputs/openwam_alpha_libero_isaacsim_baseline_6ep_seed42_20260914_2358" \
  "$REPO_DIR/outputs/openwam_alpha_libero_isaacsim_baseline_6ep_seed42_20260914_2358_ckpts"

printf -v tmux_command \
  'cd %q && exec script -qefc %q %q' \
  "$REPO_DIR" \
  "$train_command" \
  "$LOG_PATH"

tmux new-session -d -s "$SESSION_NAME" "$tmux_command"
echo "Started $SESSION_NAME from step 128000"
echo "Log: $LOG_PATH"
echo "Attach: tmux attach -t $SESSION_NAME"

