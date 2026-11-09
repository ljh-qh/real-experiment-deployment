#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_SESSION="openwam_alpha_baseline_6ep"
RUN_STATE_DIR="$PWD/outputs/openwam_alpha_libero_isaacsim_baseline_6ep_seed42_20260914_2358_ckpts/2026-09-14_23-38-07"
STATE_MARKER="$RUN_STATE_DIR/accel_state_step_88000/trainer_state.json"
WEIGHTS="$RUN_STATE_DIR/checkpoint_step_88000.safetensors"
LOG_DIR="$PWD/outputs/openwam_alpha_libero_isaacsim_baseline_6ep_seed42_20260914_2358_launcher_logs"
RESUME_LOG="$LOG_DIR/train_bs2ga8_resume.typescript"

while [[ ! -f "$STATE_MARKER" ]]; do
  step="$(tmux capture-pane -pt "$TRAIN_SESSION" -S -3 2>/dev/null | tr '\r' '\n' | sed -n 's/.*Training:.*|[[:space:]]*\([0-9][0-9]*\)\/252822.*/\1/p' | tail -1)"
  echo "[$(date --iso-8601=seconds)] waiting for step 88000 full state; current=${step:-unknown}"
  sleep 30
done

python3 - "$STATE_MARKER" "$WEIGHTS" <<'PY'
import json
import os
import sys

marker, weights = sys.argv[1:]
with open(marker) as handle:
    state = json.load(handle)
if state.get("global_step") != 88000 or state.get("opt_step") != 5500:
    raise SystemExit(f"unexpected checkpoint metadata: {state}")
if not os.path.isfile(weights) or os.path.getsize(weights) < 20_000_000_000:
    raise SystemExit(f"missing or incomplete weights: {weights}")
print(f"validated checkpoint: {state}, weights_bytes={os.path.getsize(weights)}")
PY

echo "[$(date --iso-8601=seconds)] stopping batch=1 training after completed checkpoint 88000"
tmux send-keys -t "$TRAIN_SESSION" C-c

for _ in $(seq 1 60); do
  if ! pgrep -f 'scripts/train.py.*openwam_alpha_libero_isaacsim_baseline_6ep_seed42_20260914_2358' >/dev/null; then
    break
  fi
  sleep 2
done

tmux kill-session -t "$TRAIN_SESSION" 2>/dev/null || true

for _ in $(seq 1 60); do
  if ! pgrep -f 'scripts/train.py.*openwam_alpha_libero_isaacsim_baseline_6ep_seed42_20260914_2358' >/dev/null; then
    break
  fi
  sleep 2
done

if pgrep -f 'scripts/train.py.*openwam_alpha_libero_isaacsim_baseline_6ep_seed42_20260914_2358' >/dev/null; then
  echo "training workers did not exit after checkpoint-safe shutdown" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
tmux new-session -d -s "$TRAIN_SESSION" \
  "cd '$PWD' && exec script -qefc 'bash scripts/resume_openwam_alpha_libero_isaacsim_bs2ga8.sh' '$RESUME_LOG'"
echo "[$(date --iso-8601=seconds)] launched batch=2 accumulation=8 in tmux $TRAIN_SESSION"
echo "log=$RESUME_LOG"
