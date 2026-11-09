#!/usr/bin/env bash
set -euo pipefail

URI_ROOT=${URI_ROOT:-/mnt/dataset/csx_workspace/Ideas/AF3/code/openpi/third_party/Universal_Robot_Infra}
POLICY_HOST=${POLICY_HOST:-127.0.0.1}
PORT=${PORT:-8010}
FRAMES=${FRAMES:-20}

cd "$URI_ROOT"
python -m unirobot_infra.apps.openpi_policy_control_loop \
  --config-name apps/openpi_policy_control_loop/delta_ee_step_blocking \
  policy.server.host="$POLICY_HOST" \
  policy.server.port="$PORT" \
  env.runtime.frames="$FRAMES" \
  "$@"
