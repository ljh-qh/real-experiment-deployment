#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=${OPENPI_ROOT:-/mnt/dataset/csx_workspace/openpi}
PORT=${PORT:-8010}
UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
: "${JOINT_CKPT_DIR:?Set JOINT_CKPT_DIR to the local joint checkpoint step dir, e.g. /path/to/pi05_tianji_sharpa_pour_drink_joint_absolute/.../29000}"

export UV_DEFAULT_INDEX
export UV_HTTP_TIMEOUT
unset VIRTUAL_ENV

cd "$OPENPI_ROOT"
uv run --no-dev scripts/serve_policy.py --port "$PORT" policy:checkpoint \
  --policy.config=pi05_tianji_sharpa_pour_drink_joint_absolute \
  --policy.dir="$JOINT_CKPT_DIR"
