#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CONFIRM_REAL_ROBOT:?Set CONFIRM_REAL_ROBOT=1 to send real robot actions.}"
if [[ "$CONFIRM_REAL_ROBOT" != "1" ]]; then
  echo "CONFIRM_REAL_ROBOT must be 1 to send real robot actions." >&2
  exit 2
fi

exec "$SCRIPT_DIR/control_delta_ee_step_dry_run.sh" \
  policy.execution.dry_run=false \
  policy.execution.confirm_real_execution=true \
  "$@"
