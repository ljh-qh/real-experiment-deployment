#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec "$SCRIPT_DIR/control_delta_ee_chunk_dry_run.sh" \
  policy.execution.dry_run=false \
  policy.execution.confirm_real_execution=true \
  "$@"
