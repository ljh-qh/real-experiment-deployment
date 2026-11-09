#!/usr/bin/env bash
set -euo pipefail

# Conservative preflight for OpenWAM-α. Defaults assume an 8×H100 node and
# avoid launching while another job is occupying the cards.
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-70000}"
MAX_GPU_UTIL_PCT="${MAX_GPU_UTIL_PCT:-10}"
STABLE_CHECKS="${GPU_STABLE_CHECKS:-3}"
STABLE_SLEEP="${GPU_STABLE_SLEEP:-10}"
MIN_DISK_FREE_GB="${MIN_DISK_FREE_GB:-80}"
DISK_PATH="${DISK_PATH:-$PWD}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  if ! gpu_indices=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits); then
    echo "Cannot query NVIDIA GPUs." >&2
    exit 1
  fi
  mapfile -t GPU_IDS <<< "$gpu_indices"
fi

if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "No GPUs detected." >&2
  exit 1
fi

free_gb=$(df -BG --output=avail "$DISK_PATH" | tail -1 | tr -dc '0-9')
echo "Disk $DISK_PATH free: ${free_gb}G"
if [[ -z "$free_gb" || "$free_gb" -lt "$MIN_DISK_FREE_GB" ]]; then
  echo "Disk preflight failed: need >= ${MIN_DISK_FREE_GB}G free on $DISK_PATH." >&2
  exit 2
fi

check_once() {
  local failed=0
  local gpu free util used want gpu_status
  local -A seen=()
  if ! gpu_status=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu,memory.used --format=csv,noheader,nounits); then
    echo "Cannot query NVIDIA GPU resources." >&2
    return 1
  fi
  while IFS=',' read -r gpu free util used; do
    gpu="${gpu//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    [[ -n "$gpu" ]] || continue
    for want in "${GPU_IDS[@]}"; do
      want="${want//[[:space:]]/}"
      if [[ "$gpu" == "$want" ]]; then
        seen["$want"]=1
        echo "GPU $gpu: free=${free}MiB used=${used}MiB util=${util}%"
        if [[ ! "$free" =~ ^[0-9]+$ || ! "$util" =~ ^[0-9]+$ ]]; then
          echo "GPU $gpu returned invalid resource values." >&2
          failed=1
        elif [[ "$free" -lt "$MIN_FREE_GPU_MB" || "$util" -gt "$MAX_GPU_UTIL_PCT" ]]; then
          failed=1
        fi
      fi
    done
  done <<< "$gpu_status"
  for want in "${GPU_IDS[@]}"; do
    want="${want//[[:space:]]/}"
    if [[ -z "$want" || -z "${seen[$want]:-}" ]]; then
      echo "Selected GPU '$want' was not found; use numeric GPU indices." >&2
      failed=1
    fi
  done
  return "$failed"
}

for ((i=1; i<=STABLE_CHECKS; i++)); do
  echo "Resource check ${i}/${STABLE_CHECKS} for GPUs: ${GPU_IDS[*]}"
  if ! check_once; then
    echo "GPU preflight failed: need each selected GPU free >= ${MIN_FREE_GPU_MB}MiB and util <= ${MAX_GPU_UTIL_PCT}%." >&2
    nvidia-smi pmon -c 1 || true
    exit 3
  fi
  if [[ "$i" -lt "$STABLE_CHECKS" ]]; then
    sleep "$STABLE_SLEEP"
  fi
done

echo "Resource preflight passed."
