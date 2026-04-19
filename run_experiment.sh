#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ARTIFACTS_DIR="$ROOT/artifacts"
LAST_RESULT_PATH="$ARTIFACTS_DIR/last_result.json"
TRAIN_TIMEOUT=86400
EVAL_TIMEOUT=3600

mkdir -p "$ARTIFACTS_DIR"

export CFLAGS="${CFLAGS:--I/usr/include}"
export LDFLAGS="${LDFLAGS:--L/usr/lib/x86_64-linux-gnu}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [[ "${DEBUG_CUDA:-0}" == "1" ]]; then
  export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
  export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-1}"
else
  export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
  export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-0}"
fi
if [[ -d "/usr/local/cuda/lib64" ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-true}"

run_python=(uv run python)
run_torchrun=(uv run torchrun)

write_failure_result() {
  local phase="$1"
  local exit_code="$2"
  local message="$3"
  PHASE="$phase" EXIT_CODE="$exit_code" MESSAGE="$message" LAST_RESULT_PATH="$LAST_RESULT_PATH" python3 - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "status": "error",
    "phase": os.environ["PHASE"],
    "exit_code": int(os.environ["EXIT_CODE"]),
    "message": os.environ["MESSAGE"],
}
path = Path(os.environ["LAST_RESULT_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
PY
}

if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv is not installed or not on PATH" >&2
  write_failure_result "preflight" 127 "uv is not installed or not on PATH"
  exit 127
fi

get_gpu_count() {
  "${run_python[@]}" -c "import torch; print(max(torch.cuda.device_count(), 1))"
}

extract_float() {
  local name="$1"
  local file_path="$2"
  NAME="$name" FILE_PATH="$file_path" "${run_python[@]}" - <<'PY'
import os
import pathlib
import re
import sys

name = os.environ["NAME"]
text = pathlib.Path(os.environ["FILE_PATH"]).read_text(encoding="utf-8")
match = re.search(r"^" + re.escape(name) + r":\s+([0-9.]+)", text, flags=re.MULTILINE)
if not match:
    raise SystemExit(f"Could not find {name} in output:\n{text}")
print(match.group(1))
PY
}

RESUME_ARGS=()
if [[ "${1:-}" == "--resume" ]]; then
  RESUME_ARGS+=(--resume)
elif [[ $# -gt 0 ]]; then
  echo "Usage: ./run_experiment.sh [--resume]" >&2
  exit 2
fi

GPU_COUNT="$(get_gpu_count)"
GPUS_PER_NODE="${GPUS_PER_NODE:-$GPU_COUNT}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6001}"

echo "Detected ${GPU_COUNT} GPU(s), launching training with torchrun" >&2
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all-visible>}" >&2

STARTED_AT="$(date +%s)"
TRAIN_OUTPUT="$(mktemp)"
EVAL_OUTPUT="$(mktemp)"
cleanup() {
  rm -f "$TRAIN_OUTPUT" "$EVAL_OUTPUT"
}
trap cleanup EXIT

DISTRIBUTED_ARGS=(
  --nproc_per_node "$GPUS_PER_NODE"
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
)

if timeout "$TRAIN_TIMEOUT" \
  "${run_torchrun[@]}" "${DISTRIBUTED_ARGS[@]}" train.py "${RESUME_ARGS[@]}" \
  >"$TRAIN_OUTPUT"; then
  :
else
  status=$?
  if [[ "$status" -eq 124 ]]; then
    message="training exceeded ${TRAIN_TIMEOUT}s timeout"
  else
    message="training failed with exit code ${status}"
  fi
  echo "FAIL: ${message}" >&2
  write_failure_result "train" "$status" "$message"
  exit "$status"
fi

if timeout "$EVAL_TIMEOUT" \
  "${run_python[@]}" evaluate.py --split validation \
  >"$EVAL_OUTPUT"; then
  :
else
  status=$?
  if [[ "$status" -eq 124 ]]; then
    message="evaluation exceeded ${EVAL_TIMEOUT}s timeout"
  else
    message="evaluation failed with exit code ${status}"
  fi
  echo "FAIL: ${message}" >&2
  write_failure_result "eval" "$status" "$message"
  exit "$status"
fi

TOTAL_SECONDS="$(( $(date +%s) - STARTED_AT ))"
if TRAIN_SECONDS="$(extract_float train_seconds "$TRAIN_OUTPUT")" \
  && PEAK_VRAM_MB="$(extract_float peak_vram_mb "$TRAIN_OUTPUT")" \
  && NUM_STEPS="$(extract_float num_steps "$TRAIN_OUTPUT")" \
  && VAL_SCORE="$(extract_float val_score "$EVAL_OUTPUT")"; then
  :
else
  status=$?
  echo "FAIL: could not parse training/evaluation summary" >&2
  write_failure_result "summary" "$status" "could not parse training/evaluation summary"
  exit "$status"
fi

echo "---"
printf 'val_score:      %.6f\n' "$VAL_SCORE"
printf 'train_seconds:  %.2f\n' "$TRAIN_SECONDS"
printf 'peak_vram_mb:   %.1f\n' "$PEAK_VRAM_MB"
printf 'num_steps:      %d\n' "$NUM_STEPS"
printf 'total_seconds:  %d\n' "$TOTAL_SECONDS"
printf 'gpu_count:      %d\n' "$GPUS_PER_NODE"

SUMMARY_JSON="$(VAL_SCORE="$VAL_SCORE" TRAIN_SECONDS="$TRAIN_SECONDS" PEAK_VRAM_MB="$PEAK_VRAM_MB" NUM_STEPS="$NUM_STEPS" TOTAL_SECONDS="$TOTAL_SECONDS" GPU_COUNT="$GPUS_PER_NODE" "${run_python[@]}" - <<'PY'
import json
import os

summary = {
    "val_score": float(os.environ["VAL_SCORE"]),
    "train_seconds": float(os.environ["TRAIN_SECONDS"]),
    "peak_vram_mb": float(os.environ["PEAK_VRAM_MB"]),
    "num_steps": int(float(os.environ["NUM_STEPS"])),
    "total_seconds": float(os.environ["TOTAL_SECONDS"]),
    "gpu_count": int(float(os.environ["GPU_COUNT"])),
    "status": "ok",
}
print(json.dumps(summary))
PY
)"
echo "RESULT_JSON:${SUMMARY_JSON}"
printf '%s\n' "$SUMMARY_JSON" > "$LAST_RESULT_PATH"
