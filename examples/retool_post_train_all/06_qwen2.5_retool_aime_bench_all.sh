#!/bin/bash

# Run eval-only AIME benchmarks for the four checkpoints in this recipe:
# base 4B, base 8B, 4B ReTool SFT, and 8B ReTool SFT.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

BENCH_MODELS=${BENCH_MODELS:-"qwen3-4b qwen3-8b qwen3-4b-sft qwen3-8b-sft"}
IFS=' ' read -r -a MODEL_LIST <<< "${BENCH_MODELS}"

for model in "${MODEL_LIST[@]}"; do
   echo "=== Running AIME bench for ${model} ==="
   BENCH_MODEL="${model}" bash "${SCRIPT_DIR}/05_qwen3_retool_aime_bench.sh"
done
