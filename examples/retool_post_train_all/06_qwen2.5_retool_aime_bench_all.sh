#!/bin/bash

# Run eval-only AIME benchmarks for the four checkpoints in this recipe:
# base 3B, base 7B, 3B ReTool SFT, and 7B ReTool SFT.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

BENCH_MODELS=${BENCH_MODELS:-"qwen2.5-3b qwen2.5-3b-sft qwen2.5-7b qwen2.5-7b-sft"}
IFS=' ' read -r -a MODEL_LIST <<< "${BENCH_MODELS}"

for model in "${MODEL_LIST[@]}"; do
   echo "=== Running AIME bench for ${model} ==="
   BENCH_MODEL="${model}" bash "${SCRIPT_DIR}/05_qwen2.5_retool_aime_bench.sh"
done
