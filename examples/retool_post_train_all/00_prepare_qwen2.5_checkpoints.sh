#!/bin/bash

# Prepare Qwen3 base checkpoints and ReTool data for:
# Qwen3-8B ReTool SFT -> Qwen3-8B ReTool RL -> Qwen3-4B ReTool SFT -> Qwen3-4B OPD.

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

CONVERT_QWEN3_8B=${CONVERT_QWEN3_8B:-1}
CONVERT_QWEN3_4B=${CONVERT_QWEN3_4B:-1}
PREPARE_RETOOL_SFT_DATA=${PREPARE_RETOOL_SFT_DATA:-1}
PREPARE_RETOOL_RL_DATA=${PREPARE_RETOOL_RL_DATA:-0}

cd "${SLIME_ROOT}"

if [ "${PREPARE_RETOOL_SFT_DATA}" = "1" ]; then
   python3 "${SCRIPT_DIR}/sft_data_processing.py"
fi

if [ "${PREPARE_RETOOL_RL_DATA}" = "1" ]; then
   python3 "${SCRIPT_DIR}/rl_data_preprocess.py"
fi

if [ "${CONVERT_QWEN3_8B}" = "1" ]; then
   export MODEL_ARGS_ROTARY_BASE="${QWEN3_8B_ROTARY_BASE}"
   source "${SLIME_ROOT}/scripts/models/qwen3-8B.sh"
   mkdir -p "$(dirname "${QWEN3_8B_TORCH_DIST}")"
   PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}" python3 tools/convert_hf_to_torch_dist.py \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "${QWEN3_8B_HF}" \
      --rotary-base "${QWEN3_8B_ROTARY_BASE}" \
      --save "${QWEN3_8B_TORCH_DIST}"
fi

if [ "${CONVERT_QWEN3_4B}" = "1" ]; then
   export MODEL_ARGS_ROTARY_BASE="${QWEN3_4B_ROTARY_BASE}"
   source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"
   mkdir -p "$(dirname "${QWEN3_4B_TORCH_DIST}")"
   PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}" python3 tools/convert_hf_to_torch_dist.py \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "${QWEN3_4B_HF}" \
      --rotary-base "${QWEN3_4B_ROTARY_BASE}" \
      --save "${QWEN3_4B_TORCH_DIST}"
fi
