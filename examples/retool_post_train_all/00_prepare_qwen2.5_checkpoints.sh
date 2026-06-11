#!/bin/bash

# Prepare Qwen2.5 base checkpoints and ReTool data for:
# Qwen2.5-7B-Instruct ReTool SFT -> Qwen2.5-7B-Instruct ReTool RL -> Qwen2.5-3B-Instruct ReTool SFT -> Qwen2.5-3B-Instruct OPD.

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

CONVERT_QWEN2_5_7B=${CONVERT_QWEN2_5_7B:-1}
CONVERT_QWEN2_5_3B=${CONVERT_QWEN2_5_3B:-1}
PREPARE_RETOOL_SFT_DATA=${PREPARE_RETOOL_SFT_DATA:-1}
PREPARE_RETOOL_RL_DATA=${PREPARE_RETOOL_RL_DATA:-0}

cd "${SLIME_ROOT}"

if [ "${PREPARE_RETOOL_SFT_DATA}" = "1" ]; then
   python3 "${SCRIPT_DIR}/sft_data_processing.py"
fi

if [ "${PREPARE_RETOOL_RL_DATA}" = "1" ]; then
   python3 "${SCRIPT_DIR}/rl_data_preprocess.py"
fi

if [ "${CONVERT_QWEN2_5_7B}" = "1" ]; then
   export MODEL_ARGS_ROTARY_BASE="${QWEN2_5_7B_ROTARY_BASE}"
   source "${SLIME_ROOT}/scripts/models/qwen2.5-7B.sh"
   mkdir -p "$(dirname "${QWEN2_5_7B_TORCH_DIST}")"
   PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}" python3 tools/convert_hf_to_torch_dist.py \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "${QWEN2_5_7B_HF}" \
      --rotary-base "${QWEN2_5_7B_ROTARY_BASE}" \
      --save "${QWEN2_5_7B_TORCH_DIST}"
fi

if [ "${CONVERT_QWEN2_5_3B}" = "1" ]; then
   export MODEL_ARGS_ROTARY_BASE="${QWEN2_5_3B_ROTARY_BASE}"
   source "${SLIME_ROOT}/scripts/models/qwen2.5-3B.sh"
   mkdir -p "$(dirname "${QWEN2_5_3B_TORCH_DIST}")"
   PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}" python3 tools/convert_hf_to_torch_dist.py \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "${QWEN2_5_3B_HF}" \
      --rotary-base "${QWEN2_5_3B_ROTARY_BASE}" \
      --save "${QWEN2_5_3B_TORCH_DIST}"
fi
