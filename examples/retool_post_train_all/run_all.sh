#!/bin/bash

# Run the complete post-training flow:
# 1. Qwen2.5-7B-Instruct ReTool SFT
# 2. Export Qwen2.5-7B-Instruct SFT to HF, then convert HF to a clean torch-dist checkpoint
# 3. Qwen2.5-7B-Instruct ReTool RL
# 4. Qwen2.5-3B-Instruct ReTool SFT
# 5. Qwen2.5-3B-Instruct OPD from the 7B ReTool RL teacher

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_PREPARE=${RUN_PREPARE:-0}
RUN_SFT=${RUN_SFT:-1}
RUN_7B_SFT_CLEAN_DIST=${RUN_7B_SFT_CLEAN_DIST:-1}
RUN_RL=${RUN_RL:-1}
RUN_3B_SFT=${RUN_3B_SFT:-1}
RUN_OPD=${RUN_OPD:-1}

if [ "${RUN_PREPARE}" = "1" ]; then
   bash "${SCRIPT_DIR}/00_prepare_qwen2.5_checkpoints.sh"
fi

if [ "${RUN_SFT}" = "1" ]; then
   bash "${SCRIPT_DIR}/01_qwen2.5_7b_retool_sft.sh"
fi

if [ "${RUN_7B_SFT_CLEAN_DIST}" = "1" ]; then
   bash "${SCRIPT_DIR}/02_prepare_qwen2.5_7b_sft_clean_dist.sh"
fi

if [ "${RUN_RL}" = "1" ]; then
   bash "${SCRIPT_DIR}/02_qwen2.5_7b_retool_rl.sh"
fi

if [ "${RUN_3B_SFT}" = "1" ]; then
   bash "${SCRIPT_DIR}/03_qwen2.5_3b_retool_sft.sh"
fi

if [ "${RUN_OPD}" = "1" ]; then
   bash "${SCRIPT_DIR}/04_qwen2.5_3b_opd_from_qwen2.5_7b_retool.sh"
fi
