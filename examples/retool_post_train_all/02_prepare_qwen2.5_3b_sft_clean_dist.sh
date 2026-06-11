#!/bin/bash

# Export the Qwen2.5-3B-Instruct ReTool SFT checkpoint to HF, then convert that HF
# checkpoint back to a release torch-dist checkpoint for RL/OPD initialization.

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

CONVERT_NPROC_PER_NODE=${CONVERT_NPROC_PER_NODE:-1}
CLEAN_CONVERT_FORCE=${CLEAN_CONVERT_FORCE:-0}

latest_megatron_ckpt_dir() {
   local ckpt_root="$1"
   local tracker="${ckpt_root}/latest_checkpointed_iteration.txt"
   if [ ! -f "${tracker}" ]; then
      echo "Missing checkpoint tracker: ${tracker}" >&2
      exit 1
   fi

   local iteration
   iteration="$(tr -d '[:space:]' < "${tracker}")"
   if [ "${iteration}" = "release" ]; then
      printf "%s/release" "${ckpt_root}"
   else
      printf "%s/iter_%07d" "${ckpt_root}" "${iteration}"
   fi
}

prepare_output_dir() {
   local path="$1"
   if [ -e "${path}" ]; then
      if [ "${CLEAN_CONVERT_FORCE}" = "1" ]; then
         rm -rf "${path}"
      else
         echo "Output already exists: ${path}" >&2
         echo "Set CLEAN_CONVERT_FORCE=1 to overwrite it." >&2
         exit 1
      fi
   fi
   mkdir -p "$(dirname "${path}")"
}

SFT_ITER_DIR="$(latest_megatron_ckpt_dir "${QWEN2_5_3B_SFT_SAVE}")"
if [ ! -d "${SFT_ITER_DIR}" ]; then
   echo "SFT checkpoint directory does not exist: ${SFT_ITER_DIR}" >&2
   exit 1
fi

prepare_output_dir "${QWEN2_5_3B_SFT_HF_EXPORT}"
prepare_output_dir "${QWEN2_5_3B_SFT_CLEAN_DIST}"

cd "${SLIME_ROOT}"

PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}" python3 tools/convert_torch_dist_to_hf.py \
   --input-dir "${SFT_ITER_DIR}" \
   --output-dir "${QWEN2_5_3B_SFT_HF_EXPORT}" \
   --origin-hf-dir "${QWEN2_5_3B_HF}" \
   --vocab-size 151936

export MODEL_ARGS_ROTARY_BASE="${QWEN2_5_3B_ROTARY_BASE}"
source "${SLIME_ROOT}/scripts/models/qwen2.5-3B.sh"

PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}" torchrun \
   --nproc_per_node="${CONVERT_NPROC_PER_NODE}" \
   tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint "${QWEN2_5_3B_SFT_HF_EXPORT}" \
   --rotary-base "${QWEN2_5_3B_ROTARY_BASE}" \
   --save "${QWEN2_5_3B_SFT_CLEAN_DIST}"
