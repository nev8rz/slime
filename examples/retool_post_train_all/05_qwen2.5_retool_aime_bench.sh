#!/bin/bash

# Eval-only AIME benchmark for Qwen3 ReTool checkpoints.
#
# Select a model with:
#   BENCH_MODEL=qwen3-4b|qwen3-8b|qwen3-4b-sft|qwen3-8b-sft
#
# The script intentionally passes --num-rollout 0 and no --prompt-data, so
# slime initializes the actor/rollout stack, syncs weights, and runs eval only.

set -Eeuo pipefail
IFS=$'\n\t'

if [ "${TRACE:-0}" = "1" ]; then
   set -x
fi

export PYTHONUNBUFFERED=1

stop_existing_runtime() {
   pkill -9 sglang 2>/dev/null || true
   ray stop --force 2>/dev/null || true
   pkill -9 ray 2>/dev/null || true
   pkill -9 python 2>/dev/null || true
}

if [ "${BENCH_STOP_EXISTING:-1}" = "1" ]; then
   stop_existing_runtime
   sleep 3
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

first_existing_path() {
   local fallback="$1"
   shift
   for path in "$@"; do
      if [ -n "${path}" ] && [ -e "${path}" ]; then
         printf "%s" "${path}"
         return
      fi
   done
   printf "%s" "${fallback}"
}

normalize_model_name() {
   printf "%s" "$1" | tr '[:upper:]' '[:lower:]' | tr '_' '-'
}

BENCH_MODEL="$(normalize_model_name "${BENCH_MODEL:-qwen3-4b}")"
BENCH_EVAL_INCLUDE_AIME2025=${BENCH_EVAL_INCLUDE_AIME2025:-1}
BENCH_EVAL_N_SAMPLES_PER_PROMPT=${BENCH_EVAL_N_SAMPLES_PER_PROMPT:-16}
BENCH_EVAL_MAX_CONTEXT_LEN=${BENCH_EVAL_MAX_CONTEXT_LEN:-40960}
BENCH_EVAL_MAX_RESPONSE_LEN=${BENCH_EVAL_MAX_RESPONSE_LEN:-32768}
BENCH_EVAL_TEMPERATURE=${BENCH_EVAL_TEMPERATURE:-0.6}
BENCH_EVAL_TOP_P=${BENCH_EVAL_TOP_P:-0.95}
BENCH_EVAL_TOP_K=${BENCH_EVAL_TOP_K:-20}
BENCH_ROLLOUT_MAX_CONTEXT_LEN=${BENCH_ROLLOUT_MAX_CONTEXT_LEN:-32768}
BENCH_ROLLOUT_MAX_RESPONSE_LEN=${BENCH_ROLLOUT_MAX_RESPONSE_LEN:-24576}
BENCH_MAX_TOKENS_PER_GPU=${BENCH_MAX_TOKENS_PER_GPU:-16384}
BENCH_LOG_PROBS_CHUNK_SIZE=${BENCH_LOG_PROBS_CHUNK_SIZE:-1024}
BENCH_CONTEXT_PARALLEL_SIZE=${BENCH_CONTEXT_PARALLEL_SIZE:-1}
BENCH_ACTOR_NUM_NODES=${BENCH_ACTOR_NUM_NODES:-1}
BENCH_ACTOR_GPUS_PER_NODE=${BENCH_ACTOR_GPUS_PER_NODE:-8}
BENCH_RAY_NUM_GPUS=${BENCH_RAY_NUM_GPUS:-${BENCH_ACTOR_GPUS_PER_NODE}}
BENCH_ROLLOUT_BATCH_SIZE=${BENCH_ROLLOUT_BATCH_SIZE:-1}
BENCH_N_SAMPLES_PER_PROMPT=${BENCH_N_SAMPLES_PER_PROMPT:-1}
BENCH_GLOBAL_BATCH_SIZE=${BENCH_GLOBAL_BATCH_SIZE:-${BENCH_ACTOR_GPUS_PER_NODE}}
BENCH_USE_WANDB=${BENCH_USE_WANDB:-${USE_WANDB:-1}}
BENCH_WANDB_PROJECT=${BENCH_WANDB_PROJECT:-retool-bench}
BENCH_WANDB_MODE=${BENCH_WANDB_MODE:-${WANDB_MODE:-online}}
BENCH_LOG_PASSRATE=${BENCH_LOG_PASSRATE:-1}
BENCH_DISABLE_WANDB_RANDOM_SUFFIX=${BENCH_DISABLE_WANDB_RANDOM_SUFFIX:-1}
BENCH_CLEANUP_ON_EXIT=${BENCH_CLEANUP_ON_EXIT:-0}
BENCH_SGLANG_ATTENTION_BACKEND=${BENCH_SGLANG_ATTENTION_BACKEND:-triton}
BENCH_SGLANG_SAMPLING_BACKEND=${BENCH_SGLANG_SAMPLING_BACKEND:-pytorch}
BENCH_SGLANG_DISABLE_CUDA_GRAPH=${BENCH_SGLANG_DISABLE_CUDA_GRAPH:-1}

if [ "${BENCH_CLEANUP_ON_EXIT}" = "1" ]; then
   trap stop_existing_runtime EXIT
fi

case "${BENCH_MODEL}" in
   qwen3-4b | 4b)
      BENCH_MODEL_TAG=qwen3-4B
      BENCH_HF_CHECKPOINT=${BENCH_HF_CHECKPOINT:-${QWEN3_4B_HF}}
      BENCH_LOAD=${BENCH_LOAD:-${QWEN3_4B_REF_LOAD}}
      BENCH_REF_LOAD=${BENCH_REF_LOAD:-${QWEN3_4B_REF_LOAD}}
      BENCH_ROTARY_BASE=${BENCH_ROTARY_BASE:-${QWEN3_4B_ROTARY_BASE}}
      BENCH_TENSOR_MODEL_PARALLEL_SIZE=${BENCH_TENSOR_MODEL_PARALLEL_SIZE:-1}
      BENCH_ROLLOUT_NUM_GPUS_PER_ENGINE=${BENCH_ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
      BENCH_SGLANG_MEM_FRACTION_STATIC=${BENCH_SGLANG_MEM_FRACTION_STATIC:-0.55}
      export MODEL_ARGS_ROTARY_BASE="${BENCH_ROTARY_BASE}"
      source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"
      ;;
   qwen3-8b | 8b)
      BENCH_MODEL_TAG=qwen3-8B
      BENCH_HF_CHECKPOINT=${BENCH_HF_CHECKPOINT:-${QWEN3_8B_HF}}
      BENCH_LOAD=${BENCH_LOAD:-${QWEN3_8B_REF_LOAD}}
      BENCH_REF_LOAD=${BENCH_REF_LOAD:-${QWEN3_8B_REF_LOAD}}
      BENCH_ROTARY_BASE=${BENCH_ROTARY_BASE:-${QWEN3_8B_ROTARY_BASE}}
      BENCH_TENSOR_MODEL_PARALLEL_SIZE=${BENCH_TENSOR_MODEL_PARALLEL_SIZE:-2}
      BENCH_ROLLOUT_NUM_GPUS_PER_ENGINE=${BENCH_ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
      BENCH_SGLANG_MEM_FRACTION_STATIC=${BENCH_SGLANG_MEM_FRACTION_STATIC:-0.4}
      export MODEL_ARGS_ROTARY_BASE="${BENCH_ROTARY_BASE}"
      source "${SLIME_ROOT}/scripts/models/qwen3-8B.sh"
      ;;
   qwen3-4b-sft | 4b-sft)
      BENCH_MODEL_TAG=qwen3-4B-retool-sft
      BENCH_HF_CHECKPOINT=${BENCH_HF_CHECKPOINT:-${QWEN3_4B_HF}}
      DEFAULT_LOAD="$(first_existing_path "${QWEN3_4B_SFT_SAVE}" "${QWEN3_4B_SFT_CLEAN_DIST}" "${QWEN3_4B_SFT_SAVE}")"
      BENCH_LOAD=${BENCH_LOAD:-${DEFAULT_LOAD}}
      BENCH_REF_LOAD=${BENCH_REF_LOAD:-${BENCH_LOAD}}
      BENCH_ROTARY_BASE=${BENCH_ROTARY_BASE:-${QWEN3_4B_ROTARY_BASE}}
      BENCH_TENSOR_MODEL_PARALLEL_SIZE=${BENCH_TENSOR_MODEL_PARALLEL_SIZE:-1}
      BENCH_ROLLOUT_NUM_GPUS_PER_ENGINE=${BENCH_ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
      BENCH_SGLANG_MEM_FRACTION_STATIC=${BENCH_SGLANG_MEM_FRACTION_STATIC:-0.55}
      export MODEL_ARGS_ROTARY_BASE="${BENCH_ROTARY_BASE}"
      source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"
      ;;
   qwen3-8b-sft | 8b-sft)
      BENCH_MODEL_TAG=qwen3-8B-retool-sft
      BENCH_HF_CHECKPOINT=${BENCH_HF_CHECKPOINT:-${QWEN3_8B_HF}}
      DEFAULT_LOAD="$(first_existing_path "${QWEN3_8B_SFT_CLEAN_DIST}" "${QWEN3_8B_SFT_CLEAN_DIST}" "${QWEN3_8B_SFT_SAVE}")"
      BENCH_LOAD=${BENCH_LOAD:-${DEFAULT_LOAD}}
      BENCH_REF_LOAD=${BENCH_REF_LOAD:-${BENCH_LOAD}}
      BENCH_ROTARY_BASE=${BENCH_ROTARY_BASE:-${QWEN3_8B_ROTARY_BASE}}
      BENCH_TENSOR_MODEL_PARALLEL_SIZE=${BENCH_TENSOR_MODEL_PARALLEL_SIZE:-2}
      BENCH_ROLLOUT_NUM_GPUS_PER_ENGINE=${BENCH_ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
      BENCH_SGLANG_MEM_FRACTION_STATIC=${BENCH_SGLANG_MEM_FRACTION_STATIC:-0.4}
      export MODEL_ARGS_ROTARY_BASE="${BENCH_ROTARY_BASE}"
      source "${SLIME_ROOT}/scripts/models/qwen3-8B.sh"
      ;;
   *)
      echo "Unknown BENCH_MODEL=${BENCH_MODEL}" >&2
      echo "Expected qwen3-4b, qwen3-8b, qwen3-4b-sft, or qwen3-8b-sft." >&2
      exit 1
      ;;
esac

if [ ! -e "${BENCH_HF_CHECKPOINT}" ]; then
   echo "BENCH_HF_CHECKPOINT does not exist: ${BENCH_HF_CHECKPOINT}" >&2
   exit 1
fi
if [ ! -e "${BENCH_LOAD}" ]; then
   echo "BENCH_LOAD does not exist: ${BENCH_LOAD}" >&2
   exit 1
fi

EVAL_PROMPT_DATA_ARGS=(
   "${EVAL_DATASET_NAME_2024:-aime2024}" "${EVAL_PROMPT_DATA_2024}"
)
if [ "${BENCH_EVAL_INCLUDE_AIME2025}" = "1" ]; then
   EVAL_PROMPT_DATA_ARGS+=(
      "${EVAL_DATASET_NAME_2025:-aime2025}" "${EVAL_PROMPT_DATA_2025}"
   )
fi

CKPT_ARGS=(
   --hf-checkpoint "${BENCH_HF_CHECKPOINT}"
   --ref-load "${BENCH_REF_LOAD}"
   --load "${BENCH_LOAD}"
   --rotary-base "${BENCH_ROTARY_BASE}"
   --start-rollout-id 0
   --no-load-optim
   --no-load-rng
   --finetune
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --input-key "${BENCH_INPUT_KEY:-prompt}"
   --label-key "${BENCH_LABEL_KEY:-label}"
   --num-rollout 0
   --rollout-batch-size "${BENCH_ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${BENCH_N_SAMPLES_PER_PROMPT}"
   --rollout-max-context-len "${BENCH_ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${BENCH_ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature "${BENCH_ROLLOUT_TEMPERATURE:-0.6}"
   --rollout-top-p "${BENCH_ROLLOUT_TOP_P:-0.95}"
   --rollout-top-k "${BENCH_ROLLOUT_TOP_K:-20}"
   --global-batch-size "${BENCH_GLOBAL_BATCH_SIZE}"
)

EVAL_ARGS=(
   --eval-interval "${BENCH_EVAL_INTERVAL:-1}"
   --eval-prompt-data
   "${EVAL_PROMPT_DATA_ARGS[@]}"
   --eval-input-key "${BENCH_EVAL_INPUT_KEY:-prompt}"
   --eval-label-key "${BENCH_EVAL_LABEL_KEY:-label}"
   --n-samples-per-eval-prompt "${BENCH_EVAL_N_SAMPLES_PER_PROMPT}"
   --eval-max-context-len "${BENCH_EVAL_MAX_CONTEXT_LEN}"
   --eval-max-response-len "${BENCH_EVAL_MAX_RESPONSE_LEN}"
   --eval-temperature "${BENCH_EVAL_TEMPERATURE}"
   --eval-top-p "${BENCH_EVAL_TOP_P}"
   --eval-top-k "${BENCH_EVAL_TOP_K}"
)

PERF_ARGS=(
   --tensor-model-parallel-size "${BENCH_TENSOR_MODEL_PARALLEL_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${BENCH_CONTEXT_PARALLEL_SIZE}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${BENCH_MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size "${BENCH_LOG_PROBS_CHUNK_SIZE}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${BENCH_LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay "${BENCH_WEIGHT_DECAY:-0.1}"
   --adam-beta1 "${BENCH_ADAM_BETA1:-0.9}"
   --adam-beta2 "${BENCH_ADAM_BETA2:-0.98}"
)

WANDB_ARGS=()
if [ "${BENCH_USE_WANDB}" = "1" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "${BENCH_WANDB_PROJECT}"
      --wandb-group "${BENCH_WANDB_GROUP:-${BENCH_MODEL_TAG}-aime-pass16}"
      --wandb-mode "${BENCH_WANDB_MODE}"
   )
   if [ "${BENCH_DISABLE_WANDB_RANDOM_SUFFIX}" = "1" ]; then
      WANDB_ARGS+=(--disable-wandb-random-suffix)
   fi
   if [ -n "${BENCH_WANDB_DIR:-${WANDB_DIR:-}}" ]; then
      WANDB_ARGS+=(--wandb-dir "${BENCH_WANDB_DIR:-${WANDB_DIR:-}}")
   fi
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${BENCH_ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${BENCH_SGLANG_MEM_FRACTION_STATIC}"
)

if [ -n "${BENCH_SGLANG_ATTENTION_BACKEND}" ]; then
   SGLANG_ARGS+=(--sglang-attention-backend "${BENCH_SGLANG_ATTENTION_BACKEND}")
fi
if [ -n "${BENCH_SGLANG_SAMPLING_BACKEND}" ]; then
   SGLANG_ARGS+=(--sglang-sampling-backend "${BENCH_SGLANG_SAMPLING_BACKEND}")
fi
if [ -n "${BENCH_SGLANG_CUDA_GRAPH_MAX_BS:-}" ]; then
   SGLANG_ARGS+=(--sglang-cuda-graph-max-bs "${BENCH_SGLANG_CUDA_GRAPH_MAX_BS}")
fi
if [ "${BENCH_SGLANG_DISABLE_CUDA_GRAPH}" = "1" ]; then
   SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi
if [ "${BENCH_SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-0}" = "1" ]; then
   SGLANG_ARGS+=(--sglang-disable-custom-all-reduce)
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

LOG_ARGS=()
if [ "${BENCH_LOG_PASSRATE}" = "1" ]; then
   LOG_ARGS+=(--log-passrate)
fi

EXTRA_ARGS=()
if [ -n "${BENCH_EXTRA_ARGS:-}" ]; then
   IFS=' ' read -r -a EXTRA_ARGS <<< "${BENCH_EXTRA_ARGS}"
fi

cd "${SLIME_ROOT}"

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR},${no_proxy:-}"
unset PYTORCH_CUDA_ALLOC_CONF

ray start --head \
   --node-ip-address "${MASTER_ADDR}" \
   --num-gpus "${BENCH_RAY_NUM_GPUS}" \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0 \
   --dashboard-port="${BENCH_RAY_DASHBOARD_PORT:-8265}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}:${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"HTTP_PROXY\": \"${HTTP_PROXY:-}\",
    \"HTTPS_PROXY\": \"${HTTPS_PROXY:-}\",
    \"http_proxy\": \"${http_proxy:-}\",
    \"https_proxy\": \"${https_proxy:-}\",
    \"NO_PROXY\": \"${NO_PROXY:-}\",
    \"no_proxy\": \"${no_proxy:-}\",
    \"WANDB_HTTP_PROXY\": \"${WANDB_HTTP_PROXY:-${HTTP_PROXY:-}}\",
    \"WANDB_HTTPS_PROXY\": \"${WANDB_HTTPS_PROXY:-${HTTPS_PROXY:-}}\",
    \"WANDB_NO_PROXY\": \"${WANDB_NO_PROXY:-${NO_PROXY:-}}\",
    \"RETOOL_THINKING_MODE\": \"${RETOOL_THINKING_MODE:-think}\"
  }
}"

echo "Bench model: ${BENCH_MODEL_TAG}"
echo "HF checkpoint: ${BENCH_HF_CHECKPOINT}"
echo "Load checkpoint: ${BENCH_LOAD}"
echo "Eval datasets: ${EVAL_PROMPT_DATA_ARGS[*]}"
echo "W&B project: ${BENCH_WANDB_PROJECT}"

ray job submit --address="http://127.0.0.1:${BENCH_RAY_DASHBOARD_PORT:-8265}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- bash "${SCRIPT_DIR}/ray_entrypoint.sh" python3 train.py \
   --actor-num-nodes "${BENCH_ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${BENCH_ACTOR_GPUS_PER_NODE}" \
   --num-gpus-per-node "${BENCH_ACTOR_GPUS_PER_NODE}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${LOG_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}" \
   "${EXTRA_ARGS[@]}"
