#!/bin/bash

# Qwen3-8B ReTool RL, initialized from the clean SFT torch-dist checkpoint.

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

has_megatron_checkpoint() {
   local ckpt_root="$1"
   [ -s "${ckpt_root}/latest_checkpointed_iteration.txt" ]
}

RL_INPUT_KEY=${RL_INPUT_KEY:-prompt}
RL_LABEL_KEY=${RL_LABEL_KEY:-label}
RL_NUM_EPOCH=${RL_NUM_EPOCH:-1}
if [ -n "${RL_NUM_ROLLOUT:-}" ]; then
   echo "WARN: ignoring RL_NUM_ROLLOUT=${RL_NUM_ROLLOUT}; this recipe uses RL_NUM_EPOCH=${RL_NUM_EPOCH}."
fi
RL_ROLLOUT_BATCH_SIZE=${RL_ROLLOUT_BATCH_SIZE:-16}
RL_N_SAMPLES_PER_PROMPT=${RL_N_SAMPLES_PER_PROMPT:-8}
RL_ROLLOUT_MAX_RESPONSE_LEN=${RL_ROLLOUT_MAX_RESPONSE_LEN:-24576}
RL_GLOBAL_BATCH_SIZE=${RL_GLOBAL_BATCH_SIZE:-128}
RL_SAVE_INTERVAL=${RL_SAVE_INTERVAL:-30}
RL_MAX_TOKENS_PER_GPU=${RL_MAX_TOKENS_PER_GPU:-16384}
RL_LOG_PROBS_CHUNK_SIZE=${RL_LOG_PROBS_CHUNK_SIZE:-1024}
RL_START_ROLLOUT_ID=${RL_START_ROLLOUT_ID:-}
RL_CONTEXT_PARALLEL_SIZE=${RL_CONTEXT_PARALLEL_SIZE:-2}
RL_ROLLOUT_MAX_CONTEXT_LEN=${RL_ROLLOUT_MAX_CONTEXT_LEN:-32768}
EVAL_INTERVAL=${EVAL_INTERVAL:-30}
EVAL_MAX_CONTEXT_LEN=${EVAL_MAX_CONTEXT_LEN:-40960}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-32768}
EVAL_N_SAMPLES_PER_PROMPT=${EVAL_N_SAMPLES_PER_PROMPT:-8}
EVAL_INCLUDE_AIME2025=${EVAL_INCLUDE_AIME2025:-0}
LOG_PASSRATE=${LOG_PASSRATE:-1}
AUTO_RESUME=${AUTO_RESUME:-1}

RL_ACTOR_NUM_NODES=${RL_ACTOR_NUM_NODES:-1}
RL_ACTOR_GPUS_PER_NODE=${RL_ACTOR_GPUS_PER_NODE:-8}
RL_RAY_NUM_GPUS=${RL_RAY_NUM_GPUS:-8}

export MODEL_ARGS_ROTARY_BASE="${QWEN3_8B_ROTARY_BASE}"
source "${SLIME_ROOT}/scripts/models/qwen3-8B.sh"

QWEN3_8B_RL_EFFECTIVE_LOAD="${QWEN3_8B_RL_LOAD}"
QWEN3_8B_RL_RESUME_FROM_SAVE=0
if [ "${AUTO_RESUME}" = "1" ] && has_megatron_checkpoint "${QWEN3_8B_RL_SAVE}"; then
   QWEN3_8B_RL_EFFECTIVE_LOAD="${QWEN3_8B_RL_SAVE}"
   QWEN3_8B_RL_RESUME_FROM_SAVE=1
fi
echo "rl initial load: ${QWEN3_8B_RL_LOAD}"
echo "rl effective load: ${QWEN3_8B_RL_EFFECTIVE_LOAD}"
echo "rl resume from output save: ${QWEN3_8B_RL_RESUME_FROM_SAVE}"

CKPT_ARGS=(
   --hf-checkpoint "${QWEN3_8B_HF}"
   --ref-load "${QWEN3_8B_RL_REF_LOAD}"
   --load "${QWEN3_8B_RL_EFFECTIVE_LOAD}"
   --save "${QWEN3_8B_RL_SAVE}"
   --save-interval "${RL_SAVE_INTERVAL}"
   --save-hf "${QWEN3_8B_RL_SAVE_HF}"
   --rotary-base "${QWEN3_8B_ROTARY_BASE}"
)
if [ -n "${RL_START_ROLLOUT_ID}" ]; then
   CKPT_ARGS+=(--start-rollout-id "${RL_START_ROLLOUT_ID}")
elif [ "${QWEN3_8B_RL_RESUME_FROM_SAVE}" != "1" ]; then
   CKPT_ARGS+=(--start-rollout-id 0)
fi

ROLLOUT_ARGS=(
   --prompt-data "${RL_PROMPT_DATA}"
   --input-key "${RL_INPUT_KEY}"
)

if [ -n "${RL_LABEL_KEY}" ]; then
   ROLLOUT_ARGS+=(--label-key "${RL_LABEL_KEY}")
fi

ROLLOUT_ARGS+=(
   --rollout-shuffle
   --reward-key score
   --num-epoch "${RL_NUM_EPOCH}"
   --rollout-batch-size "${RL_ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${RL_N_SAMPLES_PER_PROMPT}"
   --rollout-max-context-len "${RL_ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${RL_ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature "${RL_ROLLOUT_TEMPERATURE:-0.6}"
   --rollout-top-p "${RL_ROLLOUT_TOP_P:-0.95}"
   --rollout-top-k "${RL_ROLLOUT_TOP_K:-20}"

   --global-batch-size "${RL_GLOBAL_BATCH_SIZE}"
   --balance-data
)

EVAL_ARGS=()
if [ "${USE_EVAL:-1}" = "1" ]; then
   EVAL_PROMPT_DATA_ARGS=(
      "${EVAL_DATASET_NAME_2024:-aime2024}" "${EVAL_PROMPT_DATA_2024}"
   )
   if [ "${EVAL_INCLUDE_AIME2025}" = "1" ]; then
      EVAL_PROMPT_DATA_ARGS+=(
         "${EVAL_DATASET_NAME_2025:-aime2025}" "${EVAL_PROMPT_DATA_2025}"
      )
   fi
   EVAL_ARGS=(
      --eval-interval "${EVAL_INTERVAL}"
      --eval-prompt-data
      "${EVAL_PROMPT_DATA_ARGS[@]}"
      --n-samples-per-eval-prompt "${EVAL_N_SAMPLES_PER_PROMPT}"
      --eval-max-context-len "${EVAL_MAX_CONTEXT_LEN}"
      --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
      --eval-temperature "${EVAL_TEMPERATURE:-0.6}"
      --eval-top-p "${EVAL_TOP_P:-0.95}"
      --eval-top-k "${EVAL_TOP_K:-20}"
   )
fi

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${RL_CONTEXT_PARALLEL_SIZE}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${RL_MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size "${RL_LOG_PROBS_CHUNK_SIZE}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef "${RL_KL_LOSS_COEF:-0.00}"
   --kl-loss-type low_var_kl
   --entropy-coef "${RL_ENTROPY_COEF:-0.00}"
   --eps-clip "${RL_EPS_CLIP:-0.2}"
   --eps-clip-high "${RL_EPS_CLIP_HIGH:-0.28}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${RL_LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay "${RL_WEIGHT_DECAY:-0.1}"
   --adam-beta1 "${RL_ADAM_BETA1:-0.9}"
   --adam-beta2 "${RL_ADAM_BETA2:-0.98}"
)

WANDB_ARGS=()
if [ "${USE_WANDB:-0}" = "1" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT:-slime-dapo}"
      --wandb-group "${WANDB_GROUP:-qwen3-8B-retool-rl}"
   )
   if [ -n "${WANDB_KEY:-}" ]; then
      WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
   fi
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${RL_ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
   --sglang-mem-fraction-static "${RL_SGLANG_MEM_FRACTION_STATIC:-0.4}"
)

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
if [ "${LOG_PASSRATE}" = "1" ]; then
   LOG_ARGS+=(--log-passrate)
fi

cd "${SLIME_ROOT}"

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
unset PYTORCH_CUDA_ALLOC_CONF
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "${RL_RAY_NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

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

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- bash "${SCRIPT_DIR}/ray_entrypoint.sh" python3 train.py \
   --actor-num-nodes "${RL_ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${RL_ACTOR_GPUS_PER_NODE}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${LOG_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
