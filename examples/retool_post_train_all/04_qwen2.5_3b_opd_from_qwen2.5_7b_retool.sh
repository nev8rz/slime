#!/bin/bash

# Distill the Qwen2.5-7B-Instruct ReTool RL teacher into the clean Qwen2.5-3B-Instruct ReTool SFT student with OPD.

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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

has_megatron_checkpoint() {
   local ckpt_root="$1"
   [ -s "${ckpt_root}/latest_checkpointed_iteration.txt" ]
}

OPD_INPUT_KEY=${OPD_INPUT_KEY:-prompt}
OPD_LABEL_KEY=${OPD_LABEL_KEY:-label}
OPD_NUM_EPOCH=${OPD_NUM_EPOCH:-1}
if [ -n "${OPD_NUM_ROLLOUT:-}" ]; then
   echo "WARN: ignoring OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT}; this recipe uses OPD_NUM_EPOCH=${OPD_NUM_EPOCH}."
fi
OPD_ROLLOUT_BATCH_SIZE=${OPD_ROLLOUT_BATCH_SIZE:-16}
OPD_N_SAMPLES_PER_PROMPT=${OPD_N_SAMPLES_PER_PROMPT:-4}
OPD_ROLLOUT_MAX_RESPONSE_LEN=${OPD_ROLLOUT_MAX_RESPONSE_LEN:-24576}
OPD_GLOBAL_BATCH_SIZE=${OPD_GLOBAL_BATCH_SIZE:-64}
OPD_SAVE_INTERVAL=${OPD_SAVE_INTERVAL:-30}
OPD_MAX_TOKENS_PER_GPU=${OPD_MAX_TOKENS_PER_GPU:-16384}
OPD_LOG_PROBS_CHUNK_SIZE=${OPD_LOG_PROBS_CHUNK_SIZE:-1024}
OPD_START_ROLLOUT_ID=${OPD_START_ROLLOUT_ID:-}
OPD_CONTEXT_PARALLEL_SIZE=${OPD_CONTEXT_PARALLEL_SIZE:-2}
OPD_ROLLOUT_MAX_CONTEXT_LEN=${OPD_ROLLOUT_MAX_CONTEXT_LEN:-32768}
EVAL_INTERVAL=${EVAL_INTERVAL:-30}
EVAL_MAX_CONTEXT_LEN=${EVAL_MAX_CONTEXT_LEN:-40960}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-32768}
EVAL_N_SAMPLES_PER_PROMPT=${EVAL_N_SAMPLES_PER_PROMPT:-8}
EVAL_INCLUDE_AIME2025=${EVAL_INCLUDE_AIME2025:-0}
LOG_PASSRATE=${LOG_PASSRATE:-1}
AUTO_RESUME=${AUTO_RESUME:-1}

OPD_ACTOR_NUM_NODES=${OPD_ACTOR_NUM_NODES:-1}
OPD_ACTOR_GPUS_PER_NODE=${OPD_ACTOR_GPUS_PER_NODE:-4}
OPD_ROLLOUT_NUM_GPUS=${OPD_ROLLOUT_NUM_GPUS:-2}
OPD_RAY_NUM_GPUS=${OPD_RAY_NUM_GPUS:-6}

TEACHER_IP=${TEACHER_IP:-127.0.0.1}
TEACHER_HOST=${TEACHER_HOST:-0.0.0.0}
TEACHER_PORT=${TEACHER_PORT:-13141}
TEACHER_CUDA_VISIBLE_DEVICES=${TEACHER_CUDA_VISIBLE_DEVICES:-6,7}
TEACHER_TP=${TEACHER_TP:-2}
TEACHER_MEM_FRACTION_STATIC=${TEACHER_MEM_FRACTION_STATIC:-0.6}
TEACHER_CHUNKED_PREFILL_SIZE=${TEACHER_CHUNKED_PREFILL_SIZE:-4096}
LOG_FILE=${LOG_FILE:-/tmp/sglang_qwen2.5_7b_retool_teacher.log}

if [ ! -d "${QWEN2_5_7B_TEACHER_HF}" ]; then
   echo "QWEN2_5_7B_TEACHER_HF does not exist: ${QWEN2_5_7B_TEACHER_HF}"
   echo "Run 02_qwen2.5_7b_retool_rl.sh first or set QWEN2_5_7B_TEACHER_HF to a concrete HF export."
   exit 1
fi

CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}" python3 -m sglang.launch_server \
    --model-path "${QWEN2_5_7B_TEACHER_HF}" \
    --host "${TEACHER_HOST}" \
    --port "${TEACHER_PORT}" \
    --tp "${TEACHER_TP}" \
    --chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE}" \
    --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}" \
    > "${LOG_FILE}" 2>&1 &

TEACHER_PID=$!

cleanup() {
   kill "${TEACHER_PID}" 2>/dev/null || true
   pkill -9 sglang || true
   ray stop --force || true
   pkill -9 ray || true
}
trap cleanup EXIT

echo "Starting Qwen2.5-7B-Instruct ReTool teacher model server..."
until curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" > /dev/null; do
    echo "Waiting for the teacher model server to start..."
    tail -n 10 "${LOG_FILE}" || true
    sleep 5
done

curl "http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info"
echo "Teacher model server is up and running at ${TEACHER_IP}:${TEACHER_PORT}."
sleep 10

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export MODEL_ARGS_ROTARY_BASE="${QWEN2_5_3B_ROTARY_BASE}"
source "${SLIME_ROOT}/scripts/models/qwen2.5-3B.sh"

QWEN2_5_3B_OPD_EFFECTIVE_LOAD="${QWEN2_5_3B_OPD_LOAD}"
QWEN2_5_3B_OPD_RESUME_FROM_SAVE=0
if [ "${AUTO_RESUME}" = "1" ] && has_megatron_checkpoint "${QWEN2_5_3B_OPD_SAVE}"; then
   QWEN2_5_3B_OPD_EFFECTIVE_LOAD="${QWEN2_5_3B_OPD_SAVE}"
   QWEN2_5_3B_OPD_RESUME_FROM_SAVE=1
fi
echo "opd initial load: ${QWEN2_5_3B_OPD_LOAD}"
echo "opd effective load: ${QWEN2_5_3B_OPD_EFFECTIVE_LOAD}"
echo "opd resume from output save: ${QWEN2_5_3B_OPD_RESUME_FROM_SAVE}"

CKPT_ARGS=(
   --hf-checkpoint "${QWEN2_5_3B_HF}"
   --ref-load "${QWEN2_5_3B_OPD_REF_LOAD}"
   --save "${QWEN2_5_3B_OPD_SAVE}"
   --save-interval "${OPD_SAVE_INTERVAL}"
   --save-hf "${QWEN2_5_3B_OPD_SAVE_HF}"
   --rotary-base "${QWEN2_5_3B_ROTARY_BASE}"
)

if [ -n "${QWEN2_5_3B_OPD_EFFECTIVE_LOAD}" ]; then
   CKPT_ARGS+=(--load "${QWEN2_5_3B_OPD_EFFECTIVE_LOAD}")
fi
if [ -n "${OPD_START_ROLLOUT_ID}" ]; then
   CKPT_ARGS+=(--start-rollout-id "${OPD_START_ROLLOUT_ID}")
elif [ "${QWEN2_5_3B_OPD_RESUME_FROM_SAVE}" != "1" ]; then
   CKPT_ARGS+=(--start-rollout-id 0)
fi

ROLLOUT_ARGS=(
   --prompt-data "${OPD_PROMPT_DATA}"
   --input-key "${OPD_INPUT_KEY}"
)

if [ -n "${OPD_LABEL_KEY}" ]; then
   ROLLOUT_ARGS+=(--label-key "${OPD_LABEL_KEY}")
fi

ROLLOUT_ARGS+=(
   --rollout-shuffle
   --num-epoch "${OPD_NUM_EPOCH}"
   --rollout-batch-size "${OPD_ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${OPD_N_SAMPLES_PER_PROMPT}"
   --rollout-max-context-len "${OPD_ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${OPD_ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature "${OPD_ROLLOUT_TEMPERATURE:-0.6}"
   --rollout-top-p "${OPD_ROLLOUT_TOP_P:-0.95}"
   --rollout-top-k "${OPD_ROLLOUT_TOP_K:-20}"

   --global-batch-size "${OPD_GLOBAL_BATCH_SIZE}"
   --balance-data
)

RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
   --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
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
   --context-parallel-size "${OPD_CONTEXT_PARALLEL_SIZE}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${OPD_MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size "${OPD_LOG_PROBS_CHUNK_SIZE}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef "${OPD_KL_COEF:-1.0}"
   --use-kl-loss
   --kl-loss-coef "${OPD_KL_LOSS_COEF:-0.00}"
   --kl-loss-type low_var_kl
   --entropy-coef "${OPD_ENTROPY_COEF:-0.00}"
   --eps-clip "${OPD_EPS_CLIP:-0.2}"
   --eps-clip-high "${OPD_EPS_CLIP_HIGH:-0.28}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${OPD_LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay "${OPD_WEIGHT_DECAY:-0.1}"
   --adam-beta1 "${OPD_ADAM_BETA1:-0.9}"
   --adam-beta2 "${OPD_ADAM_BETA2:-0.98}"
)

WANDB_ARGS=()
if [ "${USE_WANDB:-0}" = "1" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "${OPD_WANDB_PROJECT}"
      --wandb-group "${WANDB_GROUP:-qwen2.5-3B-opd-from-qwen2.5-7B-retool}"
   )
   if [ -n "${WANDB_KEY:-}" ]; then
      WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
   fi
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${OPD_ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
   --sglang-mem-fraction-static "${OPD_SGLANG_MEM_FRACTION_STATIC:-0.4}"
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
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "${OPD_RAY_NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

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
    \"RETOOL_THINKING_MODE\": \"${RETOOL_THINKING_MODE:-no_think}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- bash "${SCRIPT_DIR}/ray_entrypoint.sh" python3 train.py \
   --actor-num-nodes "${OPD_ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${OPD_ACTOR_GPUS_PER_NODE}" \
   --rollout-num-gpus "${OPD_ROLLOUT_NUM_GPUS}" \
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
   "${CUSTOM_ARGS[@]}" \
   "${RM_ARGS[@]}"
