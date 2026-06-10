#!/bin/bash

# Qwen3-4B ReTool SFT before OPD.

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

SFT_NUM_EPOCH=${SFT_NUM_EPOCH:-3}
SFT_ROLLOUT_BATCH_SIZE=${SFT_ROLLOUT_BATCH_SIZE:-128}
SFT_GLOBAL_BATCH_SIZE=${SFT_GLOBAL_BATCH_SIZE:-128}
SFT_SAVE_INTERVAL=${SFT_SAVE_INTERVAL:-1000}
SFT_MAX_TOKENS_PER_GPU=${SFT_MAX_TOKENS_PER_GPU:-9216}
SFT_APPLY_CHAT_TEMPLATE_KWARGS=${SFT_APPLY_CHAT_TEMPLATE_KWARGS:-'{"enable_thinking": false}'}

SFT_ACTOR_NUM_NODES=${SFT_ACTOR_NUM_NODES:-1}
SFT_ACTOR_GPUS_PER_NODE=${SFT_ACTOR_GPUS_PER_NODE:-8}
SFT_RAY_NUM_GPUS=${SFT_RAY_NUM_GPUS:-8}

export MODEL_ARGS_ROTARY_BASE="${QWEN3_4B_ROTARY_BASE}"
source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${QWEN3_4B_HF}"
   --ref-load "${QWEN3_4B_REF_LOAD}"
   --save "${QWEN3_4B_SFT_SAVE}"
   --save-interval "${SFT_SAVE_INTERVAL}"
   --rotary-base "${QWEN3_4B_ROTARY_BASE}"
)

if [ -n "${QWEN3_4B_SFT_LOAD}" ]; then
   CKPT_ARGS+=(--load "${QWEN3_4B_SFT_LOAD}")
fi

if [ -n "${QWEN3_4B_SFT_SAVE_HF}" ]; then
   CKPT_ARGS+=(--save-hf "${QWEN3_4B_SFT_SAVE_HF}")
fi

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${SFT_PROMPT_DATA}"
   --apply-chat-template-kwargs "${SFT_APPLY_CHAT_TEMPLATE_KWARGS}"
   --input-key messages
   --tool-key tools
   --rollout-shuffle
   --num-epoch "${SFT_NUM_EPOCH}"
   --rollout-batch-size "${SFT_ROLLOUT_BATCH_SIZE}"
   --global-batch-size "${SFT_GLOBAL_BATCH_SIZE}"

   --loss-type sft_loss
   --loss-mask-type qwen3
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size "${SFT_TENSOR_MODEL_PARALLEL_SIZE:-1}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${SFT_MAX_TOKENS_PER_GPU}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${SFT_LR:-1e-5}"
   --lr-decay-style cosine
   --min-lr "${SFT_MIN_LR:-1e-6}"
   --lr-warmup-fraction "${SFT_LR_WARMUP_FRACTION:-0.1}"
   --weight-decay "${SFT_WEIGHT_DECAY:-0.1}"
   --adam-beta1 "${SFT_ADAM_BETA1:-0.9}"
   --adam-beta2 "${SFT_ADAM_BETA2:-0.95}"
)

WANDB_ARGS=()
if [ "${USE_WANDB:-0}" = "1" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT:-slime-dev}"
      --wandb-group "${WANDB_GROUP:-qwen3-4B-retool-sft}"
   )
   if [ -n "${WANDB_KEY:-}" ]; then
      WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
   fi
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

cd "${SLIME_ROOT}"

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "${SFT_RAY_NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}:${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"HTTP_PROXY\": \"${HTTP_PROXY:-}\",
    \"HTTPS_PROXY\": \"${HTTPS_PROXY:-}\",
    \"http_proxy\": \"${http_proxy:-}\",
    \"https_proxy\": \"${https_proxy:-}\",
    \"NO_PROXY\": \"${NO_PROXY:-}\",
    \"no_proxy\": \"${no_proxy:-}\",
    \"WANDB_HTTP_PROXY\": \"${WANDB_HTTP_PROXY:-${HTTP_PROXY:-}}\",
    \"WANDB_HTTPS_PROXY\": \"${WANDB_HTTPS_PROXY:-${HTTPS_PROXY:-}}\",
    \"WANDB_NO_PROXY\": \"${WANDB_NO_PROXY:-${NO_PROXY:-}}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- bash "${SCRIPT_DIR}/ray_entrypoint.sh" python3 train_async.py \
   --actor-num-nodes "${SFT_ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${SFT_ACTOR_GPUS_PER_NODE}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${MISC_ARGS[@]}"
