#!/bin/bash

# Qwen2.5-7B-Instruct ReTool SFT.

# for rerun the task
if [ "${SKIP_PRELAUNCH_CLEANUP:-0}" != "1" ]; then
   pkill -9 sglang
   sleep 3
   ray stop --force
   pkill -9 ray
   pkill -9 python
   sleep 3
   pkill -9 ray
   pkill -9 python
fi

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

export MODEL_ARGS_ROTARY_BASE="${QWEN2_5_7B_ROTARY_BASE}"
source "${SLIME_ROOT}/scripts/models/qwen2.5-7B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${QWEN2_5_7B_HF}"
   --ref-load "${QWEN2_5_7B_REF_LOAD}"
   --save "${QWEN2_5_7B_SFT_SAVE}"
   --save-interval "${SFT_SAVE_INTERVAL}"
   --rotary-base "${QWEN2_5_7B_ROTARY_BASE}"
)

if [ -n "${QWEN2_5_7B_SFT_LOAD}" ]; then
   CKPT_ARGS+=(--load "${QWEN2_5_7B_SFT_LOAD}")
fi

if [ -n "${QWEN2_5_7B_SFT_SAVE_HF}" ]; then
   CKPT_ARGS+=(--save-hf "${QWEN2_5_7B_SFT_SAVE_HF}")
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
   --loss-mask-type qwen
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
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
      --wandb-project "${SFT_WANDB_PROJECT}"
      --wandb-group "${WANDB_GROUP:-qwen2.5-7B-retool-sft}"
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
RAY_START_ARGS=(
   --head
   --node-ip-address "${MASTER_ADDR}"
   --num-gpus "${SFT_RAY_NUM_GPUS}"
   --disable-usage-stats
   --dashboard-host=0.0.0.0
   --dashboard-port="${RAY_DASHBOARD_PORT:-8265}"
)
if [ -n "${RAY_PORT:-}" ]; then
   RAY_START_ARGS+=(--port="${RAY_PORT}")
fi
if [ -n "${RAY_OBJECT_MANAGER_PORT:-}" ]; then
   RAY_START_ARGS+=(--object-manager-port="${RAY_OBJECT_MANAGER_PORT}")
fi
if [ -n "${RAY_NODE_MANAGER_PORT:-}" ]; then
   RAY_START_ARGS+=(--node-manager-port="${RAY_NODE_MANAGER_PORT}")
fi
if [ -n "${RAY_RUNTIME_ENV_AGENT_PORT:-}" ]; then
   RAY_START_ARGS+=(--runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}")
fi
if [ -n "${RAY_CLIENT_SERVER_PORT:-}" ]; then
   RAY_START_ARGS+=(--ray-client-server-port="${RAY_CLIENT_SERVER_PORT}")
fi
if [ -n "${RAY_DASHBOARD_AGENT_LISTEN_PORT:-}" ]; then
   RAY_START_ARGS+=(--dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT}")
fi
if [ -n "${RAY_DASHBOARD_AGENT_GRPC_PORT:-}" ]; then
   RAY_START_ARGS+=(--dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}")
fi
if [ -n "${RAY_METRICS_EXPORT_PORT:-}" ]; then
   RAY_START_ARGS+=(--metrics-export-port="${RAY_METRICS_EXPORT_PORT}")
fi
if [ -n "${RAY_MIN_WORKER_PORT:-}" ]; then
   RAY_START_ARGS+=(--min-worker-port="${RAY_MIN_WORKER_PORT}")
fi
if [ -n "${RAY_MAX_WORKER_PORT:-}" ]; then
   RAY_START_ARGS+=(--max-worker-port="${RAY_MAX_WORKER_PORT}")
fi
if [ -n "${RAY_TEMP_DIR:-}" ]; then
   RAY_START_ARGS+=(--temp-dir="${RAY_TEMP_DIR}")
fi
ray start "${RAY_START_ARGS[@]}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}:${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}" \
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
