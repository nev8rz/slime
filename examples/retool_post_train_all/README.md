# Qwen2.5 ReTool SFT/RL and OPD

This example combines the ReTool and on-policy distillation examples into one post-training flow:

1. Train Qwen2.5-7B-Instruct with ReTool SFT.
2. Export the SFT checkpoint to HF and convert it back to a clean torch-dist checkpoint.
3. Continue Qwen2.5-7B-Instruct with ReTool RL from that clean checkpoint.
4. Train Qwen2.5-3B-Instruct with ReTool SFT.
5. Serve the trained Qwen2.5-7B-Instruct HF checkpoint as an SGLang teacher and run OPD into the 3B student.

## Files

- `00_prepare_qwen2.5_checkpoints.sh`: optional data preparation and HF to torch-dist conversion for the 7B teacher and 3B student.
- `01_qwen2.5_7b_retool_sft.sh`: Qwen2.5-7B-Instruct ReTool SFT.
- `02_prepare_qwen2.5_7b_sft_clean_dist.sh`: Qwen2.5-7B-Instruct SFT torch-dist -> HF -> clean torch-dist.
- `02_qwen2.5_7b_retool_rl.sh`: Qwen2.5-7B-Instruct ReTool RL from the clean SFT torch-dist checkpoint.
- `03_qwen2.5_3b_retool_sft.sh`: Qwen2.5-3B-Instruct ReTool SFT.
- `04_qwen2.5_3b_opd_from_qwen2.5_7b_retool.sh`: Qwen2.5-3B-Instruct OPD using the trained 7B ReTool teacher.
- `05_qwen2.5_retool_aime_bench.sh`: eval-only AIME benchmark. Select checkpoints with `BENCH_MODEL`.
- `06_qwen2.5_retool_aime_bench_all.sh`: runs the AIME benchmark for 3B, 7B, 3B SFT, and 7B SFT.
- `run_all.sh`: runs the stages in order.
- `generate_with_retool.py` and `tool_sandbox.py`: copied ReTool rollout helpers used by RL and OPD. The custom generate path adds ReTool stop strings (`</tool_call>` and Qwen chat end tokens) before each SGLang request so a completed tool call or assistant turn does not keep sampling into hallucinated text.
- `METRICS.md`: W&B 指标说明，覆盖 train、rollout、eval、perf、passrate、multi-turn、debug 和 OPD 指标。

## Environment Variables

The scripts are written for a single 8 GPU node and can be overridden with environment variables.
Every stage sources `env.sh`; override the root variables below to move the whole run to a different filesystem.

| Variable | Default |
| --- | --- |
| `SLIME_ROOT` | `/root/slime` |
| `MEGATRON_PATH` | `/root/Megatron-LM` |
| `MODEL_ROOT` | `/root/Qwen` |
| `DATA_ROOT` | `/root` |
| `OUTPUT_ROOT` | `/root/slime_outputs/retool_post_train_all` |
| `SFT_WANDB_PROJECT` | `retool-sft` |
| `RL_WANDB_PROJECT` | `retool-rl` |
| `OPD_WANDB_PROJECT` | `retool-opd` |
| `BENCH_WANDB_PROJECT` | `retool-bench` |

Derived checkpoint and data paths can still be overridden individually:

| Variable | Default |
| --- | --- |
| `QWEN2_5_7B_HF` | `${MODEL_ROOT}/Qwen2.5-7B-Instruct` |
| `QWEN2_5_7B_TORCH_DIST` | `${MODEL_ROOT}/Qwen2.5-7B-Instruct_torch_dist` |
| `QWEN2_5_7B_SFT_SAVE` | `${OUTPUT_ROOT}/qwen2.5-7B-retool-sft` |
| `QWEN2_5_7B_SFT_HF_EXPORT` | `${OUTPUT_ROOT}/qwen2.5-7B-retool-sft-hf` |
| `QWEN2_5_7B_SFT_CLEAN_DIST` | `${OUTPUT_ROOT}/qwen2.5-7B-retool-sft-clean-dist` |
| `QWEN2_5_7B_RL_LOAD` | `${QWEN2_5_7B_SFT_CLEAN_DIST}` |
| `QWEN2_5_7B_RL_REF_LOAD` | `${QWEN2_5_7B_SFT_CLEAN_DIST}` |
| `QWEN2_5_7B_RL_SAVE` | `${OUTPUT_ROOT}/qwen2.5-7B-retool-rl` |
| `QWEN2_5_7B_RL_SAVE_HF` | `${OUTPUT_ROOT}/qwen2.5-7B-retool-rl-hf` |
| `QWEN2_5_7B_TEACHER_HF` | `${QWEN2_5_7B_RL_SAVE_HF}` |
| `QWEN2_5_3B_HF` | `${MODEL_ROOT}/Qwen2.5-3B-Instruct` |
| `QWEN2_5_3B_REF_LOAD` | `${MODEL_ROOT}/Qwen2.5-3B-Instruct_torch_dist` |
| `QWEN2_5_3B_SFT_SAVE` | `${OUTPUT_ROOT}/qwen2.5-3B-retool-sft` |
| `QWEN2_5_3B_SFT_HF_EXPORT` | `${OUTPUT_ROOT}/qwen2.5-3B-retool-sft-hf` |
| `QWEN2_5_3B_SFT_CLEAN_DIST` | `${OUTPUT_ROOT}/qwen2.5-3B-retool-sft-clean-dist` |
| `QWEN2_5_3B_OPD_LOAD` | `${QWEN2_5_3B_SFT_CLEAN_DIST}` |
| `QWEN2_5_3B_OPD_REF_LOAD` | `${QWEN2_5_3B_SFT_CLEAN_DIST}` |
| `QWEN2_5_3B_OPD_SAVE` | `${OUTPUT_ROOT}/qwen2.5-3B-retool-opd` |
| `QWEN2_5_3B_OPD_SAVE_HF` | `${OUTPUT_ROOT}/qwen2.5-3B-retool-opd-hf` |
| `SFT_PROMPT_DATA` | `./data/retool/ReTool-SFT.parquet` |
| `RL_PROMPT_DATA` | `${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl` |
| `OPD_PROMPT_DATA` | `${RL_PROMPT_DATA}` |
| `EVAL_PROMPT_DATA_2024` | `${DATA_ROOT}/aime-2024/aime-2024.jsonl` |
| `EVAL_PROMPT_DATA_2025` | `${DATA_ROOT}/aime-2025/aime-2025.jsonl` |
| `EVAL_PROMPT_DATA` | `${EVAL_PROMPT_DATA_2024}` |

## Usage

Download or prepare the base assets first:

```bash
MODEL_ROOT=${MODEL_ROOT:-/root/Qwen}
DATA_ROOT=${DATA_ROOT:-/root}

hf download Qwen/Qwen2.5-7B-Instruct --local-dir "${MODEL_ROOT}/Qwen2.5-7B-Instruct"
hf download Qwen/Qwen2.5-3B-Instruct --local-dir "${MODEL_ROOT}/Qwen2.5-3B-Instruct"
python examples/retool_post_train_all/rl_data_preprocess.py \
  --data-root "${DATA_ROOT}" \
  --overwrite
```

`rl_data_preprocess.py` follows the upstream DAPO/ReTool data shape: DAPO train
is converted to `prompt` plus `label`, AIME 2024 is deduplicated from the
upstream repeated parquet split to 30 unique problems, and AIME prompts are
normalized to request `Answer: \boxed{...}` so eval matches the RL reward parser.

Prepare the base checkpoints and ReTool SFT data:

```bash
bash examples/retool_post_train_all/00_prepare_qwen2.5_checkpoints.sh
```

Run the complete flow:

```bash
bash examples/retool_post_train_all/run_all.sh
```

Run one stage at a time:

```bash
bash examples/retool_post_train_all/01_qwen2.5_7b_retool_sft.sh
bash examples/retool_post_train_all/02_prepare_qwen2.5_7b_sft_clean_dist.sh
bash examples/retool_post_train_all/02_qwen2.5_7b_retool_rl.sh
bash examples/retool_post_train_all/03_qwen2.5_3b_retool_sft.sh
bash examples/retool_post_train_all/04_qwen2.5_3b_opd_from_qwen2.5_7b_retool.sh
```

Run eval-only AIME benchmarks. These use slime with `--num-rollout 0`, pass no train `--prompt-data`, and evaluate AIME 2024 plus AIME 2025 with 16 samples per prompt by default:

```bash
BENCH_MODEL=qwen2.5-3b bash examples/retool_post_train_all/05_qwen2.5_retool_aime_bench.sh
BENCH_MODEL=qwen2.5-7b bash examples/retool_post_train_all/05_qwen2.5_retool_aime_bench.sh
BENCH_MODEL=qwen2.5-3b-sft bash examples/retool_post_train_all/05_qwen2.5_retool_aime_bench.sh
BENCH_MODEL=qwen2.5-7b-sft bash examples/retool_post_train_all/05_qwen2.5_retool_aime_bench.sh
```

Or run all four sequentially:

```bash
bash examples/retool_post_train_all/06_qwen2.5_retool_aime_bench_all.sh
```

Benchmark W&B defaults to `BENCH_WANDB_PROJECT=retool-bench`, so it is separated from training runs. The benchmark script relies on `WANDB_API_KEY` from the environment instead of passing it as a CLI argument, keeping Ray entrypoint logs clean. With `BENCH_EVAL_N_SAMPLES_PER_PROMPT=16` and `BENCH_LOG_PASSRATE=1`, the key `eval/<dataset>-pass@16` is pass@16, meaning at least one of the 16 samples for a problem is correct. The key `eval/<dataset>-pass^16` is all16, meaning all 16 samples are correct. The key `eval/<dataset>-mean@16` is the average correct fraction across the 16 samples for each problem, averaged over problems.

The bench defaults to `BENCH_SGLANG_ATTENTION_BACKEND=triton`, `BENCH_SGLANG_SAMPLING_BACKEND=pytorch`, and `BENCH_SGLANG_DISABLE_CUDA_GRAPH=1` to avoid FlashInfer/CUDA-graph JIT cache and link failures on shared cluster nodes. Override these variables if the target machine has a healthy FlashInfer cache/toolchain.

To skip preparation in `run_all.sh`, keep `RUN_PREPARE=0` and make sure the torch-dist checkpoints already exist. To use a different teacher HF export for OPD:

```bash
QWEN2_5_7B_TEACHER_HF=/path/to/qwen2.5-7B-retool-rl-hf \
  bash examples/retool_post_train_all/04_qwen2.5_3b_opd_from_qwen2.5_7b_retool.sh
```

The OPD stage uses SGLang teacher mode because Qwen2.5-7B-Instruct and Qwen2.5-3B-Instruct have different architectures. It also keeps the ReTool custom generate function enabled for the student rollouts.

RL/OPD 训练时怎么看指标，见 `METRICS.md`。

Useful RL/OPD quality knobs:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LOG_PASSRATE` | `1` | Adds `--log-passrate`, which records train rollout `passrate/pass@k` as at-least-one-correct@k. Eval records `eval/<dataset>-pass@k` as at-least-one-correct@k, `eval/<dataset>-pass^k` as all-correct@k, and one `eval/<dataset>-mean@N` where N is `EVAL_N_SAMPLES_PER_PROMPT`. With `EVAL_N_SAMPLES_PER_PROMPT=8`, eval logs pass/pass^ for k=1/2/4/8 and `mean@8`. |
| `LOG_MULTI_TURN` | `1` | Adds `--log-multi-turn` in the 4pod RL recipes, recording train-side multi-turn length and round-number metrics. |
| `EVAL_INTERVAL` | `30` | Eval every 30 rollout steps by default. |
| `EVAL_N_SAMPLES_PER_PROMPT` | `8` | Number of samples per eval prompt. Pass@k is logged for powers of two up to this value. |
| `EVAL_INCLUDE_AIME2025` | `0` | Eval AIME 2024 only by default. Set to `1` to also evaluate AIME 2025. |
| `EVAL_MAX_CONTEXT_LEN` | `32768` | Total eval context budget used by the ReTool custom generate function. |
| `EVAL_MAX_PROMPT_LEN` | `2048` | Eval prompt token filter, aligned with the reference recipe. |
| `EVAL_MAX_RESPONSE_LEN` | `16384` | Eval generation response token cap. |
| `RL_NUM_EPOCH` | `1` | Number of passes over the RL prompt data. The script uses `--num-epoch` so slime derives rollout count from dataset size and rollout batch size. |
| `RL_SAVE_INTERVAL` | `30` | Save every 30 rollout steps by default. |
| `RL_CONTEXT_PARALLEL_SIZE` | `2` | Context parallel size for 7B RL training. This keeps two data-parallel replicas on an 8 GPU node with `TP=2`. |
| `RL_MAX_TOKENS_PER_GPU` | `16384` | Dynamic training token budget per GPU for 7B RL. |
| `RL_ROLLOUT_MAX_CONTEXT_LEN` | `32768` | Total 7B RL rollout context budget. |
| `RL_ROLLOUT_MAX_PROMPT_LEN` | `2048` | RL prompt token filter, aligned with the reference recipe. |
| `RL_ROLLOUT_MAX_RESPONSE_LEN` | `16384` | 7B RL generation response token cap. |
| `OPD_NUM_EPOCH` | `1` | Number of passes over the OPD prompt data. The script uses `--num-epoch` rather than a fixed rollout count. |
| `OPD_SAVE_INTERVAL` | `30` | Save every 30 rollout steps by default. |
| `OPD_CONTEXT_PARALLEL_SIZE` | `2` | Context parallel size for 3B OPD training. |
| `OPD_MAX_TOKENS_PER_GPU` | `16384` | Dynamic training token budget per GPU for 3B OPD. |
| `OPD_ROLLOUT_MAX_CONTEXT_LEN` | `32768` | Total 3B OPD rollout context budget. |
| `OPD_ROLLOUT_MAX_RESPONSE_LEN` | `24576` | 3B OPD generation response token cap. |

## No Thinking Mode

This flow targets `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen2.5-3B-Instruct`, which are non-thinking models: their chat template has no `<think></think>` block and ignores any `enable_thinking` / `reasoning_content` input. All reasoning is plain visible response text.

The ReTool SFT converter follows the upstream preprocessing shape: text before a tool call becomes the assistant `content` that precedes the `tool_calls`; the following `<interpreter>...</interpreter>` block becomes a `tool` role message; for the final answer turn, text before `<answer>...</answer>` is prepended to the tag contents without adding an `Answer:` prefix. Nothing is stored in `reasoning_content`. The SFT scripts still pass `--apply-chat-template-kwargs '{"enable_thinking": false}'`, which Qwen2.5 simply ignores, and use `--loss-mask-type qwen` for the Qwen2.5 chat template.

The local SFT conversion rewrites the original ReTool transcript into real multi-turn tool-call data. Each sample contains `messages` plus a `tools` column. The source `<code>```python ...```</code>` blocks become assistant `tool_calls` to `code_interpreter`; the following `<interpreter>...</interpreter>` blocks become `tool` role messages; the final `<answer>\boxed{...}</answer>` remains `\boxed{...}` visible text. The SFT scripts pass `--tool-key tools`, so Qwen2.5's native chat template injects the `# Tools` system section and renders `<tool_call>...</tool_call>` instead of training on legacy `<code>` tags.

RL and OPD also call the tokenizer's native `apply_chat_template` and pass the code-interpreter tool schema through the `tools` argument. The custom generate path does not prefill any `<think>` block, so the model produces plain reasoning followed by `<tool_call>...</tool_call>` when it wants code execution; otherwise the trajectory terminates and the reward function checks the final visible `\boxed{...}` answer. The RL/OPD scripts default `RETOOL_THINKING_MODE=no_think`; this knob is now inert for the rollout code (kept only for backward compatibility) because Qwen2.5 has no thinking mode.

The RL and OPD default sampling parameters are `temperature=0.6`, `top_p=0.95`, and `top_k=20`. Evaluation uses AIME 2024 by default with 8 samples per prompt; set `EVAL_INCLUDE_AIME2025=1` to also add AIME 2025.

## OPD Update Formula

The 3B OPD stage is pure on-policy distillation in the current script. The rollout reward returned by `slime.rollout.on_policy_distillation.post_process_rewards` is zero, so the useful training signal comes from the teacher KL term that is applied to the token-level advantages.

For each prompt `x_i`, the student rollout policy `pi_k` samples a response:

```text
y_i = (y_{i,1}, ..., y_{i,T_i}) ~ pi_k(. | x_i)
```

For each sampled response token, slime stores the student rollout logprob and asks the SGLang teacher for the teacher logprob:

```text
s_{i,t} = log pi_k(y_{i,t} | x_i, y_{i,<t})
q_{i,t} = log pi_T(y_{i,t} | x_i, y_{i,<t})
```

This is token-level and sampled-token only. It is not a full-vocab KL; slime does not sum over all vocabulary tokens for OPD.

The base advantage is computed first by the selected advantage estimator:

```text
A_base_{i,t} = AdvantageEstimator(reward_i, ref_kl, values, ...)
```

In this pure OPD script:

```text
reward_i = 0
```

With `--advantage-estimator grpo`, the base advantage is therefore effectively:

```text
A_base_{i,t} = 0
```

OPD then applies a sampled-token reverse-KL penalty to the advantages:

```text
A_{i,t} = A_base_{i,t} - beta * (s_{i,t} - q_{i,t})
```

where `beta = opd_kl_coef`. In the pure OPD case this becomes:

```text
A_{i,t} = beta * (log pi_T(y_{i,t} | x_i, y_{i,<t}) - log pi_k(y_{i,t} | x_i, y_{i,<t}))
```

The policy update still uses slime's PPO-style clipped policy loss. During training, the current student policy `pi_theta` is forwarded again:

```text
r_{i,t}(theta) =
  pi_theta(y_{i,t} | x_i, y_{i,<t}) / pi_k(y_{i,t} | x_i, y_{i,<t})
```

The minimized policy loss is:

```text
L_pg(theta) = mean_{i,t} max(
  -r_{i,t}(theta) * A_{i,t},
  -clip(r_{i,t}(theta), 1 - eps, 1 + eps_high) * A_{i,t}
)
```

If entropy or regular reference KL loss is enabled, the final loss is:

```text
L(theta) = L_pg(theta)
           - entropy_coef * entropy(pi_theta)
           + kl_loss_coef * KL_ref(pi_theta, pi_ref)
```

The default OPD script sets `entropy_coef=0` and `kl_loss_coef=0`, so the update is driven by the sampled-token reverse-KL distillation advantage above.
