# ReTool 训练指标说明

这份文档解释 `examples/retool_post_train_all` 这条 Qwen2.5 ReTool SFT/RL + OPD 线路里，slime 默认会记录的训练指标。这里不讲 W&B 自动采集的 `system/*` 机器指标，只讲模型训练、rollout、eval、perf、passrate、multi-turn 和 OPD 相关指标。

## 去哪里看

默认主要看 W&B。进入 run 后按下面这些前缀搜索：

```text
eval/*
train/*
rollout/*
perf/*
passrate/*
multi_turn/*
debug/*
```

不同前缀使用不同 step 轴：

| Step 轴 | 对应指标 | 含义 |
| --- | --- | --- |
| `eval/step` | `eval/*` | 当前 eval 对应的 rollout id。 |
| `train/step` | `train/*` | optimizer step id。`grpo_iterations > 1` 时，一个 rollout 会产生多个 train step。 |
| `rollout/step` | `rollout/*`, `perf/*`, `passrate/*`, `multi_turn/*` | rollout id。 |
| W&B internal step | `debug/*` | ReTool rollout 函数直接写入的调试指标，不显式绑定 `rollout/step`。 |

当前 recipe 的第一个质检点是：`eval/aime2024` 和 `eval/aime2025` 应该先出现在 `eval/step = 0`，然后才出现 `train/step = 0` 或 `train/step = 1`。这说明 val-before-train 真的跑了。

## 方向说明

| 方向 | 怎么理解 |
| --- | --- |
| 越大越好 | 通常希望这个指标上升。 |
| 越小越好 | 通常希望这个指标下降。 |
| 稳定 / 有界 | 没有单调越大或越小最好，重点看是否漂移、尖峰、坍塌或饱和。 |
| 诊断指标 | 主要用于解释现象，不建议单独优化它。 |

## Eval 指标

Eval 是最重要的离线质量信号。本 recipe 默认会在 AIME 2024 和 AIME 2025 上评估。

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `eval/<dataset>` | eval 数据集上的平均 reward / score。对 AIME verifier 来说，一般接近 accuracy。 | 越大越好。重点看 `eval/aime2024` 和 `eval/aime2025` 是否提升；如果 rollout reward 涨但 eval 掉，要警惕过拟合或 reward hack。 |
| `eval/<dataset>-truncated_ratio` | eval 响应被最大长度截断的比例。 | 越小越好。高说明答案被截断，eval 分数可能被低估。 |
| `eval/<dataset>/response_len/mean` | eval 有效响应长度均值。 | 诊断指标。变长可能表示推理更充分，也可能是啰嗦或工具循环变多。 |
| `eval/<dataset>/response_len/median` | eval 有效响应长度中位数。 | 诊断指标。和 mean 对比，如果 mean 远大于 median，说明少数超长样本拖尾。 |
| `eval/<dataset>/response_len/max` | eval 最长有效响应长度。 | 越小 / 有界越好。接近 `eval-max-response-len` 时说明有截断风险。 |
| `eval/<dataset>/response_len/min` | eval 最短有效响应长度。 | 诊断指标。非常小可能是空答、提前停止或 parser 失败。 |
| `eval/<dataset>/zero_std/count_<reward>` | 一个 prompt group 内所有采样响应 reward 都一样的 group 数。 | 通常越小越好。全对说明题太容易、学习信号少；全错说明题太难或解析失败、也没有 GRPO 相对优势信号。 |
| `eval/<dataset>/prefix_cache_hit_rate` | eval 时 prompt token 命中 SGLang prefix cache 的比例。 | serving 效率指标，越大越省，不是质量指标。 |
| `eval/<dataset>/avg_cached_tokens_per_sample` | 每个 eval sample 平均 cache token 数。 | 越大通常越省。如果一直是 0，说明 prefix cache 没帮上忙。 |
| `eval/<dataset>/repetition_frac` | eval 响应被判为严重重复的比例。 | 越小越好。尖峰说明生成退化。 |
| `eval/<dataset>/truncated_ratio` | dataset 前缀下的 sample-level 截断比例。 | 越小越好。它和 `<dataset>-truncated_ratio` 可能同时出现，取决于 eval 数据结构。 |
| `eval/<dataset>-pass@1`, `eval/<dataset>-pass@2`, ... | 开启 `--log-passrate` 后记录的 eval all-correct@k：从同一题的 k 次采样中抽 k 个，全部正确的概率估计。默认 `EVAL_N_SAMPLES_PER_PROMPT=8` 时会记录到 `pass@8`，表示 8 次全对。 | 越大越好，但会随 k 变大而更严格，通常单调不升。`pass@1` 等价于单次采样准确率。 |
| `eval/<dataset>-any@1`, `eval/<dataset>-any@2`, ... | eval any-correct@k：k 次采样里至少一次正确的概率估计。旧版 `eval/<dataset>-pass@k` 实际是这个语义。 | 越大越好，并且应随 k 单调不降。用于衡量采样放大后的可命中能力。 |
| `eval/<dataset>-mean@1`, `eval/<dataset>-mean@2`, ... | eval mean-correct@k：同一道题 k 次采样的平均正确比例的期望，再跨题平均。 | 越大越好。它反映平均单样本质量；在每题采样数相同且 reward 为 0/1 时，通常和 `eval/<dataset>` 数值一致，但和 `pass@k`、`any@k` 一起看更直观。 |

## Train 指标

这些来自 Megatron 训练侧 forward/backward。

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `train/loss` | 最终优化的 loss：policy loss 减 entropy bonus，加可选 reference KL loss。 | 稳定 / 有界。GRPO 中 advantage 会中心化，所以 loss 接近 0 很正常，不能只用它判断质量。重点看 NaN、inf、大尖峰。 |
| `train/pg_loss` | PPO-style clipped policy gradient loss，不含 entropy 和 KL 加项。 | 稳定 / 有界。接近 0 常见；长期大幅波动说明更新信号太猛。 |
| `train/entropy_loss` | 当前 policy 在训练 token 上的平均 entropy。 | 稳定 / 有界。太低可能策略过早确定、探索不足；太高可能仍然发散。本 recipe 通常 `entropy_coef=0`，所以主要做诊断。 |
| `train/pg_clipfrac` | PPO clipping 被触发的 token 比例。 | 越小 / 中等越好。接近 0 说明更新很小；长期很高说明 LR、advantage scale 或数据分布让更新过猛。 |
| `train/ppo_kl` | 当前训练 policy 和 rollout old policy 之间的近似 KL，来自 `old_log_probs - log_probs`。 | 越小 / 有界越好。rollout 后第一步应接近 0；尖峰表示 actor 相对采样策略移动太远。 |
| `train/kl_loss` | 当前 actor 和 reference model 的近似 KL。`--use-kl-loss` 开启时记录。本 recipe 里 `kl_loss_coef=0`，所以只监控不进 loss。 | 越小 / 有界越好。actor 和 ref 同源 clean SFT 时应从接近 0 开始；持续升高表示策略远离 SFT/ref。 |
| `train/train_rollout_logprob_abs_diff` | rollout 时记录的 logprob 和训练侧重算 logprob 的绝对差。 | 越小越好。这是 rollout/train 一致性检查；长期超过 0.1 左右很可疑。 |
| `train/grad_norm` | backward 后的全局梯度范数。 | 稳定 / 有界。尖峰表示不稳定；长期为 0 表示没有学习信号、mask 有问题或 reward/advantage 全没了。 |
| `train/lr-pg_0`, `train/lr-pg_1`, ... | optimizer 各 param group 的学习率。 | 诊断指标。应该符合脚本 schedule。本 recipe 默认常数 LR。 |
| `train/global_batch_size` | 当前 train step 实际使用的 sample 数。 | 应稳定在预期值。动态 batch 或不均匀数据可能让它变化。 |
| `train/mtp_loss` | 开启 MTP 训练时的 multi-token prediction 辅助 loss。 | 只在 MTP 开启时看，通常越小越好。 |
| `train/critic-*` | critic 训练时的 loss/grad/lr 等指标。 | 解释同 actor train 指标。本 recipe 的 GRPO 不用 critic。 |
| `train/opd_reverse_kl` | OPD batch 中的 sampled-token reverse KL proxy。 | OPD 中越小越好，表示 student 更贴近 teacher。 |
| `train/ois` | 开启 mismatch / TIS 相关指标时的 off-policy importance ratio。 | 稳定 / 有界。明显偏离 1 表示 off-policy mismatch 明显。 |
| `train/tis` | token importance sampling 权重。 | 稳定 / 有界。过大说明修正很强，可能不稳。 |
| `train/tis_clipfrac` | TIS 权重被 clip 的比例。 | 越小越好。高说明大量 off-policy 修正被截断。 |
| `train/opsm_clipfrac` | 开启 OPSM 时被 mask/clip 的比例。 | 越小 / 中等越好。高说明很多 token 更新被过滤。 |

## Rollout 指标

Rollout 指标有两类来源：

- rollout 侧：生成样本、SGLang serving、cache、truncation 等。
- train 侧：rollout 数据转成训练 batch 后，对 logprob、reward、advantage 等字段做 reduce。

### 样本与 reward

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `rollout/response_len/mean` | 生成有效响应长度均值。 | 诊断指标。要足够支持推理，但不应持续贴近最大长度。 |
| `rollout/response_len/median` | 生成有效响应长度中位数。 | 诊断指标。和 mean 对比看长尾。 |
| `rollout/response_len/max` | 最长有效响应长度。 | 越小 / 有界越好。接近 `rollout-max-response-len` 表示截断风险。 |
| `rollout/response_len/min` | 最短有效响应长度。 | 诊断指标。很小可能是空输出或异常早停。 |
| `rollout/truncated_ratio` | rollout 响应达到长度上限的比例。 | 越小越好。高截断会污染 reward 和 advantage。 |
| `rollout/repetition_frac` | 严重重复响应比例。 | 越小越好。尖峰表示生成退化。 |
| `rollout/zero_std/count_<reward>` | GRPO group 内所有样本 reward 相同的 group 数。 | 越小越好。全错或全对都缺少相对优势学习信号。 |
| `rollout/rewards` | reward 后处理 / advantage 准备后的平均 reward。 | 诊断指标。GRPO 中可能被中心化到接近 0；直观质量优先看 `raw_reward`。 |
| `rollout/raw_reward` | verifier 原始 reward 均值。 | 越大越好。二值数学 reward 下大致就是训练 rollout 成功率。 |
| `rollout/advantages` | policy gradient 使用的 advantage 均值。 | 诊断指标。归一化后均值常接近 0；更重要的是它是否带来有效更新和 eval 提升。 |
| `rollout/returns` | 训练使用的 return 均值。 | 诊断指标。无 critic GRPO 里通常和 advantages 接近。 |
| `rollout/response_lengths` | train 侧 batch conversion 后的 response length 均值。 | 诊断指标。应和 `rollout/response_len/mean` 大体一致。 |
| `rollout/total_lengths` | prompt + response 总 token 长度均值。 | 越小 / 有界越好。接近 context 上限时要警惕截断或 OOM。 |
| `rollout/truncated` | train 侧截断 flag 均值。 | 越小越好，应和 `rollout/truncated_ratio` 大体一致。 |
| `rollout/log_probs` | 训练侧重算的 actor logprob 均值。 | 诊断指标。健康初始状态通常为负且不极端。 |
| `rollout/rollout_log_probs` | SGLang rollout 时记录的 actor logprob 均值。 | 诊断指标。应和训练侧 `rollout/log_probs` 接近。 |
| `rollout/ref_log_probs` | reference model 在采样 token 上的 logprob 均值。 | 诊断指标。和 actor logprob、`rollout/kl` 一起看。 |
| `rollout/kl` | rollout 处理阶段的 actor/ref token KL。 | 越小 / 有界越好。actor/ref 同为 clean SFT 时初始应接近 0。 |
| `rollout/entropy` | rollout entropy，如果启用了相关计算。 | 稳定 / 有界。过低可能坍塌，过高可能仍然不确定。 |
| `rollout/values` | critic value 预测。 | critic 模式才有；本 GRPO recipe 不用。 |
| `rollout/teacher_log_probs` | OPD 中 teacher 对 student sampled tokens 的 logprob。 | 诊断指标。和 student logprob 对比看 teacher 是否更认可这些 token。 |
| `rollout/opd_reverse_kl` | OPD sampled-token reverse KL proxy。 | 越小越好。高说明 student 和 teacher 在 on-policy 样本上差异大。 |
| `rollout/prefix_cache_hit_rate` | SGLang prefix cache 命中比例。 | serving 效率指标，越大越省，不代表质量好。 |
| `rollout/avg_cached_tokens_per_sample` | 每个 sample 平均 cache token 数。 | 越大通常 throughput 更好。 |
| `rollout/spec_accept_rate` | speculative decoding accept rate。 | 开 speculative 时越大越好；低说明 draft 效果差。 |
| `rollout/spec_accept_length` | speculative decoding 平均接受长度。 | 开 speculative 时越大越好。 |
| `rollout/error_cat/<category>` | 开启 `--log-reward-category` 后，各 reward error category 的样本比例。 | 坏类别越小越好，用于定位失败模式。 |

### 正确样本指标

仅在 `--log-correct-samples` 开启且这些辅助字段被后续聚合记录时出现；默认质量判断不依赖它们。

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `rollout/correct_length/p25`, `p50`, `p75`, `p100` | 正确样本长度分桶。 | 诊断指标。能力变强后通常正确样本长度会更稳定，未必单调变短。 |
| `rollout/correct_entropy` | 正确样本上的 entropy。 | 稳定 / 有界。太低可能答案路径很死，太高说明即使正确也不确定。 |

## Debug 指标

这些来自 `examples/retool_post_train_all/generate_with_retool.py` 里的 ReTool rollout 函数，主要用来排查 code-interpreter 轨迹。它们不显式绑定 `rollout/step`，所以看趋势时不要和 `train/*`、`eval/*` 强行对齐。

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `debug/payload_length` | 当前 turn 发给 SGLang 的 prompt + 已生成 response 的字符长度。 | 越小 / 有界越好。持续变大说明多轮工具轨迹变长，可能导致上下文压力或 OOM。 |
| `debug/available_tools` | 当前样本可用工具数量。 | 诊断指标。本 recipe 通常应该稳定；突然变化说明 tools 注入或数据格式变了。 |
| `debug/tools_used` | 当前 rollout 已成功执行的 `code_interpreter` tool call 次数。 | 诊断指标。太低可能没学会用工具；太高可能工具循环。要和 reward/eval 一起看。 |
| `debug/turn` | 当前 ReTool 多轮生成 turn index。 | 诊断指标。高 turn 说明工具交互更深，也可能是循环或单题耗时变长。 |

## Passrate 指标

开启 `--log-passrate` 时出现。当前 slime 没有单独的 eval-only passrate 开关；这个开关会同时记录 rollout passrate 和 eval passrate。

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `passrate/pass@1` | rollout batch 内，单次采样能解题的概率估计。 | 越大越好，最接近单样本质量。 |
| `passrate/pass@2`, `passrate/pass@4`, `passrate/pass@8`, ... | k 次采样至少一次正确的概率估计。 | 越大越好，且应随 k 单调不降。pass@k 高但 pass@1 低，说明采样有多样性但单次可靠性不足。 |
| `eval/<dataset>-pass@k` | eval 上的 all-correct@k。 | 越大越好，但更严格；默认 `pass@8` 表示 8 次全对。选 checkpoint 时优先看 `pass@1`、最高 k 的 `pass@k` 和截断率。 |
| `eval/<dataset>-any@k` | eval 上的 any-correct@k。 | 越大越好；表示 k 次里至少一次正确。它和 train 侧 `passrate/pass@k` 的语义一致，但数据集不同。 |
| `eval/<dataset>-mean@k` | eval 上的 mean-correct@k。 | 越大越好；表示同题多次采样的平均正确比例，适合作为 all/any 之间的中间口径。 |

## Perf 指标

Perf 只能解释效率，不能替代质量指标。吞吐变好但 eval 掉了，不算好 run。

### Rollout 吞吐

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `perf/rollout_time` | 一个 rollout step 的墙钟时间。 | 同 batch/长度配置下越小越好。 |
| `perf/tokens_per_gpu_per_sec` | 每张 GPU 每秒生成 response token 数。 | 越大越好，是主要 rollout 吞吐指标。 |
| `perf/effective_tokens_per_gpu_per_sec` | 按有效 response length 计算的每 GPU token/s。 | 越大越好。工具 token/mask 影响较大时优先看它。 |
| `perf/longest_sample_tokens_per_sec` | 最长样本 token 数 / rollout time。 | 越大越好。低说明单个超长样本拖慢 batch。 |
| `perf/longest_effective_sample_tokens_per_sec` | 用 effective length 算的最长样本速度。 | 越大越好。 |
| `perf/non_generation_time/mean` | 每个样本非 generation 时间均值。 | 越小越好。高说明工具调用、reward、环境交互很慢。 |
| `perf/non_generation_time/median` | 非 generation 时间中位数。 | 越小越好。和 mean 对比可看长尾。 |
| `perf/non_generation_time/max` | 最大非 generation 时间。 | 越小越好。少数慢工具调用会拖住整个 rollout。 |
| `perf/non_generation_time/min` | 最小非 generation 时间。 | 诊断指标。纯生成样本通常接近 0。 |
| `perf/longest_sample_non_generation_time` | 最长样本对应的非 generation 时间。 | 越小越好，用来拆分“生成慢”还是“工具/reward 慢”。 |
| `perf/longest_sample_tokens_per_sec_without_non_generation` | 扣除非 generation 时间后的最长样本 token/s。 | 越大越好。 |
| `perf/longest_effective_sample_non_generation_time` | 最长 effective sample 的非 generation 时间。 | 越小越好。 |
| `perf/longest_effective_sample_tokens_per_sec_without_non_generation` | 扣除非 generation 时间后的最长 effective sample token/s。 | 越大越好。 |

### SGLang request timing

这些指标来自 SGLang response trace。大多数都有 `/mean`, `/median`, `/max`, `/min` 后缀。

| 指标 pattern | 含义 | 怎么看 |
| --- | --- | --- |
| `perf/request/e2e_latency/{mean,median,max,min}` | request 端到端延迟。 | 越小越好，尤其关注 max 长尾。 |
| `perf/request/queue_time/{mean,median,max,min}` | request 排队等待时间。 | 越小越好。持续升高说明 serving 饱和。 |
| `perf/decode/throughput/{mean,median,max,min}` | SGLang decode throughput。 | 越大越好。 |
| `perf/prefill/bootstrap_queue_duration/{mean,median,max,min}` | PD prefill bootstrap 排队时间。 | 越小越好。高说明 prefill 侧排队。 |
| `perf/prefill/bootstrap_duration/{mean,median,max,min}` | PD prefill bootstrap 时间。 | 越小越好。 |
| `perf/prefill/alloc_wait_duration/{mean,median,max,min}` | prefill 分配资源等待时间。 | 越小越好。高说明显存/资源竞争。 |
| `perf/prefill/forward_duration/{mean,median,max,min}` | prefill forward 时间。 | 同 prompt 长度下越小越好。 |
| `perf/prefill/transfer_queue_duration/{mean,median,max,min}` | PD 模式下 transfer 前排队时间。 | 越小越好。 |
| `perf/prefill/transfer_speed_gb_s/{mean,median,max,min}` | transfer 速度。 | 越大越好。 |
| `perf/prefill/transfer_total_mb/{mean,median,max,min}` | transfer 数据量。 | 诊断指标。数据多本身不坏，要和 latency 一起看。 |
| `perf/prefill/retry_count/{mean,median,max,min}` | prefill transfer retry 次数。 | 越小越好，非零说明链路不稳。 |
| `perf/decode/prealloc_duration/{mean,median,max,min}` | decode 预分配时间。 | 越小越好。 |
| `perf/decode/bootstrap_duration/{mean,median,max,min}` | decode bootstrap 时间。 | 越小越好。 |
| `perf/decode/alloc_wait_duration/{mean,median,max,min}` | decode 分配等待时间。 | 越小越好。 |
| `perf/decode/transfer_duration/{mean,median,max,min}` | decode transfer 时间。 | 越小越好。 |
| `perf/decode/forward_duration/{mean,median,max,min}` | decode forward 时间。 | 同生成长度下越小越好。 |

### 训练吞吐和调度

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `perf/train_wait_time` | 训练 actor 等待 rollout/data 的时间。 | 越小越好。高说明训练侧在等 rollout。 |
| `perf/train_time` | 一个训练 step 的训练侧总时间。 | 同配置下越小越好。 |
| `perf/ref_log_probs_time` | 计算 reference logprobs 的时间。 | 越小越好。高说明 ref forward 是瓶颈。 |
| `perf/log_probs_time` | 计算 actor logprobs 的时间，如果该 timer 存在。 | 越小越好。 |
| `perf/actor_train_time` | actor forward/backward/step 时间。 | 同配置下越小越好。 |
| `perf/data_preprocess_time` | rollout 数据转 train batch 的时间。 | 越小越好。高说明 packing/mask/data conversion 慢。 |
| `perf/update_weights_time` | actor 权重同步到 rollout engines 的时间。 | 越小越好。高会拖慢 train-rollout 循环。 |
| `perf/sleep_time` | colocate 模式下让 actor/engine sleep 或释放显存的时间。 | 越小越好，但非零正常。 |
| `perf/wake_up_time` | wake up actor/engine 的时间。 | 越小越好。 |
| `perf/ref_log_probs_tflops` | reference forward 估算 TFLOP/s。 | 越大越好。 |
| `perf/log_probs_tflops` | actor logprob forward 估算 TFLOP/s。 | 越大越好。 |
| `perf/actor_train_tflops` | actor 训练估算 TFLOP/s。 | 越大越好。 |
| `perf/actor_train_tok_per_s` | actor 训练 token/s。 | 越大越好。 |
| `perf/step_time` | `train_wait_time + train_time`。 | 越小越好，要结合 `wait_time_ratio` 看。 |
| `perf/wait_time_ratio` | step 时间里等待占比。 | 越小越好。接近 1 表示训练 GPU 大量等 rollout/data；接近 0 表示训练本身占主导。 |

### 权重同步指标

主要在 delta / disk weight sync 模式下出现。

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `perf/update_weights_density` | delta sync 中被传输的权重密度。 | 越小越省。很高说明 delta sync 省不了多少。 |
| `perf/update_weights_wire_bytes` | 权重同步走网络的字节数。 | 越小越好。 |
| `perf/update_weights_flushes_per_rank` | 每个 rank 的 flush 次数。 | 越小 / 稳定越好。突然变多说明 chunk 太碎或同步效率差。 |
| `perf/update_weights_disk_bytes_pre_compress` | disk sync 压缩前字节数。 | 越小越好。 |
| `perf/update_weights_disk_bytes_post_compress` | disk sync 压缩后字节数。 | 越小越好。 |
| `perf/update_weights_compression_ratio` | 压缩前 / 压缩后。 | 越大说明压缩越有效。 |

## Multi-turn 指标

开启 `--log-multi-turn` 后出现。ReTool 有工具轨迹，所以如果后面打开这个开关，这些指标很有用。

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `multi_turn/raw_response_length/response_length_mean` | 包含 observation/tool segment 的原始响应长度均值。 | 诊断指标。上升表示轨迹变长。 |
| `multi_turn/raw_response_length/response_length_max` | 原始响应最大长度。 | 越小 / 有界越好，接近上限说明截断风险。 |
| `multi_turn/raw_response_length/response_length_min` | 原始响应最小长度。 | 诊断指标。很小可能提前结束。 |
| `multi_turn/raw_response_length/response_length_clip_ratio` | 原始响应被 max length clip 的比例。 | 越小越好。 |
| `multi_turn/wo_obs_response_length/response_length_mean` | 去掉 observation/tool tokens 后、从 loss mask 角度看到的响应长度均值。 | 诊断指标，更接近真正训练到的 token。 |
| `multi_turn/wo_obs_response_length/response_length_max` | 去 observation 后最大响应长度。 | 越小 / 有界越好。 |
| `multi_turn/wo_obs_response_length/response_length_min` | 去 observation 后最小响应长度。 | 诊断指标。 |
| `multi_turn/multi_turn_metric/round_number_mean` | 平均交互轮数。 | 诊断指标。太高可能工具循环低效，太低可能没学会用工具。 |
| `multi_turn/multi_turn_metric/round_number_max` | 最大交互轮数。 | 越小 / 有界越好。 |
| `multi_turn/multi_turn_metric/round_number_min` | 最小交互轮数。 | 诊断指标。 |

## OPD 指标

开启 `--use-opd` 或 batch 里包含 OPD 字段时出现。

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `rollout/teacher_log_probs` | teacher 在 student sampled tokens 上的 logprob。 | 诊断指标。和 student logprob 对比看 teacher 是否更认可这些 token。 |
| `rollout/opd_reverse_kl` | rollout 侧 sampled-token reverse KL proxy。 | 越小越好。下降说明 student 在 on-policy 样本上更接近 teacher。 |
| `train/opd_reverse_kl` | train 侧 reduce 后的 OPD reverse KL proxy。 | 越小越好。要和 eval 一起看；KL 下降但 eval 不涨，可能是过度蒸馏或任务不匹配。 |

## 健康 run 的早期形态

一个健康的 Qwen2.5 ReTool RL 早期 run 通常应该是：

1. `eval/aime2024` 和 `eval/aime2025` 在 `eval/step = 0` 先出现。
2. `rollout/raw_reward` 非零，且没有被全错或全对 group 完全占满。
3. `rollout/zero_std/count_*` 不要占据大部分 rollout batch。
4. `rollout/truncated_ratio` 和 `eval/*-truncated_ratio` 保持较低。
5. actor/ref 来自同一个 clean SFT checkpoint 时，`train/ppo_kl`、`train/kl_loss`、`rollout/kl` 初始应接近 0。
6. `train/pg_clipfrac` 不应长期很高。
7. `train/grad_norm` 应有限且无剧烈尖峰。
8. `perf/wait_time_ratio` 用来判断是 rollout-bound 还是 train-bound。
9. `eval/*` 提升或持平，同时 `rollout/raw_reward` 提升；如果 rollout reward 涨但 eval 掉，优先怀疑 reward overfitting 或 train prompt exploit。

## 常见异常模式

| 现象 | 可能原因 | 先查什么 |
| --- | --- | --- |
| 没有 `eval/step = 0` 就开始 train | `start_rollout_id` 不是 0，或设置了 `--skip-eval-before-train`。 | 查 args log 里的 `start_rollout_id`、`skip_eval_before_train`、`eval_interval`。 |
| `rollout/truncated_ratio` 很高 | max response length 太小、模型啰嗦、thinking/tool loop 太长。 | 先看样本和 reward parser，再决定是否加长上限。 |
| `rollout/raw_reward` 接近 0，且很多 `zero_std/count_0` 或负 reward group | 题太难、verifier/parser 不匹配、工具调用失败。 | 抽样看 rollout 样本和 reward function 输出。 |
| `train/ppo_kl` 或 `train/pg_clipfrac` 尖峰 | policy update 太激进。 | 降 LR、检查 advantage scale、定位异常 batch。 |
| `train/kl_loss` 快速上升 | actor 快速远离 reference。 | 如果需要强 reference 约束，提高 `kl_loss_coef`；同时确认 ref checkpoint 没错。 |
| `train/train_rollout_logprob_abs_diff` 高 | rollout/training logprob 不一致。 | 查 tokenizer/template、routing replay、精度、rollout logprob 是否 stale。 |
| `perf/wait_time_ratio` 接近 1 | 训练侧大量等待 rollout/data。 | 提升 rollout 吞吐、降低生成长度、增加 rollout 资源。 |
| `perf/update_weights_time` 高 | 权重同步瓶颈。 | 查 sync mode、TP/PP、网络和 SGLang engine overlap。 |
| eval 下降但 rollout reward 上升 | reward overfitting 或分布偏移。 | checkpoint 选择优先看 eval；抽样查 reward hack。 |
