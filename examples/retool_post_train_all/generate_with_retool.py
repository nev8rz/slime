# Adapted from the upstream ReTool recipe.
import json
import os
import re
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample
from slime.utils.wandb_utils import setup_wandb_env_vars

try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e

from tool_sandbox import SEMAPHORE, TOOL_CONFIGS, tool_registry

# Strings/tokens that end an assistant turn: a tool call (`</tool_call>`) or the
# turn terminator (`<|im_end|>`). `no_stop_trim` keeps these in the output so we
# can append the next turn against a well-formed transcript.
RETOOL_STOP_STRINGS = ("</tool_call>", "<|im_end|>")
TOOL_CALL_PATTERN = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
DEFAULT_MAX_NEW_TOKENS_PER_TURN = 8192
DEFAULT_OVERLONG_BUFFER_LEN = 4096
DEFAULT_OVERLONG_PENALTY_FACTOR = 1.0

OVERLONG_ENABLE_ENV = "RETOOL_OVERLONG_PENALTY_ENABLE"
OVERLONG_MAX_RESPONSE_LEN_ENV = "RETOOL_OVERLONG_MAX_RESPONSE_LEN"
OVERLONG_BUFFER_LEN_ENV = "RETOOL_OVERLONG_BUFFER_LEN"
OVERLONG_PENALTY_FACTOR_ENV = "RETOOL_OVERLONG_PENALTY_FACTOR"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use Python tools to solve "
    "mathematical problems. When you need to perform calculations, use the "
    "code_interpreter tool to execute code and get results."
)


def _extract_prompt_text(prompt: Any) -> str:
    """Flatten a slime prompt (string or list of user-message dicts) to text."""
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")
    if not prompt:
        raise ValueError("DAPO-style prompt list must not be empty.")

    contents = []
    for index, message in enumerate(prompt):
        if not isinstance(message, dict):
            raise TypeError("DAPO-style prompt list must contain only dict items.")
        role = message.get("role", "user")
        if role != "user":
            raise ValueError(f"Unsupported DAPO-style prompt role at index {index}: {role!r}")
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError(f"DAPO-style prompt content at index {index} must be a string.")
        contents.append(content)

    return "\n\n".join(contents)


def _assistant_generation_prefix() -> str:
    return f"{IM_START}assistant\n"


def _retool_stop_token_ids(tokenizer) -> list[int]:
    token_ids = []
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id != getattr(tokenizer, "unk_token_id", None):
        token_ids.append(im_end_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int) and eos_token_id >= 0:
        token_ids.append(eos_token_id)
    return list(dict.fromkeys(token_ids))


def _with_retool_stop(sampling_params: dict[str, Any], tokenizer) -> dict[str, Any]:
    params = dict(sampling_params)
    params["stop"] = list(RETOOL_STOP_STRINGS)
    params["stop_token_ids"] = _retool_stop_token_ids(tokenizer)
    params["no_stop_trim"] = True
    return params


def _routing_headers(args, sample) -> dict[str, str] | None:
    if getattr(sample, "session_id", None) and getattr(args, "router_policy", None) == "consistent_hashing":
        return {"X-SMG-Routing-Key": sample.session_id}
    return None


def _record_prefix_cache_info(sample: Sample, meta_info: dict[str, Any]) -> None:
    prefix_cache_info = getattr(sample, "prefix_cache_info", None)
    add = getattr(prefix_cache_info, "add", None)
    if callable(add):
        add(meta_info=meta_info)


def _get_max_new_tokens_per_turn() -> int | None:
    raw = os.getenv("RETOOL_MAX_NEW_TOKENS_PER_TURN", str(DEFAULT_MAX_NEW_TOKENS_PER_TURN)).strip()
    if not raw:
        return None
    value = int(raw)
    return value if value > 0 else None


def _get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _get_overlong_max_response_len(args) -> int | None:
    raw = os.getenv(OVERLONG_MAX_RESPONSE_LEN_ENV, "").strip()
    if raw:
        value = int(raw)
        return value if value > 0 else None

    if args is None:
        return None
    for attr in ("rollout_max_response_len", "eval_max_response_len"):
        value = getattr(args, attr, None)
        if value is not None:
            value = int(value)
            return value if value > 0 else None
    return None


def _ensure_sample_metadata(sample) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    return metadata


def _compute_overlong_penalty(args, sample) -> tuple[float, dict[str, int | float]]:
    if not _get_bool_env(OVERLONG_ENABLE_ENV, False):
        return 0.0, {}

    max_response_len = _get_overlong_max_response_len(args)
    buffer_len = _get_int_env(OVERLONG_BUFFER_LEN_ENV, DEFAULT_OVERLONG_BUFFER_LEN)
    penalty_factor = _get_float_env(OVERLONG_PENALTY_FACTOR_ENV, DEFAULT_OVERLONG_PENALTY_FACTOR)
    if max_response_len is None or buffer_len <= 0 or penalty_factor <= 0:
        return 0.0, {}

    response_len = int(getattr(sample, "response_length", 0) or 0)
    threshold = max(max_response_len - buffer_len, 0)
    overflow = max(response_len - threshold, 0)
    if overflow <= 0:
        penalty = 0.0
    else:
        penalty = -penalty_factor * min(overflow, buffer_len) / buffer_len

    return penalty, {
        "overlong_penalty": penalty,
        "overlong_response_len": response_len,
        "overlong_max_response_len": max_response_len,
        "overlong_buffer_len": buffer_len,
        "overlong_threshold": threshold,
    }


def _close_assistant_turn(assistant_text: str) -> str:
    """`<|im_end|>` is a stop token, so the assistant text may already end with it."""
    return "\n" if assistant_text.rstrip().endswith(IM_END) else f"{IM_END}\n"


def _strip_leading_system_turn(rendered: str) -> str:
    if not rendered.startswith(f"{IM_START}system\n"):
        return rendered
    marker = f"{IM_END}\n"
    system_end = rendered.find(marker)
    if system_end < 0:
        return rendered
    return rendered[system_end + len(marker) :]


def format_tool_response_observation(tokenizer, assistant_text: str, observation: str) -> str:
    """Render the tool response via the model chat template."""
    tool_turn = tokenizer.apply_chat_template(
        [{"role": "tool", "content": observation.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return _close_assistant_turn(assistant_text) + _strip_leading_system_turn(tool_turn)


def format_conversation_with_tools(
    tokenizer,
    prompt: str,
    tools: list[dict[str, Any]] = None,
    system_prompt: str = None,
) -> str:
    """Render the initial prompt via the tokenizer's chat template."""
    messages = [{"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}]
    if prompt:
        messages.append({"role": "user", "content": prompt})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools or None,
    )


def postprocess_predictions(prediction: str) -> tuple[str | None, str]:
    """Extract a code tool call from an assistant turn when present.

    Returns ("code", python_code) or (None, ""). Final answers are left to the
    reward function, which checks the last visible ``\\boxed{...}``.
    """
    tool_call_match = re.search(TOOL_CALL_PATTERN, prediction, re.DOTALL)
    if tool_call_match:
        try:
            tool_call = json.loads(tool_call_match.group(1))
        except json.JSONDecodeError:
            tool_call = {}
        if tool_call.get("name") == "code_interpreter":
            code = tool_call.get("arguments", {}).get("code", "")
            if code.strip():
                return "code", code

    return None, ""


async def execute_predictions(prediction: str, tokenizer=None) -> tuple[str, bool, bool]:
    """Run the action in an assistant turn.

    Returns (next_observation, done, tool_called).
    """
    action, content = postprocess_predictions(prediction)

    if action == "code":
        if tokenizer is None:
            raise ValueError("tokenizer is required to render tool observations")
        async with SEMAPHORE:
            result = await tool_registry.execute_tool("code_interpreter", {"code": content.strip()})
        return format_tool_response_observation(tokenizer, prediction, result), False, True

    return "", True, False


def _log_turn_debug(turn: int, available_tools: int, tools_used: int, payload_length: int) -> None:
    try:
        import wandb
    except ImportError:
        return
    setup_wandb_env_vars()
    if wandb.run is not None:
        wandb.log(
            {
                "debug/payload_length": payload_length,
                "debug/available_tools": available_tools,
                "debug/tools_used": tools_used,
                "debug/turn": turn,
            }
        )


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    """Custom generation function supporting tool calls"""
    assert not args.partial_rollout, "Partial rollout is not supported for " "this function at the moment."

    # Retried samples (previously aborted / partial) arrive here with stale
    # rollout state from the first attempt. Clear it so this generation starts
    # clean; otherwise the concatenation below appends new tokens to old ones
    # and downstream `slice_log_prob_with_cp` sees a length mismatch.
    sample.rollout_log_probs = None
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = None

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Render the initial prompt (system + tools + user) once, outside the loop.
    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(
        state.tokenizer,
        prompt=_extract_prompt_text(sample.prompt),
        tools=tool_specs,
    )

    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0

    if evaluation and args.eval_max_context_len is not None:
        max_context_length = args.eval_max_context_len
    elif args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu
    max_new_tokens_per_turn = _get_max_new_tokens_per_turn()

    for turn in range(TOOL_CONFIGS["max_turns"]):
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        # Clamp per-turn max_new_tokens to the remaining context budget so a
        # single turn cannot push total_length past max_context_length. Without
        # this, a turn can append up to rollout_max_response_len tokens on top
        # of a total that was just barely under the cap, producing samples
        # that exceed the training-side max_tokens_per_gpu * cp_size budget
        # and crash the partition/batch code (asserts or OOMs on an oversized
        # partition).
        remaining_budget = max_context_length - total_length
        per_turn_sampling_params = _with_retool_stop(dict(sampling_params), state.tokenizer)
        max_new_tokens = sampling_params.get("max_new_tokens", remaining_budget)
        if max_new_tokens_per_turn is not None:
            max_new_tokens = min(max_new_tokens, max_new_tokens_per_turn)
        per_turn_sampling_params["max_new_tokens"] = min(
            max_new_tokens,
            remaining_budget,
        )

        payload = {
            "input_ids": prompt_tokens_ids + response_token_ids,
            "sampling_params": per_turn_sampling_params,
            "return_logprob": True,  # per-token logprobs are needed for training
        }
        _log_turn_debug(turn, len(tool_specs), tool_call_count, len(prompt) + len(response))

        output = await post(url, payload, headers=_routing_headers(args, sample))
        meta_info = output["meta_info"]
        _record_prefix_cache_info(sample, meta_info)
        finish_reason = meta_info["finish_reason"]["type"]

        if finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        if "output_token_logprobs" not in meta_info:
            # sglang returned text but no output_token_logprobs — we cannot
            # recover per-token logprobs for this turn, which would desync
            # rollout_log_probs from response_token_ids and blow up
            # `slice_log_prob_with_cp` downstream. Abort the sample so the
            # fully_async rollout manager returns the whole group to the
            # buffer for retry instead of poisoning the trainer.
            sample.status = Sample.Status.ABORTED
            return sample

        token_logprobs = meta_info["output_token_logprobs"]
        cur_response_token_ids = [item[1] for item in token_logprobs]
        cur_response = state.tokenizer.decode(cur_response_token_ids)
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += [item[0] for item in token_logprobs]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        if finish_reason == "length":
            break

        next_obs, done, tool_called = await execute_predictions(cur_response, state.tokenizer)
        if done:
            break
        if tool_called:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)
        # Observation tokens are masked out (loss_mask=0); their logprobs are
        # placeholders that keep the logprob array aligned with the tokens.
        sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)
        assert len(response_token_ids) == len(sample.rollout_log_probs), (
            f"Token/logp length mismatch at turn {turn}: "
            f"{len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"
        )

        # Tool output is appended verbatim and can push total_length past
        # max_context_length (the per-turn generation was clamped to the
        # remaining budget, but tool output is unconstrained). Trim tail
        # tokens so the final sample fits the training budget exactly.
        overflow = len(prompt_tokens_ids) + len(response_token_ids) - max_context_length
        if overflow > 0:
            response_token_ids = response_token_ids[:-overflow]
            loss_masks = loss_masks[:-overflow]
            sample.rollout_log_probs = sample.rollout_log_probs[:-overflow]
            # Resync the text field from the trimmed token list so
            # reward_func's `sample.prompt + sample.response` matches what
            # the model was actually trained on. decode(tokenize(text)) can
            # be lossy on some tokenizers (whitespace / special-token
            # collapse), but reward_func's regex is whitespace-robust and
            # the trainer sees tokens, not text — so the drift is safe.
            response = state.tokenizer.decode(response_token_ids)
            sample.status = Sample.Status.TRUNCATED
            break

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            break

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks
    sample.tool_call_count = tool_call_count
    sample.num_turns = 2 * tool_call_count + 2
    sample.metadata = sample.metadata or {}
    sample.metadata["round_number"] = tool_call_count + 1
    sample.metadata["tool_call_count"] = tool_call_count
    sample.metadata["num_turns"] = sample.num_turns

    # Payload info for wandb logging.
    payload_text = prompt + response
    sample.payload_text = payload_text
    sample.payload_has_system = "<|im_start|>system" in payload_text
    sample.payload_has_tools = "# Tools" in payload_text

    if finish_reason == "length":
        sample.status = Sample.Status.TRUNCATED
    elif finish_reason == "abort":
        sample.status = Sample.Status.ABORTED
    elif finish_reason == "stop":
        sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """Score a sample with math_dapo, then apply ReTool-specific shaping."""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    solution_str = getattr(sample, "payload_text", None) or (_extract_prompt_text(sample.prompt) + sample.response)
    ground_truth = sample.label if sample.label is not None else ""
    tool_call_count = getattr(sample, "tool_call_count", 0)
    num_turns = getattr(sample, "num_turns", 2 * tool_call_count + 2)

    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)
    base_score = float(result["score"])

    metadata = _ensure_sample_metadata(sample)
    metadata["raw_reward"] = base_score

    # Encourage the model to call tools: on wrong answers, partially offset the
    # penalty based on how many tool calls were made.
    if base_score < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, base_score + tool_call_reward)

    overlong_penalty, overlong_info = _compute_overlong_penalty(args, sample)
    if overlong_info:
        metadata.update(overlong_info)
        result["overlong_penalty"] = overlong_penalty
    result["score"] += overlong_penalty

    if result.get("pred") is None:
        result["pred"] = ""

    return result
