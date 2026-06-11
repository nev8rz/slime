import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

RETOOL_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "retool_post_train_all"
sys.path.insert(0, str(RETOOL_EXAMPLE_DIR))

_MISSING = object()
_STUBBED_MODULES = [
    "slime.rollout.sglang_rollout",
    "slime.utils.http_utils",
    "slime.utils.types",
    "slime.rollout.rm_hub.math_dapo_utils",
]
_ORIGINAL_MODULES = {name: sys.modules.get(name, _MISSING) for name in _STUBBED_MODULES}

sglang_rollout = types.ModuleType("slime.rollout.sglang_rollout")
sglang_rollout.GenerateState = object
sys.modules["slime.rollout.sglang_rollout"] = sglang_rollout

http_utils = types.ModuleType("slime.utils.http_utils")
http_utils.post = None
sys.modules["slime.utils.http_utils"] = http_utils

types_module = types.ModuleType("slime.utils.types")
types_module.Sample = object
sys.modules["slime.utils.types"] = types_module

math_dapo_utils = types.ModuleType("slime.rollout.rm_hub.math_dapo_utils")
math_dapo_utils.compute_score = lambda *args, **kwargs: {"score": 0, "pred": ""}
sys.modules["slime.rollout.rm_hub.math_dapo_utils"] = math_dapo_utils

import generate_with_retool as retool_module  # noqa: E402
from generate_with_retool import (  # noqa: E402
    _extract_prompt_text,
    _record_prefix_cache_info,
    _routing_headers,
    execute_predictions,
    format_conversation_with_tools,
    format_tool_response_observation,
    generate,
    postprocess_predictions,
    reward_func,
)

for name, module in _ORIGINAL_MODULES.items():
    if module is _MISSING:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


def test_postprocess_predictions_accepts_native_tool_call():
    payload = {
        "name": "code_interpreter",
        "arguments": {"code": "print(1 + 2)\nprint(3 + 4)"},
    }
    prediction = f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"

    assert postprocess_predictions(prediction) == ("code", "print(1 + 2)\nprint(3 + 4)")


def test_postprocess_predictions_accepts_multiline_code_string_with_raw_newlines():
    prediction = """<tool_call>
{"name": "code_interpreter", "arguments": {"code": "x = 1
y = 2
print(x + y)"}}
</tool_call>"""

    assert postprocess_predictions(prediction) == ("code", "x = 1\ny = 2\nprint(x + y)")


def test_postprocess_predictions_rejects_malformed_tool_call():
    prediction = """Let's solve it.
<tool_call>
{{"name": "code_interpreter", "arguments": {"code": "bad = True"}}
}}
<tool_call>
json
{"name": "code_interpreter", "arguments": {"code": "print(1 + 2)"}}
</tool_call>"""

    assert postprocess_predictions(prediction) == (None, "")


def test_postprocess_predictions_rejects_json_language_marker():
    prediction = """<tool_call>
json
{"name": "code_interpreter", "arguments": {"code": "print(1 + 2)"}}
</tool_call>"""

    assert postprocess_predictions(prediction) == (None, "")


def test_postprocess_predictions_rejects_legacy_code_formats():
    assert postprocess_predictions("<code>\n```python\nprint(1 + 2)\n```\n</code>") == (None, "")
    assert postprocess_predictions("```python\nprint(1 + 2)\n```") == (None, "")


def test_postprocess_predictions_does_not_parse_final_answers():
    assert postprocess_predictions("Answer: \\boxed{42}") == (None, "")
    assert postprocess_predictions("The final answer is \\boxed{42}.") == (None, "")


def test_execute_predictions_terminates_when_no_tool_call():
    assert asyncio.run(execute_predictions("The final answer is \\boxed{42}.")) == ("", True, False)


def test_reward_func_uses_trajectory_num_turns_for_tool_bonus(monkeypatch):
    def fake_score(*args, **kwargs):
        return {"score": -1.0, "pred": None, "acc": False}

    class DummySample:
        prompt = ""
        response = ""
        label = "42"
        payload_text = "The final answer is \\boxed{41}"
        tool_call_count = 1

    monkeypatch.setattr(retool_module, "math_dapo_compute_score", fake_score)
    monkeypatch.setenv("RETOOL_OVERLONG_PENALTY_ENABLE", "0")

    result = asyncio.run(reward_func(None, DummySample()))

    assert result == {"score": -0.9, "pred": "", "acc": False}


def test_reward_func_applies_dapo_style_overlong_penalty(monkeypatch):
    def fake_score(*args, **kwargs):
        return {"score": 1.0, "pred": "42", "acc": True}

    class Args:
        rollout_max_response_len = 100

    class DummySample:
        def __init__(self):
            self.prompt = ""
            self.response = ""
            self.label = "42"
            self.payload_text = "The final answer is \\boxed{42}"
            self.tool_call_count = 0
            self.response_length = 90
            self.metadata = {}

    monkeypatch.setattr(retool_module, "math_dapo_compute_score", fake_score)
    monkeypatch.setenv("RETOOL_OVERLONG_PENALTY_ENABLE", "1")
    monkeypatch.delenv("RETOOL_OVERLONG_MAX_RESPONSE_LEN", raising=False)
    monkeypatch.setenv("RETOOL_OVERLONG_BUFFER_LEN", "20")
    monkeypatch.setenv("RETOOL_OVERLONG_PENALTY_FACTOR", "1.0")

    sample = DummySample()
    result = asyncio.run(reward_func(Args(), sample))

    assert result["score"] == pytest.approx(0.5)
    assert result["overlong_penalty"] == pytest.approx(-0.5)
    assert sample.metadata["raw_reward"] == 1.0
    assert sample.metadata["overlong_threshold"] == 80


def test_tool_response_observation_uses_qwen_tool_response_turn():
    class ToolResponseTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "tool", "content": "1"}]
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return (
                "<|im_start|>system\nsys<|im_end|>\n"
                "<|im_start|>user\n"
                "<tool_response>\n"
                "1\n"
                "</tool_response><|im_end|>\n"
                "<|im_start|>assistant\n"
            )

    assistant_text = '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(1)"}}\n</tool_call>'

    observation = format_tool_response_observation(ToolResponseTokenizer(), assistant_text, "1")

    assert observation == (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<tool_response>\n"
        "1\n"
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert "<think>" not in observation
    assert "<interpreter>" not in observation
    assert "<code>" not in observation


class DummyTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"


def test_format_conversation_does_not_prefill_think():
    tokenizer = DummyTokenizer()

    rendered = format_conversation_with_tools(tokenizer, "q", tools=[])

    assert rendered.endswith("<|im_start|>assistant\n")
    assert "<think>" not in rendered
    assert "enable_thinking" not in tokenizer.kwargs


def test_extract_prompt_text_accepts_string_prompt():
    assert _extract_prompt_text("Solve 1+2") == "Solve 1+2"


def test_extract_prompt_text_accepts_dapo_roleless_prompt_list():
    assert _extract_prompt_text([{"content": "Solve 1+2"}]) == "Solve 1+2"


def test_extract_prompt_text_accepts_aime_user_prompt_list():
    assert _extract_prompt_text([{"role": "user", "content": "Solve 1+2"}]) == "Solve 1+2"


def test_extract_prompt_text_rejects_non_dict_prompt_list_items():
    try:
        _extract_prompt_text([{"content": "q"}, "bad"])
    except TypeError as exc:
        assert "only dict items" in str(exc)
    else:
        raise AssertionError("Expected TypeError")


def test_extract_prompt_text_rejects_non_user_roles():
    try:
        _extract_prompt_text([{"role": "assistant", "content": "hello"}])
    except ValueError as exc:
        assert "Unsupported DAPO-style prompt role" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_extract_prompt_text_rejects_non_string_content():
    try:
        _extract_prompt_text([{"content": ["not", "text"]}])
    except TypeError as exc:
        assert "must be a string" in str(exc)
    else:
        raise AssertionError("Expected TypeError")


def test_routing_headers_use_session_id_for_consistent_hashing():
    args = types.SimpleNamespace(router_policy="consistent_hashing")
    sample = types.SimpleNamespace(session_id="sample-123")

    assert _routing_headers(args, sample) == {"X-SMG-Routing-Key": "sample-123"}
    assert _routing_headers(types.SimpleNamespace(router_policy="cache_aware"), sample) is None
    assert _routing_headers(args, types.SimpleNamespace(session_id=None)) is None


def test_record_prefix_cache_info_accumulates_sglang_meta_info():
    class PrefixCacheInfo:
        def __init__(self):
            self.cached_tokens = 0
            self.total_prompt_tokens = 0

        def add(self, meta_info):
            self.cached_tokens += meta_info.get("cached_tokens", 0)
            self.total_prompt_tokens += meta_info.get("prompt_tokens", 0)

    sample = types.SimpleNamespace(prefix_cache_info=PrefixCacheInfo())

    _record_prefix_cache_info(sample, {"cached_tokens": 1223, "prompt_tokens": 1600})
    _record_prefix_cache_info(sample, {"cached_tokens": 777, "prompt_tokens": 1400})

    assert sample.prefix_cache_info.cached_tokens == 2000
    assert sample.prefix_cache_info.total_prompt_tokens == 3000


def test_generate_passes_session_affinity_header_to_sglang(monkeypatch):
    captured = {}

    class DummyStatus:
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        COMPLETED = "completed"

    class DummySampleClass:
        Status = DummyStatus

    class DummyTokenizer:
        eos_token_id = 2
        unk_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["add_generation_prompt"] is True
            return "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2, 3]}

        def convert_tokens_to_ids(self, token):
            return 4 if token == retool_module.IM_END else self.unk_token_id

        def decode(self, token_ids):
            return "The final answer is \\boxed{42}."

    class DummyState:
        def __init__(self, args):
            self.tokenizer = DummyTokenizer()

    async def fake_post(url, payload, max_retries=60, headers=None):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [(0.0, 101), (0.0, 102)],
            }
        }

    args = types.SimpleNamespace(
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="consistent_hashing",
        eval_max_context_len=None,
        rollout_max_context_len=32768,
        context_parallel_size=2,
        max_tokens_per_gpu=16384,
    )
    sample = types.SimpleNamespace(
        prompt="q",
        response="stale",
        response_length=10,
        rollout_log_probs=[-1.0],
        loss_mask=[1],
        session_id="sample-session",
        metadata={},
    )

    monkeypatch.setattr(retool_module, "Sample", DummySampleClass)
    monkeypatch.setattr(retool_module, "GenerateState", DummyState)
    monkeypatch.setattr(retool_module, "post", fake_post)
    monkeypatch.setattr(retool_module, "_log_turn_debug", lambda *args, **kwargs: None)
    monkeypatch.setattr(retool_module.tool_registry, "get_tool_specs", lambda: [])
    monkeypatch.setitem(retool_module.TOOL_CONFIGS, "max_turns", 1)

    result = asyncio.run(generate(args, sample, {"max_new_tokens": 16}))

    assert result.status == DummyStatus.COMPLETED
    assert captured["headers"] == {"X-SMG-Routing-Key": "sample-session"}
    assert captured["url"] == "http://127.0.0.1:30000/generate"
