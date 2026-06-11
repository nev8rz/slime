from examples.retool_post_train_all.sft_data_processing import convert, convert_assistant_turn


def test_convert_assistant_turn_matches_retool_messages():
    messages = convert_assistant_turn(
        """
We need compute this carefully.

<code>
```python
print(1 + 2)
```
</code>

<interpreter>
3
</interpreter>

The tool confirms the sum.

<answer>
\\boxed{3}
</answer>
"""
    )

    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == "We need compute this carefully."
    assert "reasoning_content" not in messages[0]
    assert messages[0]["tool_calls"] == [
        {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "arguments": {"code": "print(1 + 2)"},
            },
        }
    ]

    assert messages[1] == {
        "role": "tool",
        "content": "3",
    }

    assert messages[2]["role"] == "assistant"
    assert "reasoning_content" not in messages[2]
    assert messages[2]["content"] == "The tool confirms the sum.\n\n\n\\boxed{3}"


def test_convert_user_prompt_matches_retool_preprocess():
    row = {
        "messages": [
            {
                "role": "user",
                "content": "header\n*user question:* What is 1+2? <answer>ignore</answer>",
            },
            {
                "role": "assistant",
                "content": "<answer>\\boxed{3}</answer>",
            },
        ]
    }

    converted = convert(row)

    assert converted["messages"][0] == {"role": "user", "content": "What is 1+2? ignore"}
    assert converted["messages"][1] == {"role": "assistant", "content": "\\boxed{3}"}
