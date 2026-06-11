import os
import re
from pathlib import Path

import pandas as pd

CODE_INTERPRETER_TOOL = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": "A tool for executing code.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to execute.",
                }
            },
            "required": ["code"],
        },
    },
}

CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
PYTHON_CODE_RE = re.compile(r"```python(.*?)```", re.DOTALL)
INTERPRETER_TAG_RE = re.compile(r"<interpreter>(.*?)</interpreter>", re.DOTALL)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
USER_QUESTION_MARKER = "*user question:*"


def load_retool_sft_dataset():
    source = os.getenv("RETOOL_SFT_SOURCE", "JoeYing/ReTool-SFT")
    source_path = Path(source)
    if source_path.exists():
        if source_path.is_file():
            return pd.read_parquet(source_path)

        parquet_files = sorted(source_path.glob("*.parquet"))
        if parquet_files:
            return pd.concat((pd.read_parquet(path) for path in parquet_files), ignore_index=True)

        from datasets import load_dataset

        return load_dataset(str(source_path))["train"].to_pandas()

    from datasets import load_dataset

    return load_dataset(source)["train"].to_pandas()


def normalize_user_content(content: str) -> str:
    marker_index = content.find(USER_QUESTION_MARKER)
    assert marker_index != -1
    return (
        content[marker_index + len(USER_QUESTION_MARKER) :]
        .replace("<answer>", "")
        .replace("</answer>", "")
        .strip()
    )


def build_tool_call(code: str):
    return {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "arguments": {"code": code},
        },
    }


def build_assistant_message(content: str = "", tool_calls: list[dict] | None = None):
    message = {
        "role": "assistant",
        "content": content.strip(),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def extract_code_message(content: str) -> tuple[dict | None, str]:
    code_match = CODE_TAG_RE.search(content)
    if code_match is None:
        return None, content

    code = code_match.group(1).strip()
    python_match = PYTHON_CODE_RE.search(code)
    if python_match is not None:
        code = python_match.group(1).strip()

    message = build_assistant_message(
        content=content[: code_match.start()],
        tool_calls=[build_tool_call(code)],
    )
    return message, content[code_match.end() :]


def extract_interpreter_message(content: str) -> tuple[dict | None, str]:
    interpreter_match = INTERPRETER_TAG_RE.search(content)
    if interpreter_match is None:
        return None, content

    message = {
        "role": "tool",
        "content": interpreter_match.group(1).strip(),
    }
    return message, content[interpreter_match.end() :]


def extract_answer_message(content: str) -> tuple[dict | None, str]:
    answer_match = ANSWER_TAG_RE.search(content)
    if answer_match is None:
        return None, content

    answer = content[: answer_match.start()] + answer_match.group(1)
    message = build_assistant_message(content=answer)
    return message, content[answer_match.end() :]


def convert_assistant_turn(content: str) -> list[dict]:
    messages = []
    role = "assistant"

    while content.strip():
        if role == "assistant":
            message, content = extract_code_message(content)
            if message is None:
                message, content = extract_answer_message(content)
            assert message is not None

            messages.append(message)
            role = "tool"
        else:
            message, content = extract_interpreter_message(content)
            assert message is not None
            messages.append(message)
            role = "assistant"

    return messages


def convert(sample):
    output_messages = []

    for turn in sample["messages"]:
        role = turn["role"]
        content = turn["content"]

        if role == "user":
            output_messages.append({"role": "user", "content": normalize_user_content(content)})
        elif role == "assistant":
            output_messages.extend(convert_assistant_turn(content))
        elif role == "system":
            output_messages.append({"role": "system", "content": content})
        else:
            raise ValueError(f"Unknown role: {role}")

    return {
        "messages": output_messages,
        "tools": [CODE_INTERPRETER_TOOL],
    }


def main():
    ds = load_retool_sft_dataset()
    rows = [convert(sample) for sample in ds.to_dict(orient="records")]
    output_df = pd.DataFrame(rows)

    output_path = Path(os.getenv("RETOOL_SFT_OUTPUT", "./data/retool/ReTool-SFT.parquet"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    print(f"Wrote {len(output_df)} samples to {output_path}")


if __name__ == "__main__":
    main()
