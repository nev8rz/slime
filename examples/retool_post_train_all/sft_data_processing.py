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
PYTHON_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)
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
    marker_index = content.lower().find(USER_QUESTION_MARKER)
    if marker_index >= 0:
        question = content[marker_index + len(USER_QUESTION_MARKER) :].strip()
        return f"Solve the following problem step by step.\n\n{question}"

    content = re.sub(
        r"The Python code should be complete scripts.*?<\/code>`\.\s*",
        "",
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r"You now have the ability to selectively write executable Python code.*?arrive at the final answer\.\s*",
        "",
        content,
        flags=re.DOTALL,
    )
    return content.strip()


def build_tool_call(code: str, call_id: str):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "arguments": {"code": code.strip()},
        },
    }


def build_assistant_message(content: str = "", reasoning_content: str = "", tool_calls: list[dict] | None = None):
    # Qwen2.5-Instruct has no thinking mode, so its chat template ignores any
    # `reasoning_content` field. Fold the reasoning text into `content` so the
    # model is trained on it as part of the visible response.
    reasoning_content = reasoning_content.strip()
    content = content.strip()
    if reasoning_content and content:
        content = f"{reasoning_content}\n\n{content}"
    elif reasoning_content:
        content = reasoning_content

    message = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def normalize_answer_content(answer: str) -> str:
    answer = answer.strip()
    if answer.startswith("Answer:"):
        return answer
    return f"Answer: {answer}"


def extract_code_message(content: str, call_id: str) -> tuple[dict | None, str]:
    code_match = CODE_TAG_RE.search(content)
    if code_match is None:
        return None, content

    code = code_match.group(1).strip()
    python_match = PYTHON_CODE_RE.search(code)
    if python_match is not None:
        code = python_match.group(1).strip()

    message = build_assistant_message(
        reasoning_content=content[: code_match.start()],
        tool_calls=[build_tool_call(code, call_id)],
    )
    return message, content[code_match.end() :]


def extract_interpreter_message(content: str, call_id: str) -> tuple[dict | None, str]:
    interpreter_match = INTERPRETER_TAG_RE.search(content)
    if interpreter_match is None:
        return None, content

    message = {
        "role": "tool",
        "name": "code_interpreter",
        "tool_call_id": call_id,
        "content": interpreter_match.group(1).strip(),
    }
    return message, content[interpreter_match.end() :]


def extract_answer_message(content: str) -> tuple[dict | None, str]:
    answer_match = ANSWER_TAG_RE.search(content)
    if answer_match is None:
        return None, content

    message = build_assistant_message(
        content=normalize_answer_content(answer_match.group(1)),
        reasoning_content=content[: answer_match.start()],
    )
    return message, content[answer_match.end() :]


def convert_assistant_turn(content: str) -> list[dict]:
    messages = []
    call_index = 0
    role = "assistant"

    while content.strip():
        if role == "assistant":
            call_id = f"call_{call_index}"
            message, content = extract_code_message(content, call_id)
            if message is None:
                message, content = extract_answer_message(content)
            if message is None:
                messages.append(build_assistant_message(content=content))
                break

            messages.append(message)
            if message.get("tool_calls"):
                role = "tool"
            else:
                role = "assistant"
        else:
            call_id = f"call_{call_index}"
            message, content = extract_interpreter_message(content, call_id)
            if message is None:
                raise ValueError(f"Missing interpreter output after {call_id}")

            messages.append(message)
            call_index += 1
            role = "assistant"

    if not messages:
        messages.append(build_assistant_message(content=content))

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
