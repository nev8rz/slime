#!/usr/bin/env python3
"""Prepare DAPO train and AIME eval JSONL files for the ReTool RL recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, load_dataset

BOXED_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: \\boxed{{$Answer}} where $Answer is the answer "
    "to the problem.\n\n{problem}\n\nRemember to put your answer on its own line "
    'after "Answer:".'
)


def load_source(source: str, split: str) -> Dataset:
    path = Path(source).expanduser()
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(path), split="train")
        raise ValueError(f"Unsupported local dataset file type: {path}")
    return load_dataset(source, split=split)


def first_user_content(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role", "user") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
    return ""


def extract_label(example: dict[str, Any]) -> str:
    reward_model = example.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    for key in ("label", "answer", "ground_truth"):
        if example.get(key) is not None:
            return str(example[key])
    raise KeyError("Could not find label/answer/ground_truth in example.")


def extract_problem(example: dict[str, Any]) -> str:
    extra_info = example.get("extra_info")
    if isinstance(extra_info, dict) and isinstance(extra_info.get("raw_problem"), str):
        return extra_info["raw_problem"]
    return first_user_content(example.get("prompt"))


def boxed_prompt(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": BOXED_PROMPT_TEMPLATE.format(problem=problem.strip())}]


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path, overwrite: bool) -> int:
    if path.exists() and not overwrite:
        print(f"skip existing: {path}")
        return sum(1 for _ in path.open())

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} rows: {path}")
    return count


def convert_dapo_train(source: str, split: str) -> list[dict[str, Any]]:
    dataset = load_source(source, split)
    rows = []
    for example in dataset:
        prompt = example.get("prompt")
        if not isinstance(prompt, list):
            prompt = [{"role": "user", "content": first_user_content(prompt)}]
        rows.append({"prompt": prompt, "label": extract_label(example)})
    return rows


def _dedup_key(example: dict[str, Any]) -> Any:
    extra_info = example.get("extra_info")
    if isinstance(extra_info, dict):
        if extra_info.get("raw_problem") is not None:
            return extra_info["raw_problem"]
        if extra_info.get("index") is not None:
            return extra_info["index"]
    return extract_problem(example)


def _sort_key(example: dict[str, Any]) -> tuple[int, str]:
    extra_info = example.get("extra_info")
    if isinstance(extra_info, dict):
        index = extra_info.get("index")
        if isinstance(index, int):
            return index, extract_problem(example)
        if isinstance(index, str) and index.isdigit():
            return int(index), extract_problem(example)
    return 10**9, extract_problem(example)


def convert_aime_eval(source: str, split: str) -> list[dict[str, Any]]:
    dataset = load_source(source, split)
    selected: dict[Any, dict[str, Any]] = {}
    for example in dataset:
        key = _dedup_key(example)
        selected.setdefault(key, example)

    rows = []
    for example in sorted(selected.values(), key=_sort_key):
        problem = extract_problem(example)
        rows.append({"prompt": boxed_prompt(problem), "label": extract_label(example)})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/retool_post_train_all")
    parser.add_argument("--dapo-source", default="BytedTsinghua-SIA/DAPO-Math-17k")
    parser.add_argument("--dapo-split", default="train")
    parser.add_argument("--aime2024-source", default="BytedTsinghua-SIA/AIME-2024")
    parser.add_argument("--aime2024-split", default="train")
    parser.add_argument("--aime2025-source", default="zhuzilin/aime-2025")
    parser.add_argument("--aime2025-split", default="train")
    parser.add_argument("--skip-dapo", action="store_true")
    parser.add_argument("--skip-aime2024", action="store_true")
    parser.add_argument("--skip-aime2025", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()

    if not args.skip_dapo:
        write_jsonl(
            convert_dapo_train(args.dapo_source, args.dapo_split),
            data_root / "dapo-math-17k" / "dapo-math-17k.jsonl",
            args.overwrite,
        )
    if not args.skip_aime2024:
        write_jsonl(
            convert_aime_eval(args.aime2024_source, args.aime2024_split),
            data_root / "aime-2024" / "aime-2024.jsonl",
            args.overwrite,
        )
    if not args.skip_aime2025:
        write_jsonl(
            convert_aime_eval(args.aime2025_source, args.aime2025_split),
            data_root / "aime-2025" / "aime-2025.jsonl",
            args.overwrite,
        )


if __name__ == "__main__":
    main()
