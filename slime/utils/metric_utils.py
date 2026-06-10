from __future__ import annotations

import logging
import math
from typing import Any, Literal, MutableMapping

import numpy as np

logger = logging.getLogger(__name__)


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def update_rollout_eta_metrics(
    *,
    rollout_id: int,
    num_rollout: int | None,
    now: float,
    state: MutableMapping[str, float],
) -> dict[str, float]:
    if num_rollout is None or num_rollout <= 0:
        return {}

    completed = min(max(rollout_id + 1, 0), num_rollout)
    remaining = max(num_rollout - completed, 0)
    metrics = {
        "completed_rollouts": float(completed),
        "remaining_rollouts": float(remaining),
        "progress": completed / num_rollout,
    }

    start_time = state.setdefault("start_time", now)
    last_time = state.get("last_time")
    last_rollout_id = state.get("last_rollout_id")

    if last_time is not None and last_rollout_id is not None:
        delta_rollouts = rollout_id - int(last_rollout_id)
        delta_seconds = now - last_time
        if delta_rollouts > 0 and delta_seconds > 0:
            seconds_per_rollout = delta_seconds / delta_rollouts
            state["seconds_per_rollout"] = seconds_per_rollout

    state["last_time"] = now
    state["last_rollout_id"] = float(rollout_id)

    elapsed_seconds = max(now - start_time, 0.0)
    metrics["elapsed_seconds"] = elapsed_seconds
    metrics["elapsed_hours"] = elapsed_seconds / 3600

    seconds_per_rollout = state.get("seconds_per_rollout")
    if seconds_per_rollout is not None and seconds_per_rollout > 0:
        eta_seconds = remaining * seconds_per_rollout
        metrics["seconds_per_rollout"] = seconds_per_rollout
        metrics["rollouts_per_hour"] = 3600 / seconds_per_rollout
        metrics["eta_seconds"] = eta_seconds
        metrics["eta_hours"] = eta_seconds / 3600

    return metrics


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    if group_size == 1:
        return {}

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)

        pass_k = np.mean(pass_k_estimates)
        log_dict[f"pass@{k}"] = pass_k

    return log_dict


def compute_eval_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    if group_size == 1:
        return {}

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        all_k_estimates = _estimate_all_at_k(num_samples, num_correct, k)
        any_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)
        mean_k_estimates = num_correct / num_samples

        log_dict[f"pass@{k}"] = np.mean(all_k_estimates).item()
        log_dict[f"any@{k}"] = np.mean(any_k_estimates).item()
        log_dict[f"mean@{k}"] = np.mean(mean_k_estimates).item()

    return log_dict


def _estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct)])


def _estimate_all_at_k(num_samples, num_correct, k):
    """
    Estimates the probability that all k sampled responses are correct.
    """

    def estimator(n, c, k):
        """
        Calculates comb(c, k) / comb(n, k).
        """
        if c < k:
            return 0.0
        return np.prod((c - np.arange(k)) / (n - np.arange(k))).item()

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct)])


def compute_statistics(values: list[float]) -> dict[str, float]:
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str):
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    else:
        return False


def compute_rollout_step(args, rollout_id):
    if args.wandb_always_use_train_step:
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id
