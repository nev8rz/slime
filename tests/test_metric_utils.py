import pytest

from slime.utils.metric_utils import compute_eval_pass_rate, compute_pass_rate


def test_compute_pass_rate_is_any_correct_at_k():
    rewards = [1, 0, 0, 0]

    metrics = compute_pass_rate(rewards, group_size=4)

    assert metrics["pass@1"] == pytest.approx(0.25)
    assert metrics["pass@2"] == pytest.approx(0.5)
    assert metrics["pass@4"] == pytest.approx(1.0)


def test_compute_eval_pass_rate_logs_all_correct_and_any_correct():
    rewards = [1, 0, 0, 0]

    metrics = compute_eval_pass_rate(rewards, group_size=4)

    assert metrics["pass@1"] == pytest.approx(0.25)
    assert metrics["pass@2"] == pytest.approx(0.0)
    assert metrics["pass@4"] == pytest.approx(0.0)
    assert metrics["any@1"] == pytest.approx(0.25)
    assert metrics["any@2"] == pytest.approx(0.5)
    assert metrics["any@4"] == pytest.approx(1.0)
    assert metrics["mean@1"] == pytest.approx(0.25)
    assert metrics["mean@2"] == pytest.approx(0.25)
    assert metrics["mean@4"] == pytest.approx(0.25)


def test_compute_eval_pass_rate_pass_at_group_size_requires_all_correct():
    rewards = [1, 1, 1, 1, 1, 1, 1, 0]

    metrics = compute_eval_pass_rate(rewards, group_size=4)

    assert metrics["pass@4"] == pytest.approx(0.5)
    assert metrics["any@4"] == pytest.approx(1.0)
    assert metrics["mean@4"] == pytest.approx(0.875)
