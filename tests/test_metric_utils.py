import pytest

from slime.utils.metric_utils import compute_eval_pass_rate, compute_pass_rate

NUM_GPUS = 0


def test_compute_pass_rate_is_at_least_one_correct_at_k():
    rewards = [1, 0, 0, 0]

    metrics = compute_pass_rate(rewards, group_size=4)

    assert metrics["pass@1"] == pytest.approx(0.25)
    assert metrics["pass@2"] == pytest.approx(0.5)
    assert metrics["pass@4"] == pytest.approx(1.0)


def test_compute_eval_pass_rate_logs_pass_at_k_as_at_least_one_correct():
    rewards = [1, 0, 0, 0]

    metrics = compute_eval_pass_rate(rewards, group_size=4)

    assert metrics["pass@1"] == pytest.approx(0.25)
    assert metrics["pass@2"] == pytest.approx(0.5)
    assert metrics["pass@4"] == pytest.approx(1.0)
    assert "any@1" not in metrics
    assert "any@2" not in metrics
    assert "any@4" not in metrics
    assert metrics["pass^1"] == pytest.approx(0.25)
    assert metrics["pass^2"] == pytest.approx(0.0)
    assert metrics["pass^4"] == pytest.approx(0.0)
    assert "mean@1" not in metrics
    assert "mean@2" not in metrics
    assert metrics["mean@4"] == pytest.approx(0.25)


def test_compute_eval_pass_rate_pass_power_at_group_size_requires_all_correct():
    rewards = [1, 1, 1, 1, 1, 1, 1, 0]

    metrics = compute_eval_pass_rate(rewards, group_size=4)

    assert metrics["pass^4"] == pytest.approx(0.5)
    assert metrics["pass@4"] == pytest.approx(1.0)
    assert "any@4" not in metrics
    assert metrics["mean@4"] == pytest.approx(0.875)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
