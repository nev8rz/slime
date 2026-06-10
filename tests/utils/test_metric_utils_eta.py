from slime.utils.metric_utils import update_rollout_eta_metrics


def test_rollout_eta_metrics_wait_for_observed_interval():
    state = {}

    first = update_rollout_eta_metrics(rollout_id=0, num_rollout=10, now=100.0, state=state)

    assert first["completed_rollouts"] == 1.0
    assert first["remaining_rollouts"] == 9.0
    assert first["progress"] == 0.1
    assert "eta_seconds" not in first

    second = update_rollout_eta_metrics(rollout_id=1, num_rollout=10, now=160.0, state=state)

    assert second["completed_rollouts"] == 2.0
    assert second["remaining_rollouts"] == 8.0
    assert second["seconds_per_rollout"] == 60.0
    assert second["eta_seconds"] == 480.0
    assert second["eta_hours"] == 480.0 / 3600
    assert second["rollouts_per_hour"] == 60.0


def test_rollout_eta_metrics_handles_missing_total():
    metrics = update_rollout_eta_metrics(rollout_id=0, num_rollout=None, now=100.0, state={})

    assert metrics == {}
