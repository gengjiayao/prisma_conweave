import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "path_policy_search", Path(__file__).resolve().parents[1] / "source/path_policy_search.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_neutral_has_no_queue_preference():
    rng = np.random.default_rng(91)
    x = rng.uniform(0, 10, (100, 24))
    w, b, v, c = p.parameters(p.neutral(), np.ones(8))
    assert np.array_equal(np.maximum(x @ w + b, 0) @ v + c, np.zeros((100, 2)))


def test_native_export_preserves_signed_scores(tmp_path):
    rng = np.random.default_rng(92)
    x = rng.uniform(0, 10, (100, 24))
    x[:, [5, 6]] = rng.normal(0, 10, (100, 2))
    theta, scale = rng.normal(size=8), rng.uniform(.01, 5, 8)
    path = Path(p.export(theta, scale, tmp_path / "actor.txt"))
    a = np.fromstring("\n".join(path.read_text().splitlines()[1:]), sep=" ")
    w, b, v, c = a[:768].reshape(24, 32), a[768:800], a[800:864].reshape(32, 2), a[864:]
    q = np.maximum(x @ w + b, 0) @ v + c
    assert np.allclose(q[:, 1] - q[:, 0], (x[:, p.FEATURES] / scale) @ theta)


def test_reward_association_changes_update_direction():
    directions = np.eye(8)
    returns = np.zeros(16)
    returns[0:2] = [-10, 10]
    correct, _ = p.update(p.neutral(), directions, returns, np.arange(16))
    reversed_labels, _ = p.update(p.neutral(), directions, returns, np.arange(16).reshape(8, 2)[:, ::-1].ravel())
    assert correct[0] > 0 > reversed_labels[0]
    assert np.all(correct[1:] == 0)


def test_partial_completion_is_not_a_good_reward():
    row = dict(valid=True, completed=9, offered=10, all_mean_us=1, all_p99_us=1)
    try:
        p.reward([row])
    except ValueError:
        return
    raise AssertionError("An unfinished expensive flow must not disappear from the reward")


def test_censoring_penalty_requires_an_explicit_failure_class():
    row = dict(valid=False, completed=9, offered=10, training_failure="unfinished_at_horizon")
    assert p.reward([row], failure_penalty=-10000) == -10000
    row["training_failure"] = "simulator_crash"
    try:
        p.reward([row], failure_penalty=-10000)
    except ValueError:
        return
    raise AssertionError("Infrastructure errors must not become training rewards")
