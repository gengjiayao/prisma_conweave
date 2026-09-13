"""Reward-only path policy search without a fixed local-queue score or warm start.

The policy consumes eight existing measurements. All eight score coefficients,
including their signs, are learned. Telemetry and flow-level path holding remain
part of the environment; this module does not claim to learn either mechanism.
"""
from pathlib import Path

import numpy as np


FEATURES = (0, 1, 2, 3, 5, 6, 7, 8)
FEATURE_NAMES = (
    "local_delay", "remote_delay", "local_serialization", "remote_serialization",
    "remote_change", "local_change", "destination_recent_work", "port_recent_work",
)
SIGNED = {5, 6}


def neutral():
    """All candidates have equal scores; the existing flow hash breaks ties."""
    return np.zeros(len(FEATURES), dtype=np.float64)


def parameters(theta, scale):
    theta, scale = np.asarray(theta), np.asarray(scale)
    if theta.shape != (8,) or scale.shape != (8,):
        raise ValueError("Expected eight coefficients and eight input scales")
    if not np.isfinite(theta).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid coefficients or input scales")
    w, b, v, c = np.zeros((24, 32)), np.zeros(32), np.zeros((32, 2)), np.zeros(2)
    h = 0
    for feature, weight in zip(FEATURES, theta / scale):
        w[feature, h], v[h, 1] = 1., weight
        h += 1
        if feature in SIGNED:
            w[feature, h], v[h, 1] = -1., -weight
            h += 1
    return [w, b, v, c]


def export(theta, scale, path):
    """Use the existing native actor format; this is not DQN training."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = parameters(theta, scale)
    path.write_text("ROUTE_DQN_V1 24 32\n" + "\n".join(
        " ".join(format(float(x), ".17g") for x in a.ravel()) for a in p) + "\n")
    np.savez(path.with_suffix(".npz"), **{f"p{i}": a for i, a in enumerate(p)},
             mask=False, gamma=1., step=0, transport=True)
    return str(path)


def reward(rows, failure_penalty=None):
    """Whole-network Monte Carlo return; no expert actions or expert FCT targets."""
    if not rows:
        raise ValueError("No rollouts")
    returns = []
    for r in rows:
        if not r["valid"] or r["completed"] != r["offered"]:
            if failure_penalty is None or r.get("training_failure") != "unfinished_at_horizon":
                raise ValueError("Every offered flow must complete before scoring a policy")
            returns.append(float(failure_penalty))
        else:
            returns.append(-100 * (np.log(r["all_mean_us"] / 1000.)
                                   + .25 * np.log(r["all_p99_us"] / 1000.)))
    return float(np.mean(returns))


def update(theta, directions, raw_rewards, permutation, step=.12, top=4, cap=.30):
    """ARS-style paired perturbations, with a symmetric unit-ball constraint.

    Permutation changes only which measured return is credited to a direction.
    No coefficient is fixed positive and no learned policy is mixed with a rule.
    """
    theta, directions = np.asarray(theta), np.asarray(directions)
    raw_rewards, permutation = np.asarray(raw_rewards), np.asarray(permutation)
    n = len(directions)
    if directions.shape != (n, 8) or raw_rewards.shape != (2 * n,):
        raise ValueError("Perturbation/return shape mismatch")
    if not np.array_equal(np.sort(permutation), np.arange(2 * n)):
        raise ValueError("Return labels must be a permutation")
    used = raw_rewards[permutation].reshape(n, 2)
    selected = np.argsort(-used.max(axis=1), kind="stable")[:top]
    std = max(.1, float(used[selected].std()))
    delta = step * np.mean((used[selected, 1] - used[selected, 0])[:, None]
                           * directions[selected], axis=0) / std
    delta *= min(1., cap / max(float(np.linalg.norm(delta)), 1e-12))
    after = theta + delta
    after /= max(1., float(np.linalg.norm(after)))
    return after, dict(selected=selected.tolist(), reward_std=std,
                       update=delta.tolist(), theta=after.tolist())
