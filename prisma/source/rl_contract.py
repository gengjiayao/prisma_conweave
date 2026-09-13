"""Shared contracts for PRISMA's RL observation and transition pipeline.

The simulator emits one observation per leaf agent with this layout::

    [destination_overlay, last_action, egress_0_features..., ...]

The active flowlet/ZMQ path stores the observation in
``ConweaveObsManager::m_lastObsFeats`` as unsigned integers.  ``last_action``
is an egress index and the six features per egress are quantised with a fixed
scale of 5000.  Python performs that single, explicit conversion before a
sample reaches either the policy or replay training.

The conflicting unused ``BuildObservation`` implementation was removed in v6.
rl-core-v7 removed the active producer's dependency on dormant CONGA state.
rl-core-v8 makes every detected flowlet boundary request a fresh action and
adds a bounded learned advantage around an RL-native congestion prior, so an
under-trained checkpoint cannot prefer a saturated fast link to an idle slow
link merely because it memorised link capacity.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np


RL_CORE_VERSION = "rl-core-v8"
OBS_HEADER_DIMS = 2
FEATURES_PER_EGRESS = 6
DEFAULT_FEATURE_SCALE = 5000.0
EGRESS_FEATURE_NAMES = (
    "queue_occupancy",
    "queue_ema",
    "dre_utilization",
    "queue_trend",
    "headroom",
    "link_capacity",
)


class AlignedTransition(NamedTuple):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool
    ecmp_action: int


def expected_observation_dim(num_actions: int) -> int:
    if int(num_actions) <= 0:
        raise ValueError(f"num_actions must be positive, got {num_actions}")
    return OBS_HEADER_DIMS + int(num_actions) * FEATURES_PER_EGRESS


def infer_num_actions(observation_dim: int) -> int:
    payload_dim = int(observation_dim) - OBS_HEADER_DIMS
    if payload_dim <= 0 or payload_dim % FEATURES_PER_EGRESS != 0:
        raise ValueError(
            "Invalid observation dimension: expected "
            f"{OBS_HEADER_DIMS} + N*{FEATURES_PER_EGRESS}, got {observation_dim}"
        )
    return payload_dim // FEATURES_PER_EGRESS


def validate_observation_shape(observation_dim: int, num_actions: int) -> None:
    expected = expected_observation_dim(num_actions)
    if int(observation_dim) != expected:
        raise ValueError(
            "Python/C++ observation contract mismatch: "
            f"expected dim {expected} for {num_actions} actions, got {observation_dim}"
        )


def preprocess_observation(
    observation,
    num_actions: Optional[int] = None,
    feature_scale: Optional[float] = None,
) -> np.ndarray:
    """Convert raw simulator observations into model inputs.

    ``destination_overlay`` remains an integer-valued float so the model's
    one-hot encoder can distinguish destinations.  The active C++ producer
    emits ``last_action`` as an integer action index and egress features as
    uint32 values quantised by ``DEFAULT_FEATURE_SCALE``.  v6 guarantees all
    six fields, including the explicit relative link capacity, are bounded.
    """
    if observation is None:
        return observation

    arr = np.asarray(observation, dtype=np.float32)
    if arr.ndim not in (1, 2):
        raise ValueError(f"Observation must be rank 1 or 2, got shape {arr.shape}")
    if arr.shape[-1] == 0:
        return arr.copy()
    if not np.all(np.isfinite(arr)):
        raise ValueError("Observation contains NaN or infinity")

    if num_actions is None:
        num_actions = infer_num_actions(arr.shape[-1])
    validate_observation_shape(arr.shape[-1], num_actions)

    scale = DEFAULT_FEATURE_SCALE if feature_scale is None else float(feature_scale)
    if not np.isfinite(scale) or not np.isclose(scale, DEFAULT_FEATURE_SCALE):
        raise ValueError(
            "Python/C++ observation contract requires the simulator's fixed "
            f"feature scale {DEFAULT_FEATURE_SCALE}, got {feature_scale}"
        )

    out = arr.copy()
    tolerance = 1e-6
    last_action = out[..., 1]
    if (
        np.any(last_action < -tolerance)
        or np.any(last_action > float(num_actions - 1) + tolerance)
        or np.any(np.abs(last_action - np.rint(last_action)) > tolerance)
    ):
        raise ValueError(
            "Python/C++ observation contract mismatch: last_action must be an "
            f"integer index in [0, {num_actions - 1}]"
        )
    if num_actions > 1:
        out[..., 1] = last_action / float(num_actions - 1)
    else:
        out[..., 1] = 0.0

    raw_features = out[..., OBS_HEADER_DIMS:]
    if np.any(raw_features < -tolerance) or np.any(raw_features > scale + tolerance):
        raise ValueError(
            "Python/C++ observation contract mismatch: quantized egress "
            f"features must be in [0, {scale}]"
        )
    out[..., OBS_HEADER_DIMS:] = np.clip(raw_features, 0.0, scale) / scale
    return out


def native_action_prior(
    processed_observation,
    num_actions: Optional[int] = None,
) -> np.ndarray:
    """Compute the fixed rl-core-v8 per-egress safety prior.

    The input must already have passed :func:`preprocess_observation`.  Keeping
    this NumPy implementation next to the observation contract lets frozen
    evaluation isolate the native prior without changing or rebuilding a v8
    checkpoint.  The weights intentionally match the
    ``rl_native_action_prior`` layer in ``DQN_buffer_model`` exactly.
    """
    arr = np.asarray(processed_observation, dtype=np.float32)
    if arr.ndim not in (1, 2):
        raise ValueError(
            f"Processed observation must be rank 1 or 2, got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Processed observation contains NaN or infinity")
    if num_actions is None:
        num_actions = infer_num_actions(arr.shape[-1])
    num_actions = int(num_actions)
    validate_observation_shape(arr.shape[-1], num_actions)

    features = arr[..., OBS_HEADER_DIMS:].reshape(
        arr.shape[:-1] + (num_actions, FEATURES_PER_EGRESS)
    )
    tolerance = 1e-6
    if np.any(features < -tolerance) or np.any(features > 1.0 + tolerance):
        raise ValueError(
            "Native prior requires preprocessed egress features in [0, 1]"
        )
    features = np.clip(features, 0.0, 1.0)
    prior = (
        0.6 * features[..., 4]
        + 0.2 * (1.0 - features[..., 1])
        + 0.1 * (1.0 - features[..., 0])
        + 0.1 * (1.0 - features[..., 3])
    )
    return np.asarray(prior, dtype=np.float32)


def build_aligned_transition(
    observation,
    action: int,
    reward: float,
    next_observation,
    done: bool,
    ecmp_action: int = -1,
    num_actions: Optional[int] = None,
) -> AlignedTransition:
    """Build the transition returned by the same ``env.step(action)`` call."""
    obs = np.asarray(observation, dtype=np.float32).squeeze()
    next_obs = np.asarray(next_observation, dtype=np.float32).squeeze()
    if obs.ndim != 1 or next_obs.ndim != 1 or obs.shape != next_obs.shape:
        raise ValueError(
            f"Transition observations must be equal 1-D shapes, got {obs.shape} and {next_obs.shape}"
        )
    inferred_actions = infer_num_actions(obs.shape[0]) if num_actions is None else int(num_actions)
    validate_observation_shape(obs.shape[0], inferred_actions)
    action = int(action)
    if action < 0 or action >= inferred_actions:
        raise ValueError(f"Action {action} outside [0, {inferred_actions})")
    reward = float(reward)
    if not np.isfinite(reward):
        raise ValueError(f"Reward must be finite, got {reward}")
    return AlignedTransition(obs, action, reward, next_obs, bool(done), int(ecmp_action))


def is_valid_environment_step(
    connected: bool,
    is_control: bool,
    action_applied: bool,
) -> bool:
    """Return whether an env.step response can close a real transition.

    ns3-gym keeps the previous ``extraInfo`` when the simulation-end message is
    received.  ``connected`` must therefore gate the cached packet/action flags,
    otherwise each forwarder writes one duplicate terminal transition.
    """
    return bool(connected and not is_control and action_applied)


def scheduled_gradient_steps(
    total_samples: int,
    learning_starts: int,
    train_every: int,
) -> int:
    """Return the deterministic number of DQN updates due for one node.

    Training used to depend on how often a wall-clock trainer thread happened
    to wake up while ns-3 was running.  Tying updates to replay insertions makes
    the amount of learning reproducible for a fixed traffic trace and seed.
    The first update is due as soon as ``learning_starts`` samples exist.
    """
    total_samples = int(total_samples)
    learning_starts = int(learning_starts)
    train_every = int(train_every)
    if total_samples < 0:
        raise ValueError(f"total_samples must be non-negative, got {total_samples}")
    if learning_starts <= 0:
        raise ValueError(f"learning_starts must be positive, got {learning_starts}")
    if train_every <= 0:
        raise ValueError(f"train_every must be positive, got {train_every}")
    if total_samples < learning_starts:
        return 0
    return 1 + (total_samples - learning_starts) // train_every


def should_update_target(gradient_steps: int, interval: int) -> bool:
    """Return whether the lagged target network is due for a hard update."""
    gradient_steps = int(gradient_steps)
    interval = int(interval)
    if interval <= 0:
        raise ValueError(f"target update interval must be positive, got {interval}")
    return gradient_steps > 0 and gradient_steps % interval == 0


def checkpoint_manifest(flowlet_gap_us=20.0, dre_tau_us=1000.0) -> dict:
    if not np.isfinite(flowlet_gap_us) or flowlet_gap_us <= 0 or not np.isfinite(dre_tau_us) or dre_tau_us <= 0:
        raise ValueError('Checkpoint timing values must be finite and positive')
    return {
        "rl_core_version": RL_CORE_VERSION,
        "observation": {
            "producer": "PrepareAndSendObservation_rl_native_uint32_quantized",
            "state_source": "rl_mac_tx_dre_local_queue",
            "dre_time_constant_seconds": float(dre_tau_us) * 1e-6,
            "layout": "[destination_overlay,last_action_index]+N*6_uint32_egress_features",
            "egress_features": list(EGRESS_FEATURE_NAMES),
            "destination_encoding": "raw_integer_one_hot",
            "last_action_scaling": "raw_index_divide_by_num_actions_minus_one",
            "last_action_used_by_q_network": False,
            "continuous_feature_scale": DEFAULT_FEATURE_SCALE,
            "continuous_features": "bounded_uint32_[0,5000]_divide_by_5000",
        },
        "reward": {
            "version": "rl_native_safe_advantage_v1",
            "headroom": "relative_link_capacity*(1-dre_utilization)",
            "weights": {
                "headroom": 0.6,
                "queue_ema_health": 0.2,
                "queue_instant_health": 0.1,
                "queue_trend_health": 0.1,
            },
            "queue_trend_normalization": "queue_bytes_per_second/link_bytes_per_second",
            "reorder_sequence_scope": "per_flow",
            "reorder_role": "diagnostic_only_per_flowlet_route_ownership",
        },
        "flowlet_routing": {
            "boundary_seconds": float(flowlet_gap_us) * 1e-6,
            "fresh_action_each_new_flowlet": True,
            "additional_dwell_seconds": 0.0,
            "continuation_route_owner": "rl_flowlet_context",
            "silent_ecmp_fallback": False,
        },
        "transition": "env.step(a_t)->(s_t,a_t,r_t_plus_1,s_t_plus_1,done)",
        "q_head_activation": "linear_state_value_plus_bounded_tanh_residual",
        "q_network": "safe_rl_native_prior_plus_shared_bounded_residual",
        "safe_action_prior": {
            "weights": {
                "headroom": 0.6,
                "queue_ema_health": 0.2,
                "queue_instant_health": 0.1,
                "queue_trend_health": 0.1,
            },
            "learned_residual_bound": 0.1,
            "uses_external_routing_labels": False,
        },
        "training": {
            "scheduler": "per_node_replay_samples",
            "target_update": "per_node_gradient_steps",
            "bootstrap": "double_dqn",
            "imitation_learning": "disabled_without_flow_identity",
        },
    }
