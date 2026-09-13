"""Diagnostic routing policies for the v8 observation producer.

Invert its queue encoding and divide backlog by each link's service rate.
These evaluation-only policies do not change or retrain a saved model.
The reference reconstruction matches v8's 40 us window, 16 KiB floor and
0.001 buffer fraction; a different producer requires a different contract.
"""
import numpy as np
from source.rl_contract import infer_num_actions, native_action_prior


def _mix64(value):
    """Fixed SplitMix64 permutation, independent of Python's hash seed."""
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def choose_balanced_action(scores, flow_key, band):
    """Stable per-flow ranking restricted to a bounded native-score deficit."""
    q = np.asarray(scores, dtype=np.float32)
    if q.ndim != 1 or q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError('Balanced routing requires finite one-dimensional scores')
    if not np.isfinite(band) or not 0 <= band <= .2:
        raise ValueError('Balance band must be finite and in [0, 0.2]')
    best = int(np.argmax(q))
    if band == 0:
        return best
    if not isinstance(flow_key, int) or isinstance(flow_key, bool) or not 0 <= flow_key < (1 << 64):
        raise ValueError('Balanced routing requires the exact unsigned 64-bit flow key')
    candidates = [i for i in range(q.size) if float(q[best]) - float(q[i]) <= band]
    return max(candidates, key=lambda i: _mix64(flow_key ^ _mix64(i + 0xD1B54A32D192ED03)))


def choose_hysteresis_action(scores, previous_action, has_previous, margin):
    """Retain an established route unless a new flowlet gains more than margin.

    A first flowlet always uses the native argmax. Zero margin preserves the
    historical argmax and tie-breaking exactly, including established routes.
    """
    q = np.asarray(scores, dtype=np.float32)
    if q.ndim != 1 or q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError('Hysteresis requires a finite one-dimensional score vector')
    if not np.isfinite(margin) or not 0 <= margin <= .2:
        raise ValueError('Switch margin must be finite and in [0, 0.2]')
    best = int(np.argmax(q))
    if margin == 0:
        return best
    if has_previous is None:
        raise ValueError('Simulator did not provide current-flowlet route ownership')
    if not has_previous:
        return best
    if not 0 <= previous_action < q.size:
        raise ValueError('Previous route is outside the candidate action set')
    if float(q[best]) - float(q[previous_action]) <= margin:
        return int(previous_action)
    return best


def service_delay_us(observation, buffer_bytes, max_link_gbps, ema_weight=0.0,
                     horizon_us=0.0):
    obs = np.asarray(observation, dtype=np.float64)
    native_action_prior(obs)  # validate the normalized feature contract
    if buffer_bytes <= 0 or max_link_gbps <= 0 or not 0 <= ema_weight <= 1 or horizon_us < 0:
        raise ValueError("Invalid service-delay parameters")
    n = infer_num_actions(obs.shape[-1])
    f = obs[..., 2:].reshape(obs.shape[:-1] + (n, 6))
    rates = f[..., 5] * max_link_gbps * 1e9 / 8.0
    if np.any(rates <= 0):
        raise ValueError("Service-delay policy requires positive candidate capacities")
    reference = np.maximum(np.maximum(rates * 40e-6, 16384.0), buffer_bytes * 0.001)
    # The quantized producer saturates at one. Interpret that conservatively
    # as at least a full configured buffer rather than dividing by zero.
    def invert(x):
        return np.where(x >= 1.0, buffer_bytes, reference * x / np.maximum(1.0-x, 1e-12))
    backlog = (1-ema_weight)*invert(f[..., 0]) + ema_weight*invert(f[..., 1])
    growth = np.maximum(0.0, 2*f[..., 3]-1.0)
    return (backlog + 1024.0) / rates * 1e6 + horizon_us * growth


def diagnostic_scores(observation, mode, buffer_bytes, max_link_gbps,
                      horizon_us=20.0, slack_us=20.0, learned_q=None):
    if mode not in ("queue", "queue_ema", "predictive", "guard"):
        raise ValueError("Unknown diagnostic policy")
    if slack_us < 0:
        raise ValueError("Negative delay slack")
    delay = service_delay_us(observation, buffer_bytes, max_link_gbps,
                             ema_weight=0.0 if mode == "queue" else 0.25,
                             horizon_us=horizon_us if mode in ("predictive", "guard") else 0.0)
    if mode != "guard":
        return 1.0 / (1.0 + delay / 20.0)
    q = np.asarray(learned_q, dtype=np.float64)
    if q.shape != delay.shape or not np.all(np.isfinite(q)):
        raise ValueError("Guard requires finite learned scores of matching shape")
    allowed = delay <= np.min(delay, axis=-1, keepdims=True) + slack_us
    # Retain the learned ordering only inside the service-delay allowance.
    floor = np.min(q, axis=-1, keepdims=True)-1.0
    return np.where(allowed, q, floor)
