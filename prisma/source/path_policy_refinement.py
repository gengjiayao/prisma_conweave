"""Condition exploration using completed training trajectories, not expert scores."""
import numpy as np


def direction_scale(previous_scale, sum_squares, count, lower=.25, upper=4.):
    old = np.asarray(previous_scale, dtype=float)
    squares = np.asarray(sum_squares, dtype=float)
    if old.shape != (8,) or squares.shape != (8,) or count <= 0:
        raise ValueError('Expected eight features and at least one training observation')
    if not np.isfinite(old).all() or not np.isfinite(squares).all() or np.any(old <= 0) or np.any(squares < 0):
        raise ValueError('Invalid training observation statistics')
    observed = np.maximum(np.sqrt(squares / count), 1e-6)
    factor = np.clip(old / observed, lower, upper)
    # An identically zero feature has no measured direction to explore.
    factor[squares == 0] = 0.
    return factor, observed


def conditioned_directions(rng, factor, count=8):
    factor = np.asarray(factor)
    if factor.shape != (8,) or not np.isfinite(factor).all() or np.any(factor < 0):
        raise ValueError('Invalid direction scales')
    return rng.normal(size=(count, 8)) * factor
