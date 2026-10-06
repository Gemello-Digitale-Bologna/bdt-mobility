"""Shared graph functions registered with the evaluation engine at evaluate()."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import combinations

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants shared across transport sub-models
# ---------------------------------------------------------------------------

P_PROB_THRESHOLD = 0.005
P_RECORD_FREQUENCY = 12
P_RECORD_HEADWAY = 1 / P_RECORD_FREQUENCY
P_DWELL_TIME_ZOI = 1 / 3


# ---------------------------------------------------------------------------
# Shared graph functions
# ---------------------------------------------------------------------------


def _ts_sum(ts: np.ndarray) -> np.ndarray:
    return ts.sum(axis=-1, keepdims=True)


def ts_solve(
    ts: np.ndarray,
    dwell_time: float = P_DWELL_TIME_ZOI,
    max_iter: int = 100,
    min_diff: float = 1e-5,
) -> np.ndarray:
    """Solve steady-state traffic iteratively."""
    tau = dwell_time * P_RECORD_FREQUENCY
    series = ts.copy()
    for iter in range(max_iter):
        mu = 1 + (tau - 1)
        alfa = (mu - 1) / mu
        next_series = ts + np.roll(series, 1, axis=-1) * alfa

        if np.max(np.abs(series - next_series)) < min_diff:
            break
        series = next_series
    return series


def ts_b_choose(w_a: np.ndarray, *list_w_b: np.ndarray) -> np.ndarray:
    """Competitive probability allocation across behaviour fractions."""
    if np.all(w_a == 1):
        return np.ones_like(w_a)
    if len(list_w_b) == 0:
        return np.ones_like(w_a)
    for w_b in list_w_b:
        if np.all(w_b == 1):
            return np.zeros_like(w_a)
    p = np.zeros_like(w_a)
    for r in range(len(list_w_b) + 1):
        for indices in combinations(range(len(list_w_b)), r):
            p_tmp = w_a.copy()
            for i, w_b in enumerate(list_w_b):
                if i in indices:
                    p_tmp = p_tmp * w_b
                else:
                    p_tmp = p_tmp * (1 - w_b)
            if r > 0:
                denominator = w_a.copy()
                for idx in indices:
                    denominator = denominator + list_w_b[idx]
                weight = np.divide(w_a, denominator, out=np.zeros_like(denominator), where=denominator != 0)
                p_tmp = p_tmp * weight
            p = p + p_tmp
    return p


def ts_anticipate(
    number_anticipating: np.ndarray,  delta_from_start: np.ndarray, p50_anticipating: float,  # need to be scalar
) -> np.ndarray:
    """Redistribute anticipating vehicles before the policy window."""
    # First, check the shapes of the ndarrays
    if number_anticipating.ndim != 2: ## 2D Array (n, m)
        raise ValueError(f"DimensionalityError: 'number_anticipating' must be 2D (n, m), got shape {number_anticipating.shape}")
    n, m = number_anticipating.shape
    if delta_from_start.ndim != 1: ## 1D Array (m,)
        raise ValueError(f"DimensionalityError: 'delta_from_start' must be 1D (m,), got shape {delta_from_start.shape}")
    if delta_from_start.shape[0] != m:
        raise ValueError(f"DimensionalityError: 'delta_from_start' length must be {m}, got {delta_from_start.shape[0]}")
    if not (np.isscalar(p50_anticipating) or np.ndim(p50_anticipating) == 0):
        raise ValueError(f"DimensionalityError: 'p50_anticipating' must be a scalar, got shape {p50_anticipating.shape}")

    # Then compute anticipating redistribution
    t0 = np.where(delta_from_start == 0)[0][0]
    n, tmax = number_anticipating.shape
    range1 = np.arange(0, tmax) * P_RECORD_HEADWAY
    range2 = np.arange(1, tmax + 1) * P_RECORD_HEADWAY

    v1 = np.exp(range1 / p50_anticipating * np.log(0.5)) - np.exp(
        range2 / p50_anticipating * np.log(0.5)
    )
    v1 = np.where(v1 < P_PROB_THRESHOLD, 0, v1)
    number_anticipated = np.zeros_like(number_anticipating)
    for t in range(t0, tmax):
        v1_here = v1[(t - t0):]
        v1_here_2d = v1_here[np.newaxis, :]
        v1_here_sum = _ts_sum(v1_here_2d)

        if v1_here_sum[0, 0] > 0:
            v1_normalized = v1_here_2d / v1_here_sum
            for deltat in range(1, t0 + 1):
                if deltat - 1 < v1_normalized.shape[1]:
                    weight = v1_normalized[0, deltat - 1]
                    number_anticipated[:, t0 - deltat] += (
                        number_anticipating[:, t] * weight
                    )

    return number_anticipated


def ts_postpone(
    number_postponing: np.ndarray, delta_to_end: np.ndarray,  p50_postponing: float,
) -> np.ndarray:
    """Redistribute postponing vehicles after the policy window."""

    # First, check the shapes of the ndarrays
    if number_postponing.ndim != 2: ## 2D Array (n, m)
        raise ValueError(f"DimensionalityError: 'number_postponing' must be 2D (n, m), got shape {number_postponing.shape}")
    n, m = number_postponing.shape
    if delta_to_end.ndim != 1: ## 1D Array (m,)
        raise ValueError(f"DimensionalityError: 'delta_to_end' must be 1D (m,), got shape {delta_to_end.shape}")
    if delta_to_end.shape[0] != m:
        raise ValueError(f"DimensionalityError: 'delta_to_end' length must be {m}, got {delta_to_end.shape[0]}")
    if not (np.isscalar(p50_postponing) or np.ndim(p50_postponing) == 0):
        raise ValueError(f"DimensionalityError: 'p50_postponing' must be a scalar, got shape {p50_postponing.shape}")

    # Then compute postponement redistribution
    t0 = np.where(delta_to_end == 0)[0][0]
    n, tmax = number_postponing.shape

    range1 = np.arange(0, tmax) * P_RECORD_HEADWAY
    range2 = np.arange(1, tmax + 1) * P_RECORD_HEADWAY

    v1 = np.exp(range1 / p50_postponing * np.log(0.5)) - np.exp(
        range2 / p50_postponing * np.log(0.5)
    )
    v1 = np.where(v1 < P_PROB_THRESHOLD, 0, v1)
    number_postponed = np.zeros_like(number_postponing)
    for t in range(t0, -1, -1):
        number_postponing_here = number_postponing[:, t]
        v1_here = v1[(t0 - t):]
        v1_here_2d = v1_here[np.newaxis, :]
        v1_here_sum = _ts_sum(v1_here_2d)
        if v1_here_sum[0, 0] > 0:
            v1_normalized = v1_here_2d / v1_here_sum
            for deltat in range(tmax - t0 - 1):
                if deltat < v1_normalized.shape[1]:
                    weight = v1_normalized[0, deltat]
                    number_postponed[:, t0 + 1 + deltat] += (
                        number_postponing_here * weight
                    )

    return number_postponed
