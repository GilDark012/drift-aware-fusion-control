"""Tests for the α-policy family."""

from __future__ import annotations

import numpy as np

from drift_reco.adaptation.policy import (
    AdaptationConfig,
    ContinuousAlphaPolicy,
    StateAlphaPolicy,
    StaticAlphaPolicy,
    build_policy,
)
from drift_reco.detectors.detector import DriftState

_Z = np.zeros(4)


def test_static_policy_is_constant() -> None:
    alpha = StaticAlphaPolicy(0.5).alpha_series(np.zeros(4, dtype=np.int64), _Z)
    assert np.allclose(alpha, 0.5)


def test_continuous_policy_lowers_alpha_with_z() -> None:
    policy = ContinuousAlphaPolicy(high=0.8, sensitivity=0.1, minimum=0.2)
    z = np.array([0.0, 5.0, 10.0])
    alpha = policy.alpha_series(np.zeros(3, dtype=np.int64), z)
    assert np.allclose(alpha, [0.8, 0.3, 0.2])


def test_state_abrupt_maps_each_state_to_its_target() -> None:
    levels = {0: 0.8, 1: 0.5, 2: 0.2, 3: 0.8}
    states = np.array([0, 1, 2, 0], dtype=np.int64)
    alpha = StateAlphaPolicy(levels, step=None).alpha_series(states, np.zeros(4))
    assert np.allclose(alpha, [0.8, 0.5, 0.2, 0.8])


def test_state_gradual_is_rate_limited() -> None:
    levels = {0: 0.8, 1: 0.5, 2: 0.2, 3: 0.8}
    states = np.array([2, 2, 2], dtype=np.int64)
    alpha = StateAlphaPolicy(levels, step=0.1).alpha_series(states, np.zeros(3))
    assert np.allclose(alpha, [0.7, 0.6, 0.5])
    assert np.all(np.abs(np.diff(alpha)) <= 0.1 + 1e-9)


def test_build_policy_selects_the_right_class() -> None:
    assert isinstance(build_policy(AdaptationConfig(policy="static")), StaticAlphaPolicy)
    grad = build_policy(AdaptationConfig(policy="state", mode="gradual"))
    abrupt = build_policy(AdaptationConfig(policy="state", mode="abrupt"))
    assert isinstance(grad, StateAlphaPolicy) and grad.step is not None
    assert isinstance(abrupt, StateAlphaPolicy) and abrupt.step is None
    # TEMPORARY behaves like STABLE
    levels = grad.levels
    assert levels[int(DriftState.TEMPORARY)] == levels[int(DriftState.STABLE)]
