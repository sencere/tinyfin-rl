import os
import sys

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.algos.ppo import compute_returns as ppo_returns
from tinyfin_rl.algos.ppo import compute_gae
from tinyfin_rl.algos.a2c import compute_returns as a2c_returns


def test_compute_returns_with_dones():
    rewards = [1.0, 1.0, 1.0]
    dones = [False, False, True]
    gamma = 0.9
    expected = [2.71, 1.9, 1.0]
    out = ppo_returns(rewards, dones, gamma)
    assert len(out) == len(expected)
    for got, exp in zip(out, expected):
        assert abs(got - exp) < 1e-6


def test_compute_returns_a2c_matches_ppo():
    rewards = [0.5, 0.0, 2.0]
    dones = [False, True, False]
    gamma = 0.95
    out_ppo = ppo_returns(rewards, dones, gamma)
    out_a2c = a2c_returns(rewards, dones, gamma)
    for got, exp in zip(out_a2c, out_ppo):
        assert abs(got - exp) < 1e-6


def test_compute_gae_simple():
    rewards = [1.0, 1.0]
    dones = [False, True]
    values = [0.5, 0.2]
    gamma = 0.9
    lam = 0.95
    expected = [1.364, 0.8]
    out = compute_gae(rewards, dones, values, gamma, lam)
    assert len(out) == len(expected)
    for got, exp in zip(out, expected):
        assert abs(got - exp) < 1e-6


def _compute_gae_reference(rewards, dones, values, gamma, lam):
    advantages = []
    last_adv = 0.0
    next_value = 0.0
    for r, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
        if done:
            next_value = 0.0
            last_adv = 0.0
        delta = r + gamma * next_value - value
        last_adv = delta + gamma * lam * last_adv
        advantages.append(last_adv)
        next_value = value
    advantages.reverse()
    return advantages


def test_compute_gae_reference_match():
    rewards = [0.2, 0.0, 1.0, 0.5]
    dones = [False, False, True, False]
    values = [0.1, -0.2, 0.3, 0.0]
    gamma = 0.97
    lam = 0.92
    expected = _compute_gae_reference(rewards, dones, values, gamma, lam)
    out = compute_gae(rewards, dones, values, gamma, lam)
    assert len(out) == len(expected)
    for got, exp in zip(out, expected):
        assert abs(got - exp) < 1e-9
