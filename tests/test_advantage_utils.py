import os
import sys

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

import math

from tinyfin_rl.algos.utils import normalize_advantages, policy_entropy, value_loss


def test_normalize_advantages_basic():
    adv = [1.0, 2.0, 3.0]
    out = normalize_advantages(adv)
    mean = sum(out) / len(out)
    var = sum((x - mean) ** 2 for x in out) / len(out)
    assert abs(mean) < 1e-7
    assert abs(math.sqrt(var) - 1.0) < 1e-6


def test_normalize_advantages_constant():
    adv = [5.0, 5.0, 5.0]
    out = normalize_advantages(adv)
    assert all(abs(x) < 1e-7 for x in out)


def test_policy_entropy_uniform():
    probs = [0.25, 0.25, 0.25, 0.25]
    ent = policy_entropy(probs)
    assert abs(ent - math.log(4.0)) < 1e-6


def test_policy_entropy_peaked():
    probs = [1.0, 0.0, 0.0]
    ent = policy_entropy(probs)
    assert abs(ent) < 1e-7


def test_value_loss_scalar():
    loss = value_loss(2.0, 1.0, coef=0.5)
    assert abs(loss - 0.5) < 1e-8
