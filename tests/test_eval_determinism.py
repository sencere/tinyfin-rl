import os
import sys

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

import pytest
from tinyfin_rl.eval import evaluate_policy
from tinyfin_rl.policy import RandomPolicy


def test_eval_determinism():
    pytest.skip("Python BanditEnv removed; use C env plugin adapter instead.")
    policy = RandomPolicy(env.action_space)
    out1 = evaluate_policy(env, policy, episodes=5, seed=123)
    out2 = evaluate_policy(env, policy, episodes=5, seed=123)
    assert out1["returns"] == out2["returns"]
