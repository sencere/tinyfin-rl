import os
import sys

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.envs import BanditEnv
from tinyfin_rl.eval import evaluate_policy
from tinyfin_rl.policy import RandomPolicy


def test_eval_determinism():
    env = BanditEnv([0.2, 0.8])
    policy = RandomPolicy(env.action_space)
    out1 = evaluate_policy(env, policy, episodes=5, seed=123)
    out2 = evaluate_policy(env, policy, episodes=5, seed=123)
    assert out1["returns"] == out2["returns"]
