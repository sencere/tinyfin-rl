import os
import sys

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.factory import make, emulate
from tinyfin_rl.envs import BanditEnv
from tinyfin_rl.vector_env import VectorEnv


def test_make_single():
    env = make("bandit")
    assert isinstance(env, BanditEnv)


def test_make_vector():
    envs = make("gridworld", num_envs=3)
    assert isinstance(envs, VectorEnv)
    assert len(envs.envs) == 3


def test_emulate_requires_factory_for_args():
    env = BanditEnv([0.2, 0.8])
    envs = emulate(env, num_envs=2, env_factory=lambda: BanditEnv([0.2, 0.8]))
    assert isinstance(envs, VectorEnv)
