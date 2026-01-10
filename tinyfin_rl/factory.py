from typing import List

from .envs import BanditEnv, GridworldEnv, Env
from .vector_env import VectorEnv


def make(name: str, num_envs: int = 1) -> Env | VectorEnv:
    name = name.lower()
    if name == "bandit":
        def _make():
            return BanditEnv([0.1, 0.5, 0.9])
    elif name == "gridworld":
        def _make():
            return GridworldEnv()
    else:
        raise ValueError(f"unknown env name: {name}")
    if num_envs <= 1:
        return _make()
    envs: List[Env] = [_make() for _ in range(num_envs)]
    return VectorEnv(envs)


def emulate(env: Env, num_envs: int = 1, env_factory=None) -> Env | VectorEnv:
    if num_envs <= 1:
        return env
    if env_factory is None:
        try:
            env_factory = env.__class__
        except AttributeError as exc:
            raise ValueError("env_factory required for multi-env emulation") from exc
    return VectorEnv([env_factory() for _ in range(num_envs)])
