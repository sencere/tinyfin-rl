from typing import List

from .envs import Env
from .vector_env import VectorEnv


def make(name: str, num_envs: int = 1) -> Env | VectorEnv | object:
    name = name.lower()
    if name.startswith("gym:") or name.startswith("gymnasium:"):
        gym_name = name.split(":", 1)[1]

        def _make():
            from .adapters.gymnasium_adapter import make_gymnasium
            return make_gymnasium(gym_name)
    elif name.startswith("pettingzoo:"):
        if num_envs > 1:
            raise ValueError("pettingzoo envs do not support num_envs > 1")
        pz_name = name.split(":", 1)[1]
        from .adapters.pettingzoo_adapter import make_pettingzoo_parallel
        return make_pettingzoo_parallel(pz_name)
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
