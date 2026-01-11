from __future__ import annotations

import importlib
import importlib.util
from typing import Dict


class ParallelPettingZooEnv:
    def __init__(self, env):
        self.env = env
        self.possible_agents = getattr(env, "possible_agents", [])

    def reset(self):
        obs, info = self.env.reset()
        return obs, info

    def step(self, actions: Dict):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        dones = {agent: bool(terminations[agent] or truncations[agent]) for agent in obs.keys()}
        return obs, rewards, dones, infos


def make_pettingzoo_parallel(name: str, **kwargs) -> ParallelPettingZooEnv:
    try:
        import pettingzoo
    except ImportError as exc:
        raise RuntimeError("pettingzoo is not installed") from exc

    if "/" not in name:
        raise ValueError("pettingzoo env name must include module, e.g. 'butterfly/coop_pong_v5'")
    module_name, env_name = name.split("/", 1)
    module = importlib.import_module(f"pettingzoo.{module_name}")
    env_fn = None
    spec = importlib.util.find_spec(f"pettingzoo.{module_name}.{env_name}")
    if spec is not None:
        submodule = importlib.import_module(f"pettingzoo.{module_name}.{env_name}")
        env_fn = getattr(submodule, "parallel_env", None)
        if env_fn is None:
            raise AttributeError(f"missing parallel_env for {name}")
        env = env_fn(**kwargs)
    else:
        env_attr = getattr(module, env_name, None)
        if env_attr is None:
            raise AttributeError(f"missing env {name}")
        env = env_attr.parallel_env(**kwargs)
    return ParallelPettingZooEnv(env)
