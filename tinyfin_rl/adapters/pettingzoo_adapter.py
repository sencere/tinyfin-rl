from __future__ import annotations

import importlib
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
    env_fn = getattr(module, env_name)
    env = env_fn.parallel_env(**kwargs)
    return ParallelPettingZooEnv(env)
