from __future__ import annotations

from typing import Optional

from ..envs import Env
from ..spaces import Discrete, Box, Space
from .env_plugin import EnvPlugin


class EnvPluginAdapter(Env):
    def __init__(
        self,
        plugin: EnvPlugin,
        observation_space: Optional[Space] = None,
        action_space: Optional[Space] = None,
        obs_n: Optional[int] = None,
        action_n: Optional[int] = None,
    ):
        if observation_space is None and action_space is None and obs_n is None and action_n is None:
            meta = plugin.metadata()
            if meta.get("observation_type") == 1 and meta.get("observation_n", 0) > 0:
                obs_n = meta["observation_n"]
            if meta.get("action_type") == 1 and meta.get("action_n", 0) > 0:
                action_n = meta["action_n"]
            if meta.get("observation_type") == 2 and meta.get("observation_dims", 0) > 0:
                observation_space = Box(meta["observation_low"], meta["observation_high"], (meta["observation_dims"],))
            if meta.get("action_type") == 2 and meta.get("action_dims", 0) > 0:
                action_space = Box(meta["action_low"], meta["action_high"], (meta["action_dims"],))
        if observation_space is None and obs_n is not None:
            observation_space = Discrete(obs_n)
        if action_space is None and action_n is not None:
            action_space = Discrete(action_n)
        self.observation_space = observation_space
        self.action_space = action_space
        self._plugin = plugin

    def reset(self):
        return self._plugin.reset()

    def step(self, action):
        return self._plugin.step(action)

    def seed(self, seed: int) -> int:
        return self._plugin.seed(seed)

    def close(self) -> None:
        self._plugin.close()
