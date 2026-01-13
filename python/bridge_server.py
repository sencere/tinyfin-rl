#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import socket
import threading


def _space_info(space):
    if hasattr(space, "n"):
        return {
            "type": "discrete",
            "n": int(space.n),
            "dims": 1,
            "low": 0.0,
            "high": float(space.n - 1),
        }
    if hasattr(space, "shape") and hasattr(space, "low") and hasattr(space, "high"):
        shape = tuple(int(x) for x in space.shape)
        if len(shape) != 1 or shape[0] > 4:
            raise ValueError("only 1D Box with dims <= 4 supported")
        low = float(space.low[0]) if hasattr(space.low, "__len__") else float(space.low)
        high = float(space.high[0]) if hasattr(space.high, "__len__") else float(space.high)
        return {"type": "box", "n": 0, "dims": shape[0], "low": low, "high": high}
    raise ValueError("unsupported space")


def _make_env(kind, env_id):
    if kind == "gymnasium":
        import gymnasium as gym

        return gym.make(env_id)
    if kind == "retro":
        import retro

        return retro.make(env_id)
    if kind == "pettingzoo":
        module = importlib.import_module(f"pettingzoo.{env_id}")
        if hasattr(module, "parallel_env"):
            return module.parallel_env()
        if hasattr(module, "env"):
            return module.env()
        raise ValueError("pettingzoo env requires parallel_env or env")
    raise ValueError(f"unknown kind {kind}")


class EnvSession:
    def __init__(self, fp):
        self.fp = fp
        self.env = None
        self.kind = None
        self.env_id = None
        self.agent_ids = []
        self.obs_info = None
        self.act_info = None
        self.agent_count = 1
        self.max_steps = 0

    def _write(self, line):
        self.fp.write(line + "\n")
        self.fp.flush()

    def _init(self, kind, env_id, seed):
        self.kind = kind
        self.env_id = env_id
        self.env = _make_env(kind, env_id)
        if kind == "pettingzoo" and hasattr(self.env, "agents"):
            self.agent_ids = list(self.env.agents)
            self.agent_count = len(self.agent_ids)
        else:
            self.agent_count = 1
        if hasattr(self.env, "action_space"):
            self.act_info = _space_info(self.env.action_space)
            self.obs_info = _space_info(self.env.observation_space)
        else:
            spaces = self.env.action_spaces
            obs_spaces = self.env.observation_spaces
            first_agent = self.agent_ids[0]
            self.act_info = _space_info(spaces[first_agent])
            self.obs_info = _space_info(obs_spaces[first_agent])
        if hasattr(self.env, "spec") and getattr(self.env.spec, "max_episode_steps", None):
            self.max_steps = int(self.env.spec.max_episode_steps)
        self._write(
            "OK SPEC {obs_type} {obs_n} {obs_dims} {act_type} {act_n} {act_dims} {obs_low:.6f} {obs_high:.6f} {act_low:.6f} {act_high:.6f} {max_steps} {agent_count}".format(
                obs_type=0 if self.obs_info["type"] == "discrete" else 1,
                obs_n=self.obs_info["n"],
                obs_dims=self.obs_info["dims"],
                act_type=0 if self.act_info["type"] == "discrete" else 1,
                act_n=self.act_info["n"],
                act_dims=self.act_info["dims"],
                obs_low=self.obs_info["low"],
                obs_high=self.obs_info["high"],
                act_low=self.act_info["low"],
                act_high=self.act_info["high"],
                max_steps=self.max_steps,
                agent_count=self.agent_count,
            )
        )

    def _encode_obs(self, obs):
        if self.obs_info["type"] == "discrete":
            return [int(obs)]
        return [float(x) for x in obs]

    def _reset(self, seed):
        if self.kind == "gymnasium":
            obs, _info = self.env.reset(seed=seed)
            obs_list = [self._encode_obs(obs)]
        elif self.kind == "retro":
            obs = self.env.reset()
            obs_list = [self._encode_obs(obs)]
        else:
            obs = self.env.reset(seed=seed)
            if isinstance(obs, tuple):
                obs = obs[0]
            if hasattr(obs, "keys"):
                self.agent_ids = list(obs.keys())
                self.agent_count = len(self.agent_ids)
                obs_list = [self._encode_obs(obs[a]) for a in self.agent_ids]
            else:
                obs_list = [self._encode_obs(obs)]
        flat = []
        for o in obs_list:
            flat.extend(o)
        self._write("OK RESET {} {} {}".format(self.agent_count, 0 if self.obs_info["type"] == "discrete" else 1, " ".join(str(x) for x in flat)))

    def _step(self, actions):
        if self.kind == "gymnasium":
            obs, reward, terminated, truncated, _info = self.env.step(actions[0])
            done = terminated or truncated
            obs_list = [self._encode_obs(obs)]
            rewards = [float(reward)]
            dones = [1 if done else 0]
        elif self.kind == "retro":
            obs, reward, done, _info = self.env.step(actions[0])
            obs_list = [self._encode_obs(obs)]
            rewards = [float(reward)]
            dones = [1 if done else 0]
        else:
            act_dict = {a: actions[i] for i, a in enumerate(self.agent_ids)}
            obs, rewards, term, trunc, _info = self.env.step(act_dict)
            obs_list = [self._encode_obs(obs[a]) for a in self.agent_ids]
            rewards = [float(rewards[a]) for a in self.agent_ids]
            dones = [1 if (term[a] or trunc[a]) else 0 for a in self.agent_ids]
        flat = []
        for o in obs_list:
            flat.extend(o)
        payload = " ".join(str(x) for x in flat + rewards + dones)
        self._write("OK STEP {} {} {}".format(len(obs_list), 0 if self.obs_info["type"] == "discrete" else 1, payload))

    def _close(self):
        try:
            if self.env and hasattr(self.env, "close"):
                self.env.close()
        finally:
            self.env = None

    def handle(self):
        for line in self.fp:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0]
            try:
                if cmd == "INIT":
                    kind, env_id, seed = parts[1], parts[2], int(parts[3])
                    self._init(kind, env_id, seed)
                elif cmd == "RESET":
                    seed = int(parts[1])
                    self._reset(seed)
                elif cmd == "STEP":
                    agent_count = int(parts[1])
                    action_type = int(parts[2])
                    if action_type == 0:
                        actions = [int(x) for x in parts[3 : 3 + agent_count]]
                    else:
                        dims = self.act_info["dims"]
                        floats = [float(x) for x in parts[3 : 3 + agent_count * dims]]
                        actions = [floats[i * dims : (i + 1) * dims] for i in range(agent_count)]
                    self._step(actions)
                elif cmd == "CLOSE":
                    self._close()
                    break
                else:
                    self._write("ERR unknown_command")
            except Exception as exc:  # pragma: no cover - bridge errors
                self._write("ERR {}".format(str(exc).replace(" ", "_")))
        self._close()


def _client_thread(conn):
    with conn:
        fp = conn.makefile("r+")
        session = EnvSession(fp)
        session.handle()


def main():
    parser = argparse.ArgumentParser(description="Tinyfin-RL Python env bridge")
    parser.add_argument("--socket", default="/tmp/tfrl_py_bridge.sock")
    args = parser.parse_args()

    if os.path.exists(args.socket):
        os.unlink(args.socket)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    server.listen(5)
    print(f"bridge listening on {args.socket}")
    try:
        while True:
            conn, _addr = server.accept()
            thread = threading.Thread(target=_client_thread, args=(conn,), daemon=True)
            thread.start()
    finally:
        server.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)


if __name__ == "__main__":
    main()
