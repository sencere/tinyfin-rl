from _bootstrap import setup_path
setup_path()

from tinyfin_rl.adapters.pettingzoo_adapter import make_pettingzoo_parallel


def _action_space(env, agent):
    if hasattr(env, "action_space"):
        return env.action_space(agent)
    action_spaces = getattr(env, "action_spaces", {})
    return action_spaces.get(agent)


def main():
    env = make_pettingzoo_parallel("butterfly/cooperative_pong_v5", render_mode="human")
    obs, _ = env.reset()
    steps = 0
    while steps < 200:
        actions = {}
        for agent in obs.keys():
            space = _action_space(env.env, agent)
            if space is None:
                raise RuntimeError(f"missing action space for {agent}")
            actions[agent] = space.sample()
        obs, rewards, dones, _ = env.step(actions)
        env.env.render()
        steps += 1
        if all(dones.values()):
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
