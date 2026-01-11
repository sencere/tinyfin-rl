from _bootstrap import setup_path
setup_path()

from tinyfin_rl.adapters.gymnasium_adapter import make_gymnasium
from tinyfin_rl.algos.ppo import PPOTrainer
from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.policy_tinyfin import PPOPolicy
from tinyfin_rl.seed import set_seed


class DiscretizedEnv:
    def __init__(self, env, bins):
        self.env = env
        self.bins = bins
        self.action_space = env.action_space
        total = 1
        for _, _, n_bins in bins:
            total *= n_bins
        self.observation_space = type("Space", (), {"n": total})()

    def _disc(self, obs):
        idx = 0
        for val, (low, high, n_bins) in zip(obs, self.bins):
            if val <= low:
                b = 0
            elif val >= high:
                b = n_bins - 1
            else:
                b = int((val - low) / (high - low) * n_bins)
            idx = idx * n_bins + b
        return idx

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return self._disc(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._disc(obs), reward, done, info


def main():
    set_seed(0)
    base_env = make_gymnasium("CartPole-v1")
    bins = [(-2.4, 2.4, 6), (-3.0, 3.0, 6), (-0.5, 0.5, 6), (-2.0, 2.0, 6)]
    env = DiscretizedEnv(base_env, bins=bins)
    backend = TinyfinBackend(device="cpu")
    policy = PPOPolicy(backend=backend, obs_dim=env.observation_space.n, action_space=env.action_space)
    trainer = PPOTrainer(env=env, policy=policy, steps_per_batch=64, epochs=2)
    trainer.train(updates=5)


if __name__ == "__main__":
    main()
