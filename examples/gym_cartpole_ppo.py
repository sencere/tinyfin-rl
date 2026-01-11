from _bootstrap import setup_path
setup_path()

import argparse

from tinyfin_rl.adapters.gymnasium_adapter import make_gymnasium
from tinyfin_rl.algos.ppo import PPOTrainer
from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.policy_tinyfin import PPOPolicy
from tinyfin_rl.seed import set_seed


class DiscretizedEnv:
    def __init__(self, env, bins, render: bool = False):
        self.env = env
        self.bins = bins
        self.action_space = env.action_space
        self.render = render
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

    def _render(self):
        if not self.render:
            return
        if hasattr(self.env, "render"):
            self.env.render()
        elif hasattr(self.env, "env") and hasattr(self.env.env, "render"):
            self.env.env.render()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._render()
        return self._disc(obs), reward, done, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--steps-per-batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--bins-per-dim", type=int, default=4)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    render_mode = "human" if args.render else None
    base_env = make_gymnasium("CartPole-v1", render_mode=render_mode)
    b = args.bins_per_dim
    bins = [(-2.4, 2.4, b), (-3.0, 3.0, b), (-0.5, 0.5, b), (-2.0, 2.0, b)]
    env = DiscretizedEnv(base_env, bins=bins, render=args.render)
    backend = TinyfinBackend(device="cpu")
    policy = PPOPolicy(backend=backend, obs_dim=env.observation_space.n, action_space=env.action_space)
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        steps_per_batch=args.steps_per_batch,
        epochs=args.epochs,
        lr=args.lr,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
    )
    trainer.train(updates=args.updates)


if __name__ == "__main__":
    main()
