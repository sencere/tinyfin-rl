from _bootstrap import setup_path
setup_path()

import argparse

from tinyfin_rl import BanditEnv, set_seed
from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.policy_tinyfin import PPOPolicy
from tinyfin_rl.algos import PPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--steps-per-batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    env = BanditEnv([0.1, 0.5, 0.9])
    backend = TinyfinBackend(device="cpu")
    policy = PPOPolicy(backend=backend, obs_dim=1, action_space=env.action_space)
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        gamma=args.gamma,
        lr=args.lr,
        clip_eps=args.clip_eps,
        steps_per_batch=args.steps_per_batch,
        epochs=args.epochs,
        entropy_coef=args.entropy_coef,
    )
    trainer.train(updates=args.updates)


if __name__ == "__main__":
    main()
