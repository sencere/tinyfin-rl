from _bootstrap import setup_path
setup_path()

from tinyfin_rl import BanditEnv, set_seed
from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.policy_tinyfin import PPOPolicy
from tinyfin_rl.algos import PPOTrainer
from tinyfin_rl.vector_env import VectorEnv


def main():
    set_seed(0)
    envs = VectorEnv([BanditEnv([0.1, 0.5, 0.9]) for _ in range(8)])
    backend = TinyfinBackend(device="cpu")
    policy = PPOPolicy(backend=backend, obs_dim=1, action_space=envs.envs[0].action_space)
    trainer = PPOTrainer(env=envs.envs[0], policy=policy, vector_env=envs, gamma=0.9, lr=0.05, steps_per_batch=16)
    trainer.train(updates=10)


if __name__ == "__main__":
    main()
