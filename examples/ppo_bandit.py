from tinyfin_rl import BanditEnv, set_seed
from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.policy_tinyfin import PPOPolicy
from tinyfin_rl.algos import PPOTrainer


def main():
    set_seed(0)
    env = BanditEnv([0.1, 0.5, 0.9])
    backend = TinyfinBackend(device="cpu")
    policy = PPOPolicy(backend=backend, obs_dim=1, action_space=env.action_space)
    trainer = PPOTrainer(env=env, policy=policy, gamma=0.9, lr=0.05, clip_eps=0.2, steps_per_batch=20)
    trainer.train(updates=10)


if __name__ == "__main__":
    main()
