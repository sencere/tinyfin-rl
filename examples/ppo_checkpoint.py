from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.envs import BanditEnv
from tinyfin_rl.policy_tinyfin import PPOPolicy
from tinyfin_rl.algos.ppo import PPOTrainer
from tinyfin_rl.spaces import Discrete
from tinyfin_rl.seed import set_seed
from tinyfin_rl.checkpoint import save_policy_checkpoint, load_policy_checkpoint


def main():
    set_seed(0)
    backend = TinyfinBackend(device="cpu")
    env = BanditEnv([0.2, 0.8])
    policy = PPOPolicy(backend=backend, obs_dim=1, action_space=Discrete(2))
    trainer = PPOTrainer(env=env, policy=policy, steps_per_batch=8, epochs=1)
    trainer.train(updates=2)
    save_policy_checkpoint("ppo_checkpoint.npz", policy, meta={"algo": "ppo"})

    policy2 = PPOPolicy(backend=backend, obs_dim=1, action_space=Discrete(2))
    load_policy_checkpoint("ppo_checkpoint.npz", policy2)
    trainer2 = PPOTrainer(env=env, policy=policy2, steps_per_batch=8, epochs=1)
    trainer2.train(updates=2)


if __name__ == "__main__":
    main()
