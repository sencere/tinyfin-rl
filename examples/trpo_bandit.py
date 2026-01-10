from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.policy_tinyfin import PPOPolicy
from tinyfin_rl.algos.trpo import TRPOTrainer
from tinyfin_rl.envs import BanditEnv
from tinyfin_rl.seed import set_seed
from tinyfin_rl.spaces import Discrete


def main():
    set_seed(0)
    backend = TinyfinBackend(device="cpu")
    env = BanditEnv([0.2, 0.8])
    policy = PPOPolicy(backend=backend, obs_dim=1, action_space=Discrete(2))
    trainer = TRPOTrainer(env=env, policy=policy)
    trainer.train(updates=5)


if __name__ == "__main__":
    main()
