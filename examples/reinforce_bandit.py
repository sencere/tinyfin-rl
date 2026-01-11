from _bootstrap import setup_path
setup_path()

from tinyfin_rl import BanditEnv, set_seed
from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.policy_tinyfin import SoftmaxPolicy
from tinyfin_rl.algos import ReinforceTrainer


def main():
    set_seed(0)
    env = BanditEnv([0.1, 0.5, 0.9])
    backend = TinyfinBackend(device="cpu")
    policy = SoftmaxPolicy(backend=backend, obs_dim=1, action_space=env.action_space)
    trainer = ReinforceTrainer(env=env, policy=policy, gamma=0.9, lr=0.1)
    trainer.train(episodes=20)


if __name__ == "__main__":
    main()
