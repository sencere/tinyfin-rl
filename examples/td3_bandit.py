from _bootstrap import setup_path
setup_path()

from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.envs import ContinuousBanditEnv
from tinyfin_rl.algos.td3 import Actor, Critic, TD3Trainer
from tinyfin_rl.replay import ReplayBuffer
from tinyfin_rl.seed import set_seed


def main():
    set_seed(0)
    backend = TinyfinBackend(device="cpu")
    env = ContinuousBanditEnv(target=0.7)
    actor = Actor(backend=backend)
    critic1 = Critic(backend=backend)
    critic2 = Critic(backend=backend)
    actor_t = Actor(backend=backend)
    critic1_t = Critic(backend=backend)
    critic2_t = Critic(backend=backend)
    replay = ReplayBuffer(capacity=1000)
    trainer = TD3Trainer(
        env=env,
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_target=actor_t,
        critic1_target=critic1_t,
        critic2_target=critic2_t,
        replay=replay,
        batch_size=16,
    )
    trainer.train(steps=300)
    action = actor.act(0.0, noise_std=0.0)
    print(f"action={action:.3f}")


if __name__ == "__main__":
    main()
