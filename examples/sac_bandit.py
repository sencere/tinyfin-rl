from tinyfin_rl.algos.sac import SACActor, SACCritic, SACTrainer
from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.envs import ContinuousBanditEnv
from tinyfin_rl.replay import ReplayBuffer
from tinyfin_rl.seed import set_seed


def main():
    set_seed(0)
    env = ContinuousBanditEnv()
    backend = TinyfinBackend(device="cpu")
    actor = SACActor(backend=backend)
    critic1 = SACCritic(backend=backend)
    critic2 = SACCritic(backend=backend)
    critic1_t = SACCritic(backend=backend)
    critic2_t = SACCritic(backend=backend)
    trainer = SACTrainer(
        env=env,
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        critic1_target=critic1_t,
        critic2_target=critic2_t,
        replay=ReplayBuffer(capacity=10000),
        action_limit=float(env.action_space.high),
    )
    trainer.train(steps=500)


if __name__ == "__main__":
    main()
