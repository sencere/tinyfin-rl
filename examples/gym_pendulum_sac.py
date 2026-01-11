from _bootstrap import setup_path
setup_path()

from tinyfin_rl.adapters.gymnasium_adapter import make_gymnasium
from tinyfin_rl.algos.sac import SACActor, SACCritic, SACTrainer
from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.replay import ReplayBuffer
from tinyfin_rl.seed import set_seed


class ScalarObsEnv:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return float(obs[0])

    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        return float(obs[0]), reward, done, info


def main():
    set_seed(0)
    base_env = make_gymnasium("Pendulum-v1")
    env = ScalarObsEnv(base_env)
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
        action_limit=float(base_env.action_space.high[0]),
    )
    trainer.train(steps=500)


if __name__ == "__main__":
    main()
