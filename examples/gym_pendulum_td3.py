from _bootstrap import setup_path
setup_path()

from tinyfin_rl.adapters.gymnasium_adapter import make_gymnasium
from tinyfin_rl.algos.td3 import Actor, Critic, TD3Trainer
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
        if hasattr(self.env, "render"):
            self.env.render()
        elif hasattr(self.env, "env") and hasattr(self.env.env, "render"):
            self.env.env.render()
        return float(obs[0]), reward, done, info


def main():
    set_seed(0)
    base_env = make_gymnasium("Pendulum-v1", render_mode="human")
    env = ScalarObsEnv(base_env)
    backend = TinyfinBackend(device="cpu")
    actor = Actor(backend=backend)
    critic1 = Critic(backend=backend)
    critic2 = Critic(backend=backend)
    actor_t = Actor(backend=backend)
    critic1_t = Critic(backend=backend)
    critic2_t = Critic(backend=backend)
    trainer = TD3Trainer(
        env=env,
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_target=actor_t,
        critic1_target=critic1_t,
        critic2_target=critic2_t,
        replay=ReplayBuffer(capacity=10000),
        action_limit=float(base_env.action_space.high[0]),
    )
    trainer.train(steps=500)


if __name__ == "__main__":
    main()
