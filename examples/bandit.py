from _bootstrap import setup_path
setup_path()

from tinyfin_rl import BanditEnv, RandomPolicy, Agent, Trainer, set_seed


def main():
    set_seed(42)
    env = BanditEnv([0.1, 0.5, 0.9])
    agent = Agent(RandomPolicy(env.action_space))
    trainer = Trainer(env, agent)
    trainer.train(episodes=5)


if __name__ == "__main__":
    main()
