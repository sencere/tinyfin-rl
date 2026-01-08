from tinyfin_rl import GridworldEnv, RandomPolicy, Agent, Trainer, set_seed


def main():
    set_seed(123)
    env = GridworldEnv(width=5, height=5, max_steps=30)
    agent = Agent(RandomPolicy(env.action_space))
    trainer = Trainer(env, agent)
    trainer.train(episodes=3)


if __name__ == "__main__":
    main()
