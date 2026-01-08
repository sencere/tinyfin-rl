import sys

from tinyfin_rl import Agent, RandomPolicy, Trainer


def main():
    try:
        import gymnasium as gym
    except ImportError:
        print("gymnasium is required for this example: pip install gymnasium")
        sys.exit(1)

    env = gym.make("CartPole-v1")
    agent = Agent(RandomPolicy(env.action_space))
    trainer = Trainer(env, agent)
    trainer.train(episodes=2)


if __name__ == "__main__":
    main()
