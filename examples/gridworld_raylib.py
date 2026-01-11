from _bootstrap import setup_path
setup_path()

from tinyfin_rl import Agent, GridworldEnv, RandomPolicy, RaylibGridworldVisualizer, Trainer


def main():
    env = GridworldEnv(width=6, height=6, max_steps=60)
    agent = Agent(RandomPolicy(env.action_space))
    visualizer = RaylibGridworldVisualizer(cell_size=64, fps=30)
    trainer = Trainer(env, agent, visualizer=visualizer)
    trainer.train(episodes=2)


if __name__ == "__main__":
    main()
