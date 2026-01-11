from _bootstrap import setup_path
setup_path()

import sys


def main():
    try:
        import gymnasium as gym
    except ImportError:
        print("gymnasium is required for this example: pip install gymnasium")
        sys.exit(1)

    env = gym.make("CartPole-v1", render_mode="human")
    for episode in range(20):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            env.render()
            steps += 1
        print(f"episode={episode} steps={steps}")


if __name__ == "__main__":
    main()
