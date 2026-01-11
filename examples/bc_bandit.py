from _bootstrap import setup_path
setup_path()

from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.policy_tinyfin import SoftmaxPolicy
from tinyfin_rl.algos.bc import BCTrainer
from tinyfin_rl.envs import BanditEnv
from tinyfin_rl.seed import set_seed
from tinyfin_rl.spaces import Discrete


def collect_dataset(env, action: int, episodes: int = 200):
    data = []
    for _ in range(episodes):
        obs = env.reset()
        next_obs, reward, done, _ = env.step(action)
        data.append((obs, action, reward, next_obs, done))
    return data


def main():
    set_seed(0)
    backend = TinyfinBackend(device="cpu")
    env = BanditEnv([0.1, 0.9])
    dataset = collect_dataset(env, action=1, episodes=200)

    policy = SoftmaxPolicy(backend=backend, obs_dim=1, action_space=Discrete(2))
    trainer = BCTrainer(policy=policy, epochs=5, lr=0.2)
    trainer.train(dataset)

    correct = 0
    for _ in range(200):
        action = policy.act(0)
        if action == 1:
            correct += 1
    print(f"policy_action1_rate={correct/200:.3f}")


if __name__ == "__main__":
    main()
