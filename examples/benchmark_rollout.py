import time

from tinyfin_rl import BanditEnv, set_seed
from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.policy_tinyfin import SoftmaxPolicy
from tinyfin_rl.vector_env import VectorEnv
from tinyfin_rl.rollout import rollout_vector


def main():
    set_seed(0)
    backend = TinyfinBackend(device="cpu")
    envs = VectorEnv([BanditEnv([0.1, 0.5, 0.9]) for _ in range(16)])
    policy = SoftmaxPolicy(backend=backend, obs_dim=1, action_space=envs.envs[0].action_space)

    steps = 200
    t0 = time.time()
    batch = rollout_vector(envs, policy, steps=steps)
    dt = time.time() - t0
    print(f"rollout steps={steps * len(envs.envs)} seconds={dt:.3f} steps/s={(steps * len(envs.envs)) / dt:.1f}")
    print(f"sample reward mean={sum(batch.rewards)/len(batch.rewards):.3f}")


if __name__ == "__main__":
    main()
