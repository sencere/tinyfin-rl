import argparse
import json
from pathlib import Path
import time

from tinyfin_rl import BanditEnv, set_seed
from tinyfin_rl.backends import TinyfinBackend
from tinyfin_rl.policy_tinyfin import SoftmaxPolicy
from tinyfin_rl.vector_env import VectorEnv
from tinyfin_rl.rollout import rollout_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["text", "json", "csv"], default="text")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    backend = TinyfinBackend(device="cpu")
    envs = VectorEnv([BanditEnv([0.1, 0.5, 0.9]) for _ in range(args.num_envs)])
    policy = SoftmaxPolicy(backend=backend, obs_dim=1, action_space=envs.envs[0].action_space)

    total_steps = args.steps * len(envs.envs)
    for _ in range(max(args.warmup, 0)):
        rollout_vector(envs, policy, steps=args.steps)

    durations = []
    steps_rates = []
    reward_means = []
    for _ in range(max(args.iters, 1)):
        t0 = time.perf_counter()
        batch = rollout_vector(envs, policy, steps=args.steps)
        dt = time.perf_counter() - t0
        durations.append(dt)
        steps_rates.append(total_steps / dt)
        reward_means.append(sum(batch.rewards) / len(batch.rewards))

    mean_seconds = sum(durations) / len(durations)
    mean_steps_per_s = sum(steps_rates) / len(steps_rates)
    mean_reward = sum(reward_means) / len(reward_means)

    if args.format == "json":
        payload = {
            "envs": args.num_envs,
            "steps_per_env": args.steps,
            "steps": total_steps,
            "iters": len(durations),
            "seconds_mean": mean_seconds,
            "seconds_min": min(durations),
            "seconds_max": max(durations),
            "steps_per_s_mean": mean_steps_per_s,
            "steps_per_s_min": min(steps_rates),
            "steps_per_s_max": max(steps_rates),
            "reward_mean": mean_reward,
        }
        out = json.dumps(payload)
    elif args.format == "csv":
        out = "envs,steps_per_env,steps,iters,seconds_mean,seconds_min,seconds_max,steps_per_s_mean,steps_per_s_min,steps_per_s_max,reward_mean\n"
        out += (
            f"{args.num_envs},{args.steps},{total_steps},{len(durations)},"
            f"{mean_seconds:.6f},{min(durations):.6f},{max(durations):.6f},"
            f"{mean_steps_per_s:.3f},{min(steps_rates):.3f},{max(steps_rates):.3f},"
            f"{mean_reward:.6f}\n"
        )
    else:
        out = (
            f"rollout envs={args.num_envs} steps_per_env={args.steps} iters={len(durations)} "
            f"steps={total_steps}\n"
            f"seconds mean={mean_seconds:.3f} min={min(durations):.3f} max={max(durations):.3f}\n"
            f"steps/s mean={mean_steps_per_s:.1f} min={min(steps_rates):.1f} max={max(steps_rates):.1f}\n"
            f"reward mean={mean_reward:.3f}\n"
        )

    if args.output:
        Path(args.output).write_text(out)
    else:
        print(out, end="")


if __name__ == "__main__":
    main()
