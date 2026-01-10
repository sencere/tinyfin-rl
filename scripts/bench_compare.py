#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", choices=["list", "prealloc", "array"])
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--min-steps-per-s", type=float)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--targets", type=str, default="benchmarks/rollout_targets.json")
    args = parser.parse_args()

    if args.storage is None or args.num_envs is None or args.steps is None or args.min_steps_per_s is None:
        with open(args.targets, "r") as handle:
            targets = json.load(handle)
        args.storage = args.storage or targets.get("storage", "array")
        args.num_envs = args.num_envs or int(targets.get("num_envs", 64))
        args.steps = args.steps or int(targets.get("steps_per_env", 200))
        args.min_steps_per_s = args.min_steps_per_s or float(targets.get("min_steps_per_s", 0.0))

    cmd = [
        sys.executable,
        "examples/benchmark_rollout.py",
        "--storage",
        args.storage,
        "--num-envs",
        str(args.num_envs),
        "--steps",
        str(args.steps),
        "--format",
        "json",
        "--min-steps-per-s",
        str(args.min_steps_per_s),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode not in (0, 2):
        sys.stderr.write(result.stderr)
        return result.returncode
    payload = json.loads(result.stdout.strip())
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(payload, handle)
    meets = payload.get("meets_threshold", True)
    if not meets:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
