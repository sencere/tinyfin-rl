# Determinism

Tinyfin-RL is designed to be deterministic when run with the same binary,
configuration, and seed. This document spells out what is guaranteed and
what can break determinism.

## Deterministic Mode

Use `--deterministic` to force the most stable execution path:

- Forces single-threaded stepping (`--threads 1`).
- If `--seed` is unset (or 0), it defaults to `1` and prints the seed.

This is the recommended mode for regression tests and golden runs.

## What Is Deterministic

With the same CLI flags, seed, and environment count (`--envs`):

- Environment resets use `seed + env_index`.
- C environments use a per-env RNG seeded from the reset seed.
- Action selection is called in fixed index order.
- Replay sampling uses the same RNG sequence for a fixed update order.

If you keep all parameters identical, results should be reproducible.

## What Can Break Determinism

- `--threads > 1` can change execution timing and ordering.
- Changing `--envs`, `--batch-size`, or `--replay-size` changes sampling order.
- Different binaries or compiler settings can change floating-point behavior.
- Aggressive flags like `-ffast-math` or `-march=native` can change numerics.
 - Replay sampling uses deterministic RNG only when `--deterministic` is set.
 - Epsilon policies and action-noise paths use deterministic RNG only when `--deterministic` is set.
 - SAC/TD3 stochastic policy sampling uses deterministic RNG only when `--deterministic` is set.

## Recommended Practice

- Use `--deterministic` for baselines and golden runs.
- Pin `--seed`, `--envs`, and all algorithm hyperparameters.
- Avoid threading when comparing learning curves.
