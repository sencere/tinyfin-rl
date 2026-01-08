# API conventions

This document describes the conventions tinyfin-rl follows today and the expectations for future extensions.

## Environments

- `reset()` returns either `obs` or `(obs, info)`.
- `step(action)` returns either `(obs, reward, done, info)` or `(obs, reward, terminated, truncated, info)`.
- `reward` is a float and `done` is a boolean.

## Spaces

- `Discrete(n)` expects integer actions in `[0, n)`.
- `Box(low, high, shape)` expects a flat iterable matching the product of `shape`.

## Policies

- `Policy.act(observation)` returns a valid action for the environment's action space.
- Policies may be stateless or stateful; rollout state management will be added as needed.

## Agents

- `Agent.observe(transition)` receives `(obs, action, reward, next_obs, done)`.
- Agents may store transitions and update parameters between episodes.

## Trainer

- `Trainer.train(episodes)` runs complete episodes and logs returns/steps.
- Optional `Visualizer` hooks are called on reset and step; these will migrate to C/Raylib.

## Tinyfin integration

- Tinyfin is optional and lives in a separate repository.
- The `CBackend` loader can load Tinyfin shared libraries when available, or be replaced by another tensor backend.
