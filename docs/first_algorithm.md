# First algorithm: REINFORCE sketch

This walkthrough describes the minimal structure for a REINFORCE-style policy gradient loop as it will be implemented in C modules. It is an algorithm-level sketch intended to guide the upcoming implementation.

## Key pieces

- A policy that maps observations to action distributions.
- A rollout buffer that stores trajectories (obs, action, reward).
- A return computation over each episode.

## Flow outline

1. Collect a full episode trajectory.
2. Compute returns (discounted sum of rewards).
3. Compute policy gradient loss.
4. Apply optimizer step using the tensor backend.
5. Clear rollout storage and repeat.

## Notes

- The future implementation will likely extend the trainer with per-episode hooks so `agent_update()` can run at the end of each rollout.
- Advantage normalization, baseline/value heads, and entropy bonuses will be added as part of Milestone 1.
