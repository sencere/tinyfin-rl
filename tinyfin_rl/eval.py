from typing import Dict, List

from .seed import set_seed


def evaluate_policy(env, policy, episodes: int = 10, seed: int = 0) -> Dict:
    set_seed(seed)
    try:
        env.seed(seed)
    except Exception:
        pass
    returns: List[float] = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            action = policy.act(obs)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                obs, reward, done, _ = step_out
            total += float(reward)
        returns.append(total)
    mean_return = sum(returns) / max(len(returns), 1)
    return {"returns": returns, "return_mean": mean_return}
