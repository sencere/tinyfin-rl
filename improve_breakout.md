If I’m designing a “breakout reinforcement” controller for a Game Boy–style implementation (think: tight compute, discrete inputs, pixel grid, simple physics), I’d keep the **observation small + Markov-ish**, the **action space strictly discrete**, and the **reward mostly dense with a tiny amount of shaping** so it learns fast without finding degenerate loops.

Below is a design that works well in practice.

## Observations (obs)

### Minimal but effective (state vector)

Use a fixed-length numeric vector rather than raw pixels:

**Ball**

* `ball_x, ball_y` (normalized to [0,1])
* `ball_vx, ball_vy` (normalized; or just `sign(vx), sign(vy)` if you want super small)

**Paddle**

* `paddle_x` (center x, normalized)
* (optional) `paddle_vx` if your paddle has inertia; otherwise omit

**Bricks**

* If the brick layout is small: a **bitmask / binary grid** flattened (e.g., 10×6 = 60 bits)
* If that’s too big: a compressed form:

  * `bricks_remaining`
  * `nearest_brick_dx, nearest_brick_dy` (relative to ball)
  * or “columns remaining” counts

**Derived features (very helpful)**

* `dx = ball_x - paddle_x`
* `time_to_paddle` estimate (if `ball_vy < 0`: distance to paddle line / |vy|, else 0)
* `predicted_intercept_x` where the ball will cross the paddle y-line (include wall bounces if cheap)

A really strong compact obs is:

* `[ball_x, ball_y, ball_vx, ball_vy, paddle_x, dx, predicted_intercept_x, bricks_bitmask...]`

If you can afford it, **pixels are fine**, but on “Game Boy constraints,” vector features usually learn much faster.

## Actions (action space)

Keep it **3 discrete actions** (classic):

* `0: NOOP`
* `1: LEFT`
* `2: RIGHT`

If your environment supports it and you want finesse:

* Add `FAST_LEFT, FAST_RIGHT` (5 actions) or a “hold duration,” but 3 is the sweet spot for stability.

## Rewards (reward shaping)

### Core reward (sparse objective)

* **+1.0** for each brick destroyed
* **+10.0** for clearing the level
* **-10.0** for losing a life / ball out

This alone works but can be slow early on.

### Add light shaping (dense but safe)

You want shaping that encourages *positioning under the ball* without letting the agent farm reward.

Good shaping terms:

**1) Hit the ball**

* **+0.2 to +1.0** when the paddle hits the ball
  This teaches survival quickly.

**2) Move toward intercept (potential-based)**
Compute `d = |paddle_x - predicted_intercept_x|`.

Use **potential-based shaping** so it doesn’t change the optimal policy:

* Define potential: `Φ(s) = -k * d` (k small, e.g. `k=0.05`)
* Add shaping reward each step:
  `r_shape = γ * Φ(s') - Φ(s)`

This gives a dense “getting warmer” signal that can’t be exploited by oscillation as easily as naive `-d`.

**3) Small time penalty**

* **-0.001 to -0.01 per step**
  Prevents stalling / infinite play when it can farm paddle hits.

**4) Action penalty (optional)**

* **-0.0005** for LEFT/RIGHT, 0 for NOOP
  Encourages minimal movement (nice for classic paddle behavior) but keep tiny.

### Put it together (example)

Per step:

* `r = 1.0 * bricks_destroyed`
* `+ 0.5 if paddle_hit_ball else 0`
* `- 10.0 if life_lost else 0`
* `+ 10.0 if level_cleared else 0`
* `+ (γ*Φ(s') - Φ(s))`  (k=0.05)
* `- 0.002` time penalty
* `- 0.0005 * moved`

This is dense enough to learn quickly, but still anchored to true game success.

## Termination and resets

* Terminate episode on **life lost** or **level cleared** (or after N steps)
* If you have multiple lives, you can either:

  * treat each life as an episode (simpler credit assignment), or
  * keep whole game as episode but give strong negative on each loss

## Common failure modes and fixes

* **Learns to “juggle” ball and ignore bricks:** increase brick reward relative to paddle-hit reward, add step penalty.
* **Oscillation exploit with distance shaping:** use **potential-based** shaping (above), not raw `-distance`.
* **Can’t generalize across layouts:** include brick grid/bitmask, not just `bricks_remaining`.

## If you want the ultra-compact “Game Boy mode”

Obs:

* `ball_x, ball_y` (8-bit each)
* `ball_dir_x, ball_dir_y` (2 bits)
* `paddle_x` (8-bit)
* `predicted_intercept_x` (8-bit)
* `brick_bitmask` (packed bytes)

Actions: {NOOP, LEFT, RIGHT}
Rewards: brick (+1), life lost (-10), clear (+10), potential shaping on intercept, small step penalty.

If you tell me your **brick grid size**, whether the **ball speed/angles are discrete**, and whether you’re using **DQN/PPO**, I can tune the exact coefficients and the obs packing for your memory budget.
