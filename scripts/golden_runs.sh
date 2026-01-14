#!/usr/bin/env sh
set -eu

make

run_case() {
    name="$1"
    cmd="$2"
    last_threshold="$3"
    peak_threshold="$4"
    min_reports="$5"
    output=$(sh -c "$cmd")
    last_return=$(printf "%s\n" "$output" | awk -F'return=' '/return=/{v=$2} END{print v+0}')
    peak_return=$(printf "%s\n" "$output" | awk -F'return=' '/return=/{if ($2>m) m=$2} END{print m+0}')
    report_count=$(printf "%s\n" "$output" | awk -F'return=' '/return=/{c++} END{print c+0}')
    if ! awk "BEGIN { exit !($report_count >= $min_reports) }"; then
        printf "golden run failed: %s reports=%s min=%s\n" "$name" "$report_count" "$min_reports" >&2
        return 1
    fi
    if ! awk "BEGIN { exit !($last_return >= $last_threshold) }"; then
        printf "golden run failed: %s return=%s threshold=%s\n" "$name" "$last_return" "$last_threshold" >&2
        return 1
    fi
    if ! awk "BEGIN { exit !($peak_return >= $peak_threshold) }"; then
        printf "golden run failed: %s peak=%s threshold=%s\n" "$name" "$peak_return" "$peak_threshold" >&2
        return 1
    fi
    printf "golden run ok: %s return=%s peak=%s\n" "$name" "$last_return" "$peak_return"
}

run_case "lineworld_dqn" "./build/tinyfin-rl train --algo dqn --env lineworld --steps 400 --log-every 1 --deterministic --seed 1" 0.2 0.2 1
run_case "lineworld_duo_dqn" "./build/tinyfin-rl train --algo dqn --env lineworld_duo --steps 400 --log-every 1 --deterministic --seed 1 --agents 2 --share-policy --replay-size 2000 --batch-size 32" 0.2 0.2 1
run_case "lineworld_rainbow" "./build/tinyfin-rl train --algo rainbow --env lineworld --steps 400 --log-every 1 --deterministic --seed 1 --replay-size 2000 --batch-size 32" 0.2 0.2 1
run_case "lineworld_qrdqn" "./build/tinyfin-rl train --algo qrdqn --env lineworld --steps 400 --log-every 1 --deterministic --seed 1 --replay-size 2000 --batch-size 32" 0.2 0.2 1
run_case "lineworld_cont_dqn" "./build/tinyfin-rl train --algo dqn --env lineworld_cont --steps 400 --log-every 1 --deterministic --seed 1" 0.2 0.2 1
run_case "lineworld_mcts" "./build/tinyfin-rl train --algo mcts --env lineworld --steps 200 --log-every 1 --deterministic --seed 1 --mcts-sims 200 --mcts-depth 40" 0.2 0.2 1
run_case "lineworld_ppo" "./build/tinyfin-rl train --algo ppo --env lineworld --steps 400 --log-every 1 --deterministic --seed 1 --steps-per-batch 64 --epochs 2 --clip-eps 0.2 --gae-lambda 0.95 --entropy-coef 0.01" -0.2 0.0 1
run_case "lineworld_ppo_clip01" "./build/tinyfin-rl train --algo ppo --env lineworld --steps 400 --log-every 1 --deterministic --seed 1 --steps-per-batch 64 --epochs 2 --clip-eps 0.1 --gae-lambda 0.95 --entropy-coef 0.01" -0.2 0.0 1
run_case "maze_rooms_dqn" "./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 600 --log-every 1 --deterministic --seed 1" -0.2 -0.2 1
run_case "point1d_sac" "./build/tinyfin-rl train --algo sac --env point1d --steps 600 --log-every 1 --deterministic --seed 1 --replay-size 2000 --batch-size 32" -0.2 -0.2 1
run_case "point1d_td3" "./build/tinyfin-rl train --algo td3 --env point1d --steps 600 --log-every 1 --deterministic --seed 1 --replay-size 2000 --batch-size 32" -0.2 -0.2 1
run_case "coin_maze_duo_dqn" "./build/tinyfin-rl train --algo dqn --env coin_maze_duo --steps 600 --log-every 1 --deterministic --seed 1 --replay-size 2000 --batch-size 32 --share-policy" -0.2 -0.2 1
