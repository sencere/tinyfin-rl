#!/usr/bin/env sh
set -eu

make

run_case() {
    name="$1"
    cmd="$2"
    threshold="$3"
    output=$(sh -c "$cmd")
    last_return=$(printf "%s\n" "$output" | awk -F'return=' '/return=/{v=$2} END{print v+0}')
    if ! awk "BEGIN { exit !($last_return >= $threshold) }"; then
        printf "golden run failed: %s return=%s threshold=%s\n" "$name" "$last_return" "$threshold" >&2
        return 1
    fi
    printf "golden run ok: %s return=%s\n" "$name" "$last_return"
}

run_case "lineworld_dqn" "./build/tinyfin-rl train --algo dqn --env lineworld --steps 400 --log-every 1 --deterministic --seed 1" 0.2
run_case "lineworld_cont_dqn" "./build/tinyfin-rl train --algo dqn --env lineworld_cont --steps 400 --log-every 1 --deterministic --seed 1" 0.2
run_case "lineworld_mcts" "./build/tinyfin-rl train --algo mcts --env lineworld --steps 200 --log-every 1 --deterministic --seed 1 --mcts-sims 200 --mcts-depth 40" 0.2
