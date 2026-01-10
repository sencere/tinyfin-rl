#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${root_dir}"

make train_q
./train_q --episodes 2000 --report 200 --out q_table.csv

make
./first_env
