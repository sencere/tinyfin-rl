#!/usr/bin/env sh
set -eu

make
./scripts/run_tests.sh
./scripts/golden_runs.sh
