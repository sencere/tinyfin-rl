#!/usr/bin/env sh

TFRL_HOME="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export TFRL_HOME
export LD_LIBRARY_PATH="$TFRL_HOME/build:$TFRL_HOME/tinyfin:$TFRL_HOME/raylib-src/src:${LD_LIBRARY_PATH:-}"
export PATH="$TFRL_HOME/build:$PATH"

echo "TFRL_HOME=$TFRL_HOME"
echo "PATH includes $TFRL_HOME/build"
echo "LD_LIBRARY_PATH includes $TFRL_HOME/build, $TFRL_HOME/tinyfin, and $TFRL_HOME/raylib-src/src"
