#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

mkdir -p data
echo "[run_monitor] Launching continuous monitor at $(date)"
exec "$PYTHON" monitor.py "$@"
