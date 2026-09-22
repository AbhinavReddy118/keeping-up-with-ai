#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

LOG_FILE="data/daily_$(date +%Y%m%d_%H%M%S).log"
mkdir -p data

echo "[run_daily] starting pipeline at $(date)" | tee -a "$LOG_FILE"
"$PYTHON" main.py --output json 2>&1 | tee -a "$LOG_FILE"
RC=${PIPESTATUS[0]}
echo "[run_daily] finished rc=$RC at $(date)" | tee -a "$LOG_FILE"
exit "$RC"
