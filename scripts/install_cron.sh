#!/usr/bin/env bash
set -euo pipefail

PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/com.ai-news-digest.daily.plist"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_DST="$HOME/Library/LaunchAgents/com.ai-news-digest.daily.plist"

mkdir -p "$HOME/Library/LaunchAgents"

sed "s|PROJECT_DIR|$PROJECT_DIR|g" "$PLIST_SRC" > "$PLIST_DST"
chmod 644 "$PLIST_DST"

RUN_SCRIPT="$PROJECT_DIR/run_daily.sh"
chmod +x "$RUN_SCRIPT"

launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

echo "Installed daily pipeline schedule."
echo "Plist:  $PLIST_DST"
echo "Script: $RUN_SCRIPT"
echo "Logs:   $PROJECT_DIR/data/daily.*.log"
echo
echo "Next run: 09:00 Asia/Kolkata."
echo "Test now:  launchctl start com.ai-news-digest.daily"
echo "Uninstall: launchctl unload $PLIST_DST && rm $PLIST_DST"
