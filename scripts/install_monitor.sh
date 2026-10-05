#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="$HOME/Library/LaunchAgents"
PLIST_NAME="com.ai-news-digest.monitor.plist"
TARGET_PLIST="$TARGET_DIR/$PLIST_NAME"

echo "=== AI News Digest - Continuous Monitor Launchd Installer ==="
echo "Project Directory: $PROJECT_DIR"
echo "Target Plist:      $TARGET_PLIST"

mkdir -p "$TARGET_DIR"
mkdir -p "$PROJECT_DIR/data"

# Substitute PROJECT_DIR into plist template
sed "s|PROJECT_DIR|$PROJECT_DIR|g" "$SCRIPT_DIR/$PLIST_NAME" > "$TARGET_PLIST"
chmod 644 "$TARGET_PLIST"

echo "Generated LaunchAgent plist at: $TARGET_PLIST"
echo ""
echo "Manage the monitor service with the following commands:"
echo "  Start:   launchctl load $TARGET_PLIST"
echo "  Stop:    launchctl unload $TARGET_PLIST"
echo "  Restart: launchctl unload $TARGET_PLIST && launchctl load $TARGET_PLIST"
echo "  Status:  curl -s http://localhost:5000/status | jq ."
echo "  Logs:    tail -f $PROJECT_DIR/data/monitor.out.log"
echo "           tail -f $PROJECT_DIR/data/monitor.err.log"
