#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/elf/fusion-dashboard"
APP_SCRIPT="$APP_DIR/board_stream_server.py"
LOG_FILE="/tmp/fusion-dashboard.log"

mkdir -p "$(dirname "$LOG_FILE")"
cd "$APP_DIR"

exec /usr/bin/python3 "$APP_SCRIPT" >>"$LOG_FILE" 2>&1
