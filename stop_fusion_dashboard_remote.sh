#!/usr/bin/env bash
set -euo pipefail

SYSTEMCTL="systemctl --user"

$SYSTEMCTL stop fusion-dashboard.service >/dev/null 2>&1 || true
pkill -9 -f "[b]oard_stream_server.py" >/dev/null 2>&1 || true
fuser -k -n tcp 8765 >/dev/null 2>&1 || true
pkill -9 -f chromium-browser >/dev/null 2>&1 || true
pkill -9 -f chrome_crashpad_handler >/dev/null 2>&1 || true
sleep 1

echo "fusion dashboard stopped"
echo "action: stop"
