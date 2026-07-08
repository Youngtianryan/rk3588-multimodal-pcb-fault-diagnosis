#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/elf/fusion-dashboard"
APP_SCRIPT="$APP_DIR/board_stream_server.py"
RUNNER_SCRIPT="$APP_DIR/run_fusion_dashboard_service.sh"
SERVICE_FILE_SOURCE="$APP_DIR/fusion-dashboard.service"
SERVICE_DIR="/home/elf/.config/systemd/user"
SERVICE_FILE_TARGET="$SERVICE_DIR/fusion-dashboard.service"
LOG_FILE="/tmp/fusion-dashboard.log"
DASHBOARD_URL="${DASHBOARD_URL:-http://127.0.0.1:8765}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"
SYSTEMCTL="systemctl --user"

choose_xauthority() {
  for candidate in \
    "${XAUTHORITY:-}" \
    "/run/user/1000/gdm/Xauthority" \
    "/home/elf/.Xauthority"
  do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ ! -f "$APP_SCRIPT" ]]; then
  echo "missing: $APP_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$RUNNER_SCRIPT" ]]; then
  echo "missing: $RUNNER_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$SERVICE_FILE_SOURCE" ]]; then
  echo "missing: $SERVICE_FILE_SOURCE" >&2
  exit 1
fi

mkdir -p "$SERVICE_DIR"
install -m 644 "$SERVICE_FILE_SOURCE" "$SERVICE_FILE_TARGET"
chmod +x "$RUNNER_SCRIPT"

$SYSTEMCTL stop fusion-dashboard.service >/dev/null 2>&1 || true
pkill -9 -f "[b]oard_stream_server.py" >/dev/null 2>&1 || true
fuser -k -n tcp 8765 >/dev/null 2>&1 || true
pkill -9 -f chromium-browser >/dev/null 2>&1 || true
pkill -9 -f chrome_crashpad_handler >/dev/null 2>&1 || true
sleep 1

$SYSTEMCTL daemon-reload
$SYSTEMCTL start fusion-dashboard.service
sleep 4

echo "started: $APP_SCRIPT"
echo "log: $LOG_FILE"
echo "action: start"
${SYSTEMCTL} --no-pager --full status fusion-dashboard.service | head -n 20 || true
curl -fsS --max-time 8 "$DASHBOARD_URL/api/status" || true

if [[ "$OPEN_BROWSER" != "0" ]]; then
  export DISPLAY="${DISPLAY:-:0}"
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/1000}"
  export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}"
  chosen_xauth="$(choose_xauthority || true)"
  if [[ -n "$chosen_xauth" ]]; then
    export XAUTHORITY="$chosen_xauth"
  fi
  if command -v chromium-browser >/dev/null 2>&1; then
    nohup chromium-browser \
      --user-data-dir=/home/elf/.config/chromium-fusion \
      --no-first-run \
      --no-default-browser-check \
      --disable-session-crashed-bubble \
      --disable-infobars \
      --start-maximized \
      --new-window \
      "$DASHBOARD_URL" >/tmp/fusion-dashboard-browser.log 2>&1 </dev/null &
  elif command -v chromium >/dev/null 2>&1; then
    nohup chromium \
      --user-data-dir=/home/elf/.config/chromium-fusion \
      --no-first-run \
      --no-default-browser-check \
      --disable-session-crashed-bubble \
      --disable-infobars \
      --start-maximized \
      --new-window \
      "$DASHBOARD_URL" >/tmp/fusion-dashboard-browser.log 2>&1 </dev/null &
  elif command -v xdg-open >/dev/null 2>&1; then
    nohup xdg-open "$DASHBOARD_URL" >/tmp/fusion-dashboard-browser.log 2>&1 </dev/null &
  fi
fi
echo
echo "browser: $DASHBOARD_URL"
