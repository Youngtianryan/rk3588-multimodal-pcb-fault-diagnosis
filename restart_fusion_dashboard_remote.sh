#!/usr/bin/env bash
set -euo pipefail

echo "action: restart"
/home/elf/fusion-dashboard/stop_fusion_dashboard_remote.sh
exec /home/elf/fusion-dashboard/start_fusion_dashboard_remote.sh
