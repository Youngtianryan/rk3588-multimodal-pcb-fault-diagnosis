#!/usr/bin/env bash
set -euo pipefail

echo "========== 关闭 Fusion Dashboard =========="
/home/elf/fusion-dashboard/stop_fusion_dashboard_remote.sh
echo
read -r -p "关闭完成，按回车关闭终端..." _
