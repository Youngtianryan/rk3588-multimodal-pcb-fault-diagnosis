#!/usr/bin/env bash
set -euo pipefail

echo "========== 重启 Fusion Dashboard =========="
/home/elf/fusion-dashboard/restart_fusion_dashboard_remote.sh
echo
read -r -p "重启完成，按回车关闭终端..." _
