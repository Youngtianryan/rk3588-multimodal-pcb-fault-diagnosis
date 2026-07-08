#!/usr/bin/env bash
set -euo pipefail

echo "========== 启动 Fusion Dashboard =========="
/home/elf/fusion-dashboard/start_fusion_dashboard_remote.sh
echo
read -r -p "启动完成，按回车关闭终端..." _
