from pathlib import Path
import os
import sys

import paramiko


HOST = os.environ.get("BOARD_HOST", "rk3588.local")
USER = os.environ.get("BOARD_USER", "elf")
PASSWORD = (os.environ.get("BOARD_PASSWORD") or "").strip() or None
REMOTE_DIR = os.environ.get("BOARD_REMOTE_DIR", "/home/elf/fusion-dashboard")

COMMANDS = {
    "start": f"bash {REMOTE_DIR}/start_fusion_dashboard_remote.sh",
    "stop": f"bash {REMOTE_DIR}/stop_fusion_dashboard_remote.sh",
    "restart": f"bash {REMOTE_DIR}/restart_fusion_dashboard_remote.sh",
    "status": (
        "systemctl --user --no-pager --full status fusion-dashboard.service | head -n 20; "
        "echo; curl -s http://127.0.0.1:8765/api/status || true; "
        "echo; tail -n 40 /tmp/fusion-dashboard-browser.log 2>/dev/null || true"
    ),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: python remote_control.py [start|stop|restart|status] [host]")
        raise SystemExit(1)

    action = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else HOST

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=USER, password=PASSWORD, timeout=12, banner_timeout=12, auth_timeout=12)
    try:
        stdin, stdout, stderr = ssh.exec_command(COMMANDS[action], timeout=90)
        sys.stdout.write(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err.strip():
            sys.stderr.write(err)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
