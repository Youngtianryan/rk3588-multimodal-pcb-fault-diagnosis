from pathlib import Path
import os
import sys

import paramiko


HOST = os.environ.get("BOARD_HOST", "rk3588.local")
USER = os.environ.get("BOARD_USER", "elf")
PASSWORD = (os.environ.get("BOARD_PASSWORD") or "").strip() or None
REMOTE_DIR = os.environ.get("BOARD_REMOTE_DIR", "/home/elf/fusion-dashboard")


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else HOST
    local_root = Path(__file__).resolve().parent
    files = [
        "board_stream_server.py",
        "app.js",
        "index.html",
        "styles.css",
        "start_fusion_dashboard_remote.sh",
        "stop_fusion_dashboard_remote.sh",
        "restart_fusion_dashboard_remote.sh",
        "run_fusion_dashboard_service.sh",
        "fusion-dashboard.service",
        "board_desktop_launcher.sh",
        "board_desktop_stop.sh",
        "board_desktop_restart.sh",
        "fusion-dashboard.desktop",
        "fusion-dashboard-stop.desktop",
        "fusion-dashboard-restart.desktop",
    ]

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    optional_files = [
        "llm_config.json",
    ]

    ssh.connect(host, username=USER, password=PASSWORD, timeout=12, banner_timeout=12, auth_timeout=12)

    try:
        sftp = ssh.open_sftp()
        try:
            upload_files = list(files)
            for name in optional_files:
                if (local_root / name).exists():
                    upload_files.append(name)
            for name in upload_files:
                local_path = local_root / name
                remote_path = f"{REMOTE_DIR}/{name}"
                print(f"upload {name} -> {remote_path}")
                sftp.put(str(local_path), remote_path)
        finally:
            sftp.close()

        joined = (
            f"cd {REMOTE_DIR}; "
            f"chmod +x {REMOTE_DIR}/run_fusion_dashboard_service.sh {REMOTE_DIR}/start_fusion_dashboard_remote.sh {REMOTE_DIR}/stop_fusion_dashboard_remote.sh {REMOTE_DIR}/restart_fusion_dashboard_remote.sh {REMOTE_DIR}/board_desktop_launcher.sh {REMOTE_DIR}/board_desktop_stop.sh {REMOTE_DIR}/board_desktop_restart.sh; "
            f"install -m 755 {REMOTE_DIR}/board_desktop_launcher.sh /home/elf/Desktop/board_desktop_launcher.sh; "
            f"install -m 755 {REMOTE_DIR}/board_desktop_stop.sh /home/elf/Desktop/board_desktop_stop.sh; "
            f"install -m 755 {REMOTE_DIR}/board_desktop_restart.sh /home/elf/Desktop/board_desktop_restart.sh; "
            f"install -m 755 {REMOTE_DIR}/fusion-dashboard.desktop /home/elf/Desktop/fusion-dashboard.desktop; "
            f"install -m 755 {REMOTE_DIR}/fusion-dashboard-stop.desktop /home/elf/Desktop/fusion-dashboard-stop.desktop; "
            f"install -m 755 {REMOTE_DIR}/fusion-dashboard-restart.desktop /home/elf/Desktop/fusion-dashboard-restart.desktop; "
            f"mkdir -p /home/elf/.config/systemd/user; "
            f"install -m 644 {REMOTE_DIR}/fusion-dashboard.service /home/elf/.config/systemd/user/fusion-dashboard.service; "
            f"systemctl --user daemon-reload; "
            f"bash {REMOTE_DIR}/restart_fusion_dashboard_remote.sh; "
            f"sleep 5; "
            f"systemctl --user --no-pager --full status fusion-dashboard.service | head -n 20; "
            f"echo; curl -s http://127.0.0.1:8765/api/status; echo; "
            f"tail -n 60 /tmp/fusion-dashboard.log; "
            f"echo; tail -n 40 /tmp/fusion-dashboard-browser.log 2>/dev/null || true"
        )
        stdin, stdout, stderr = ssh.exec_command(joined, timeout=60)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        sys.stdout.write(out)
        if err.strip():
            sys.stderr.write(err)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
