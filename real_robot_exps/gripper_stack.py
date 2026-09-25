"""Kill and restart the lfd_apples gripper controller stack.

The suction gripper's valve and finger stepper are driven by an ESP32 running
micro-ROS over a Wi-Fi hotspot this laptop hosts (see
lfd_apples_ws/src/lfd_apples/launch/lfd_gripper.launch.py). Two things make the
gripper "very finicky":

1. Fast robot motion (e.g. handeye_auto_calibrate's free-drive-to-pose moves, or a
   fast pull) can jostle the ESP32 or its antenna enough to drop the Wi-Fi link for
   a moment; whatever gripper_grab / toggle_valve call was in flight then gets no
   reply and the client call hangs until its timeout.
2. A previous launch that was never cleanly stopped (closed terminal, killed
   session, crashed field_session run) leaves its processes running. The next
   launch then runs *alongside* the old one: two ``automatic_gripper`` nodes with
   the same name, and worse, two ``micro_ros_agent`` processes both bound to the
   same UDP port, splitting the ESP32's traffic between them. Roughly half of any
   given request then goes to the "wrong" agent and is silently dropped, which
   looks exactly like intermittent flakiness. This was observed directly: two
   ``lfd_automatic_gripper`` processes and two ``micro_ros_agent udp4 --port 8888``
   processes (from two different workspaces) running at once.

kill_stray_gripper_processes() + launch_gripper_stack() give a clean, repeatable
restart: always kill first, never trust whatever is already running.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from real_robot_exps.field_session import Proc

DEFAULT_SSID = "alejos"
DEFAULT_PASSWORD = "harvesting"

# Matched against the full command line (pkill -f), not just an executable name,
# so a micro_ros_agent started from any workspace sharing this port is caught too
# -- that cross-workspace duplicate is exactly what caused the flakiness above.
GRIPPER_PROCESS_PATTERNS = (
    "lfd_automatic_gripper",
    "lfd_apples lfd_gripper.launch.py",
    "micro_ros_agent.*--port 8888",
)


def kill_stray_gripper_processes(*, wait_s: float = 1.5) -> None:
    """pkill everything matching GRIPPER_PROCESS_PATTERNS. Safe to call when none are running."""
    for pattern in GRIPPER_PROCESS_PATTERNS:
        subprocess.run(["pkill", "-f", pattern], capture_output=True)
    time.sleep(wait_s)
    # A second, forceful pass for anything that ignored SIGTERM.
    for pattern in GRIPPER_PROCESS_PATTERNS:
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)


def launch_gripper_stack(
    ros_ws: Path,
    log_path: Path,
    *,
    ssid: str = DEFAULT_SSID,
    password: str = DEFAULT_PASSWORD,
    env: dict[str, str] | None = None,
) -> "Proc":
    """Start ``ros2 launch lfd_apples lfd_gripper.launch.py ssid:=... password:=...`` in the background.

    Call kill_stray_gripper_processes() first -- this does not check for or kill
    anything already running.
    """
    from real_robot_exps.field_session import Proc, ros_command  # deferred: avoid an import cycle

    cmd = ros_command(
        f"ros2 launch lfd_apples lfd_gripper.launch.py ssid:={ssid} password:={password}", ros_ws
    )
    return Proc("gripper_stack", cmd, log_path, env=env)


def gripper_stack_ready(*, mock: bool = False, timeout_s: float = 30.0) -> tuple[bool, str | None]:
    """True once gripper_grab answers.

    The controller node only registers gripper_grab after its toggle_valve /
    move_stepper clients connect to the ESP32 (see lfd_automatic_gripper.py
    initialize_ros_service_clients), so gripper_grab responding is already
    confirmation that the ESP32 is on the network and bridged.
    """
    from real_robot_exps.gripper_test import GripperClient

    try:
        client = GripperClient(mock=mock, timeout_s=timeout_s)
    except Exception as exc:
        return False, str(exc)
    client.terminate()
    return True, None
