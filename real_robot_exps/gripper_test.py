"""Small ROS2 gripper client with an optional no-op mock mode.

    python -m real_robot_exps.gripper_test close     # air on, then fingers out (gripper_grab)
    python -m real_robot_exps.gripper_test open      # fingers in, then air off (gripper_grab)
    python -m real_robot_exps.gripper_test air-on    # air only, fingers untouched
    python -m real_robot_exps.gripper_test air-off

close/open go through the gripper_controller node's ``gripper_grab`` service;
air-on/air-off call the micro-ROS valve service directly, so they only need the
micro-ROS agent.
"""

from __future__ import annotations

import argparse
import sys
import time

GRAB_SERVICE = "gripper_grab"
VALVE_SERVICE = "/microROS/toggle_valve"


class GripperClient:
    """Connect to a gripper SetBool service, or act as a no-op mock.

    ``service`` defaults to ``gripper_grab`` (air + fingers); pass
    ``VALVE_SERVICE`` to switch only the air.
    """

    def __init__(self, mock: bool = False, timeout_s: float | None = None, service: str = GRAB_SERVICE):
        """``timeout_s``: give up waiting for the service (None = wait forever)."""
        self._mock = bool(mock)
        self._service = service
        self._rclpy = None
        self._node = None
        self._client = None
        self._setbool = None
        if self._mock:
            print(f"[MockGripper] {service} connection disabled; using no-op client.")
            return

        import rclpy
        from rclpy.node import Node
        from std_srvs.srv import SetBool

        self._rclpy = rclpy
        self._setbool = SetBool
        rclpy.init()
        self._node = Node("gripper_grab_client" if service == GRAB_SERVICE else "gripper_valve_client")
        self._client = self._node.create_client(SetBool, service)
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        while not self._client.wait_for_service(timeout_sec=1.0):
            if deadline is not None and time.monotonic() > deadline:
                self.terminate()
                raise TimeoutError(
                    f"{service} service not available after {timeout_s:.0f} s; "
                    "is the gripper node / micro-ROS agent running?"
                )
            self._node.get_logger().info(f"{service} not available, waiting...")

    def send_request(self, grab: bool):
        """True = close / air on, False = open / air off (depending on the service)."""
        if self._mock:
            return None
        if self._client is None or self._node is None or self._rclpy is None or self._setbool is None:
            raise RuntimeError("GripperClient is not initialized")
        req = self._setbool.Request()
        req.data = grab
        future = self._client.call_async(req)
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=15.0)
        if not future.done():
            raise TimeoutError(f"{self._service}({grab}) got no response in 15 s")
        return future.result()

    def terminate(self):
        if self._mock:
            return
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy is not None:
            self._rclpy.shutdown()


# mode -> (service, request value, message)
MODES = {
    "close": (GRAB_SERVICE, True, "Grab (air on, then fingers out)"),
    "open": (GRAB_SERVICE, False, "Release (fingers in, then air off)"),
    "air-on": (VALVE_SERVICE, True, "Air on (fingers untouched)"),
    "air-off": (VALVE_SERVICE, False, "Air off (fingers untouched)"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock-gripper", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("mode", nargs="?", choices=tuple(MODES), default="open")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for the service")
    args = parser.parse_args()

    service, value, message = MODES[args.mode]
    gc = GripperClient(mock=bool(args.mock_gripper), timeout_s=args.timeout, service=service)
    try:
        print(f"{message}: {service} <- {value}")
        response = gc.send_request(value)
        if response is not None:
            print(("Accepted" if response.success else "REJECTED") + (f": {response.message}" if response.message else ""))
            if not response.success:
                sys.exit(1)
    finally:
        gc.terminate()


if __name__ == "__main__":
    main()
