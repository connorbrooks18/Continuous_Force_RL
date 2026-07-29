"""Small ROS2 gripper client with an optional no-op mock mode."""

from __future__ import annotations

import argparse
import sys


class GripperClient:
    """Connect to the gripper service, or act as a no-op mock."""

    def __init__(self, mock: bool = False):
        self._mock = bool(mock)
        self._rclpy = None
        self._node = None
        self._client = None
        self._setbool = None
        if self._mock:
            print("[MockGripper] Gripper connection disabled; using no-op client.")
            return

        import rclpy
        from rclpy.node import Node
        from std_srvs.srv import SetBool

        self._rclpy = rclpy
        self._setbool = SetBool
        rclpy.init()
        self._node = Node("gripper_grab_client")
        self._client = self._node.create_client(SetBool, "gripper_grab")
        while not self._client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info("Service not available, waiting...")

    def send_request(self, grab: bool):
        if self._mock:
            return None
        if self._client is None or self._node is None or self._rclpy is None or self._setbool is None:
            raise RuntimeError("GripperClient is not initialized")
        req = self._setbool.Request()
        req.data = grab
        future = self._client.call_async(req)
        self._rclpy.spin_until_future_complete(self._node, future)
        return future.result()

    def terminate(self):
        if self._mock:
            return
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy is not None:
            self._rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock-gripper", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("mode", nargs="?", choices=("open", "close"), default="open")
    args = parser.parse_args()

    gc = GripperClient(mock=bool(args.mock_gripper))
    try:
        if args.mode == "close":
            print("Sending Grab Request...")
            response = gc.send_request(True)
            if response and response.success:
                print("Grab command accepted!")
        else:
            print("Sending Release Request...")
            response = gc.send_request(False)
            if response and response.success:
                print("Release command accepted!")
    finally:
        gc.terminate()


if __name__ == "__main__":
    main()
