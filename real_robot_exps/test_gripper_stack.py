import unittest
from unittest.mock import patch

from real_robot_exps.gripper_stack import (
    GRIPPER_PROCESS_PATTERNS,
    gripper_stack_ready,
    kill_stray_gripper_processes,
)


class KillStrayProcessesTest(unittest.TestCase):
    def test_kills_every_pattern_gently_then_forcefully(self):
        with patch("real_robot_exps.gripper_stack.subprocess.run") as run, \
             patch("real_robot_exps.gripper_stack.time.sleep"):
            kill_stray_gripper_processes()
        calls = [call.args[0] for call in run.call_args_list]
        for pattern in GRIPPER_PROCESS_PATTERNS:
            self.assertIn(["pkill", "-f", pattern], calls)
            self.assertIn(["pkill", "-9", "-f", pattern], calls)
        # gentle pass (SIGTERM) happens before the forceful pass (SIGKILL)
        self.assertLess(calls.index(["pkill", "-f", GRIPPER_PROCESS_PATTERNS[0]]),
                        calls.index(["pkill", "-9", "-f", GRIPPER_PROCESS_PATTERNS[0]]))

    def test_matches_any_workspace_micro_ros_agent_on_the_gripper_port(self):
        # Two micro_ros_agent processes from different workspaces, both bound to
        # 8888, were observed fighting over the ESP32's traffic; the pattern must
        # catch either regardless of install path.
        pattern = next(p for p in GRIPPER_PROCESS_PATTERNS if "micro_ros_agent" in p)
        import re

        self.assertRegex("/home/u/franka_ros2_ws/install/.../micro_ros_agent udp4 --port 8888", pattern)
        self.assertRegex("/home/u/microros_ws/install/.../micro_ros_agent udp4 --port 8888", pattern)
        self.assertNotRegex("/usr/bin/micro_ros_agent udp4 --port 9999", pattern)


class GripperStackReadyTest(unittest.TestCase):
    def test_mock_is_always_ready(self):
        ready, error = gripper_stack_ready(mock=True, timeout_s=1.0)
        self.assertTrue(ready)
        self.assertIsNone(error)

    def test_reports_the_underlying_error_when_not_ready(self):
        with patch("real_robot_exps.gripper_test.GripperClient", side_effect=TimeoutError("no service")):
            ready, error = gripper_stack_ready(mock=False, timeout_s=1.0)
        self.assertFalse(ready)
        self.assertIn("no service", error)


if __name__ == "__main__":
    unittest.main()
