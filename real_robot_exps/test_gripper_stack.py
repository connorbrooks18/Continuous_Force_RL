import subprocess
import unittest
from unittest.mock import patch

from real_robot_exps.gripper_stack import (
    GRIPPER_PROCESS_PATTERNS,
    gripper_stack_ready,
    kill_stray_gripper_processes,
    run_gripper_command,
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
        failed = subprocess.CompletedProcess([], 1, "", "TimeoutError: gripper_grab service not available")
        with patch("real_robot_exps.gripper_stack.subprocess.run", return_value=failed) as run:
            ready, error = gripper_stack_ready(mock=False, timeout_s=1.0)
        self.assertFalse(ready)
        self.assertIn("not available", error)
        self.assertIn("ready", run.call_args.args[0])  # ran `gripper_test ready` in its own process


class RunGripperCommandTest(unittest.TestCase):
    def test_runs_gripper_test_in_a_fresh_process(self):
        ok = subprocess.CompletedProcess([], 0, "Air on (fingers untouched): ... <- True\nAccepted", "")
        with patch("real_robot_exps.gripper_stack.subprocess.run", return_value=ok) as run:
            output = run_gripper_command("air-on", timeout_s=15.0)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[1:4], ["-m", "real_robot_exps.gripper_test", "air-on"])
        self.assertIn("Accepted", output)

    def test_timeout_and_rejection_raise_distinct_errors(self):
        timed_out = subprocess.CompletedProcess([], 1, "", "TimeoutError: toggle_valve got no response in 15 s")
        rejected = subprocess.CompletedProcess([], 1, "REJECTED: busy", "")
        with patch("real_robot_exps.gripper_stack.subprocess.run", return_value=timed_out):
            with self.assertRaises(TimeoutError):
                run_gripper_command("air-on")
        with patch("real_robot_exps.gripper_stack.subprocess.run", return_value=rejected):
            with self.assertRaisesRegex(RuntimeError, "REJECTED"):
                run_gripper_command("close")

    def test_a_hung_process_is_a_timeout(self):
        with patch("real_robot_exps.gripper_stack.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1)):
            with self.assertRaises(TimeoutError):
                run_gripper_command("open", timeout_s=1.0)


if __name__ == "__main__":
    unittest.main()
