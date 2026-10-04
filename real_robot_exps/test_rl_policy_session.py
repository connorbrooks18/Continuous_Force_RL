import argparse
import io
import os
import pty
import select
import tempfile
import termios
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from real_robot_exps.field_session import Console
from real_robot_exps.rl_policy_env import KeypressStop
from real_robot_exps.rl_policy_session import PolicySession


class ScriptedConsole(Console):
    def __init__(self, answers):
        self.answers = list(answers)
        self.output = []
        super().__init__(input_fn=self._next, print_fn=lambda *a, **k: self.output.append(" ".join(map(str, a))))

    def _next(self, prompt):
        self.output.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


class FakeRobot:
    def __init__(self, calls):
        self.calls = calls

    def end_control(self):
        self.calls.append("end_control")


class FakeEnv:
    def __init__(self, calls):
        self.calls = calls
        self.robot = FakeRobot(calls)

    def calibrate_ft_bias(self):
        self.calls.append("calibrate")
        return [0.0] * 6

    def reset(self):
        self.calls.append("reset")
        return np.zeros(40, dtype=np.float32), {}

    def step(self, action):
        self.calls.append("step")
        return np.zeros(40, dtype=np.float32), 0.0, False, False, {"vic_action": np.zeros(19), "env_action": np.zeros(13)}

    def close(self):
        self.calls.append("env_close")


class FakePolicy:
    def __init__(self, calls, ood_answers=()):
        self.calls = calls
        self.ood_answers = list(ood_answers)

    def out_of_distribution(self, obs):
        return self.ood_answers.pop(0) if self.ood_answers else []

    def reset(self):
        pass

    def act(self, obs):
        return torch.zeros(13)


def _args(tmp, **overrides):
    base = dict(
        log_dir=tmp, log=None, steps=3, allow_ood=False, mock_gripper=True, no_gripper_stack=False,
        ros_ws=tmp, gripper_ssid="s", gripper_password="p",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _session(tmp, answers, *, ood=(), **overrides):
    calls = []
    console = ScriptedConsole(answers)
    session = PolicySession(_args(tmp, **overrides), console, policy=FakePolicy(calls, ood), env=FakeEnv(calls))
    real_gripper = session.gripper

    def gripper(mode):
        calls.append(mode)
        real_gripper(mode)

    session.gripper = gripper
    return session, calls, console


class PolicySessionTest(unittest.TestCase):
    def test_rejected_grasp_reopens_recalibrates_and_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            # hand-guide, "not held", retry, hand-guide, "held", no next apple, release Enter
            session, calls, _ = _session(tmp, ["", "n", "r", "", "y", "n", ""])
            self.assertEqual(session.run(), 0)
            self.assertEqual(
                calls,
                ["calibrate", "close", "open", "calibrate", "close",
                 "reset", "step", "step", "step", "env_close", "open", "air-off"],
            )

    def test_air_is_switched_off_before_the_gripper_stack_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, calls, _ = _session(tmp, ["", "y", "n", ""])

            class FakeStack:
                def stop(self):
                    calls.append("stack_stop")

            session._stack = FakeStack()
            session.run()
            self.assertEqual(calls[-3:], ["open", "air-off", "stack_stop"])

    def test_quitting_after_a_rejected_grasp_still_switches_the_air_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, calls, _ = _session(tmp, ["", "n", "q"])
            self.assertEqual(session.run(), 1)
            self.assertEqual(calls[-3:], ["open", "env_close", "air-off"])

    def test_ood_start_pose_runs_nothing_and_regrasps_with_torque_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            ood = [[("d_pos.x", 1.0, 9.0)], []]
            session, calls, _ = _session(tmp, ["", "y", "g", "", "y", "n", ""], ood=ood)
            self.assertEqual(session.run(), 0)
            first_reset = calls.index("reset")
            self.assertEqual(calls[first_reset:first_reset + 3], ["reset", "end_control", "open"])
            self.assertEqual(calls.count("step"), 3)

    def test_interrupted_release_leaves_gripper_closed_and_says_how_to_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, calls, console = _session(tmp, ["", "y", "n"])  # no next apple; EOF at the release prompt
            self.assertEqual(session.run(), 0)
            self.assertEqual(calls[-1], "env_close")  # no air-off: the apple is still held
            self.assertTrue(session.gripper_closed)
            self.assertTrue(any("gripper_test open" in line for line in console.output))

    def test_gripper_fault_defaults_to_restarting_the_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            # stack launch skipped (--no-gripper-stack) so only open_and_confirm runs on restart
            session, calls, _ = _session(
                tmp, ["y", "", "", "y", "", "y", "n", ""], mock_gripper=False, no_gripper_stack=True,
            )
            results = iter([None, TimeoutError("gripper_test close: Timeout"), None, None, None, None, None])

            def fake_command(mode, **_):
                result = next(results)
                if isinstance(result, Exception):
                    raise result
                return ""

            with patch("real_robot_exps.rl_policy_session.run_gripper_command", side_effect=fake_command):
                self.assertEqual(session.run(), 0)
            # open (startup), close fails, open (restart), close, policy, open (release)
            self.assertEqual([c for c in calls if c in ("open", "close")], ["open", "close", "open", "close", "open"])

    def test_rollout_log_is_saved_in_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, _, _ = _session(tmp, ["", "y", "n", ""])
            session.run()
            data = np.load(f"{tmp}/rollout.npz")
            self.assertEqual(data["action"].shape, (3, 13))
            self.assertEqual(data["env_action"].shape, (3, 13))

    def test_yes_to_next_apple_recalibrates_and_keeps_the_arm_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            # grasp, yes next, release, grasp again, no next, release
            session, calls, _ = _session(tmp, ["", "y", "y", "", "", "y", "n", ""])
            self.assertEqual(session.run(), 0)
            self.assertEqual(
                calls,
                ["calibrate", "close", "reset", "step", "step", "step",
                 "end_control", "open",
                 "calibrate", "close", "reset", "step", "step", "step",
                 "env_close", "open", "air-off"],
            )
            second = np.load(f"{tmp}/rollout_02.npz")
            self.assertEqual(second["action"].shape, (3, 13))

    def test_wrong_contract_checkpoint_is_refused_before_anything_starts(self):
        from real_robot_exps import rl_policy_session

        with patch.object(rl_policy_session, "HarvestPolicy",
                          side_effect=RuntimeError("action_frame=None; tool-frame checkpoints only")), \
                patch.object(rl_policy_session, "FrankaVicHarvestEnv") as env_cls, \
                patch.object(rl_policy_session, "PolicySession") as session_cls:
            self.assertEqual(rl_policy_session.main(["--checkpoint", "ckpt", "--mock", "--mock-gripper"]), 2)
            env_cls.assert_not_called()
            session_cls.assert_not_called()

    def test_keypress_stops_the_rollout_and_goes_on_to_the_next_apple_prompt(self):
        from real_robot_exps import rl_policy_session

        class StopAfterTwo:
            def __init__(self):
                self.calls = 0

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            def __call__(self):
                self.calls += 1
                return self.calls > 2

        with tempfile.TemporaryDirectory() as tmp, patch.object(rl_policy_session, "KeypressStop", StopAfterTwo):
            session, calls, console = _session(tmp, ["", "y", "n", ""], steps=50)
            self.assertEqual(session.run(), 0)
            self.assertEqual(calls.count("step"), 2)
            self.assertTrue(any("stopped by operator after 2 steps" in line for line in console.output))
            self.assertTrue(any("Move to the next apple?" in line for line in console.output))
            self.assertEqual(np.load(f"{tmp}/rollout.npz")["action"].shape, (2, 13))


class KeypressStopTest(unittest.TestCase):
    def test_key_on_a_tty_stops_and_is_flushed_and_terminal_restored(self):
        master, slave = pty.openpty()
        try:
            with os.fdopen(slave, "r", closefd=False) as stream:
                before = termios.tcgetattr(stream)
                with KeypressStop(stream) as stop:
                    self.assertFalse(stop())
                    self.assertFalse(termios.tcgetattr(stream)[3] & termios.ICANON)  # cbreak: no Enter needed
                    os.write(master, b"s")
                    time.sleep(0.05)
                    self.assertTrue(stop())
                    self.assertTrue(stop())  # latched
                self.assertEqual(termios.tcgetattr(stream), before)
                self.assertEqual(select.select([stream], [], [], 0)[0], [])  # stop key flushed
        finally:
            os.close(master)
            os.close(slave)

    def test_without_a_tty_never_stops(self):
        with KeypressStop(io.StringIO("x")) as stop:
            self.assertFalse(stop())


if __name__ == "__main__":
    unittest.main()
