"""Tests for the bug-hunt fixes and field_verify, on synthetic data only (no hardware)."""

import argparse
import io
import json
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "at-tracking"))

from DataCollector import DataCollector  # noqa: E402

from real_robot_exps.apple_pullto_static import save_robot_hold_parquet  # noqa: E402
from real_robot_exps.field_pull import ProcessGripper  # noqa: E402
from real_robot_exps.field_session import (  # noqa: E402
    Apple,
    Console,
    FieldSession,
    Proc,
    Session,
    is_log_noise,
    run_logged,
    supersede_for_redo,
)
from real_robot_exps.field_verify import verify_apple  # noqa: E402


def _refuse(*args, **kwargs):
    raise AssertionError("a test tried to reach the real gripper")


_GUARDS = [
    patch("real_robot_exps.field_session.run_gripper_command", side_effect=_refuse),
    patch("real_robot_exps.gripper_stack.run_gripper_command", side_effect=_refuse),
    patch("real_robot_exps.field_session.kill_stray_gripper_processes", side_effect=_refuse),
]


def setUpModule():
    for guard in _GUARDS:
        guard.start()


def tearDownModule():
    for guard in _GUARDS:
        guard.stop()


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


def _field(tmp, answers=(), **overrides):
    args = argparse.Namespace(
        session="s", data_root=tmp, no_detector=False, ros_ws=tmp, record_video=False, mock_gripper=False,
        skip_calibration=False, confirm_each=False, calib_poses=3, calib_rotation_deg=25.0,
        no_gripper_stack=False, gripper_ssid="alejos", gripper_password="harvesting", ee_check=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    field = FieldSession(args, ScriptedConsole(answers))
    field.session.dir.mkdir(parents=True, exist_ok=True)
    field.session.data = {
        "config_path": str(REPO / "real_robot_exps" / "config.yaml"),
        "tracking_config_path": str(REPO / "at-tracking" / "tracking_config.yaml"),
        "directions": [{"index": 0, "theta": 1.57, "phi": 1.57}],
        "stops": 2,
    }
    return field


class LoggingTest(unittest.TestCase):
    def test_noise_filter_drops_only_noise(self):
        self.assertTrue(is_log_noise("Error, more than one new minima found.\n"))
        self.assertTrue(is_log_noise("[ros2-2] \x1b[35m[1790.1]\x1b[m info | \x1b[34mUDPv4AgentLinux.cpp\x1b[m | "
                                     "\x1b[37minit    \x1b[m | running... | port: 8888"))
        self.assertFalse(is_log_noise("[ros2-2] info | SessionManager.hpp | establish_session | session established"))
        self.assertFalse(is_log_noise("[field_pull] ABORT: RuntimeError: No state available"))

    def test_run_logged_tees_output_and_keeps_prompts_without_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.txt"
            script = ("import sys; print('Error, more than one new minima found.'); print('real line'); "
                      "sys.stdout.write('press Enter: '); sys.stdout.flush(); sys.exit(3)")
            terminal = io.StringIO()
            with redirect_stdout(terminal):
                code = run_logged([sys.executable, "-c", script], log)
            self.assertEqual(code, 3)
            self.assertIn("press Enter: ", terminal.getvalue())  # prompt reached the terminal
            text = log.read_text()
            self.assertIn("real line", text)
            self.assertIn("press Enter: ", text)
            self.assertNotIn("\nError, more than one new minima found.\n", text)  # noise kept off the log
            self.assertIn("exit 3", text)

    def test_proc_filters_noise_into_the_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.txt"
            script = "print('Error, more than one new minima found.'); print('detector started')"
            proc = Proc("t", [sys.executable, "-c", script], log)
            for _ in range(50):
                if not proc.alive():
                    break
                time.sleep(0.05)
            proc.stop()
            text = log.read_text()
            self.assertIn("detector started", text)
            self.assertNotIn("\nError, more than one new minima found.\n", text)


class BaselineLoopTest(unittest.TestCase):
    def test_a_failed_baseline_does_not_stop_the_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = _field(tmp, [""])
            field.session.data["directions"] = [{"index": i, "theta": 1.0, "phi": 1.0} for i in range(3)]
            apple = Apple(field.session, "A001")
            for i in range(3):
                apple.direction_files(i)["robot"].write_bytes(b"x")
            attempted = []

            def fake_run(cmd, log_path, **kwargs):
                index = cmd[cmd.index("--output") + 1]
                attempted.append(index)
                return 1 if "d01" in index else 0

            with patch("real_robot_exps.field_session.run_logged", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "d01"):
                    field.step_baseline(apple)
            self.assertEqual(len(attempted), 3)  # d02 still ran after d01 failed


class CalibrationResultTest(unittest.TestCase):
    def test_stale_result_is_removed_and_a_missing_one_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = _field(tmp)
            apple = Apple(field.session, "A001")
            handeye = Path(tmp) / "handeye"
            (handeye / "calibrations").mkdir(parents=True)
            (handeye / "samples").mkdir()
            stale = handeye / "calibrations" / "s_A001.calib"
            stale.write_text("from an earlier attempt")

            class FakeProc:
                def __init__(self, *args, **kwargs):
                    pass

                def alive(self):
                    return True

                def stop(self):
                    return 0

            report = type("R", (), {"stdout": "Verdict: GOOD", "stderr": ""})()
            with patch("real_robot_exps.field_session.HANDEYE_DIR", handeye), \
                 patch("real_robot_exps.field_session.Proc", FakeProc), \
                 patch("real_robot_exps.field_session.run_logged", return_value=0), \
                 patch("real_robot_exps.field_session.subprocess.run", return_value=report), \
                 patch("real_robot_exps.field_session.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "did not save a result"):
                    field._run_calibration(apple, "s_A001", apple.dir / "calib")
            self.assertFalse(stale.exists())  # the earlier attempt's result was never used


class GraspTagCheckTest(unittest.TestCase):
    def _run(self, answers, error):
        tmp = tempfile.mkdtemp()
        field = _field(tmp, answers)
        apple = Apple(field.session, "A001")
        opens = []
        field._gripper_call = lambda service, value, what: opens.append((value, what))

        def fake_snapshot(apple_, label):
            (apple_.dir / "snapshots" / f"{label}.json").write_text(json.dumps(
                {"error": error, "tracker_seen_counts": {"Branch": 30, "Spur": 30, "Apple": 0}}))
            raise RuntimeError(error)

        field.snapshot = fake_snapshot
        return field, apple, opens

    def test_hidden_apple_tag_offers_a_regrasp(self):
        field, apple, opens = self._run(["r"], "never saw: Apple")
        with self.assertRaisesRegex(RuntimeError, "Apple"):
            field._grasp_tag_check(apple)
        self.assertEqual(opens, [(False, "open gripper")])  # opened for the new grasp
        self.assertEqual(apple.data["grasp_tag_check"]["missing"], ["Apple"])

    def test_operator_can_continue_anyway(self):
        field, apple, opens = self._run(["c"], "never saw: Apple")
        field._grasp_tag_check(apple)
        self.assertEqual(opens, [])
        self.assertEqual(apple.data["grasp_tag_check"]["operator"], "c")


class RedoTest(unittest.TestCase):
    def test_redo_pulls_moves_old_data_aside_and_forgets_completed_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = _field(tmp)
            apple = Apple(field.session, "A001")
            apple.step("pulls")["completed_directions"] = [0, 1]
            for i in range(2):
                apple.direction_files(i)["robot"].write_bytes(b"r")
                apple.direction_files(i)["baseline"].write_bytes(b"b")
            moved = supersede_for_redo(apple, "pulls")
            self.assertEqual(len(moved), 4)
            self.assertFalse(list(apple.dir.glob("pulls/d*_robot.parquet")))
            self.assertTrue(all(path.exists() and "superseded-" in str(path) for path in moved))
            self.assertNotIn("completed_directions", Apple(field.session, "A001").step("pulls"))
            self.assertEqual(Apple(field.session, "A001").step("baseline")["status"], "pending")

    def test_redo_baseline_keeps_the_pulls(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = _field(tmp)
            apple = Apple(field.session, "A001")
            apple.direction_files(0)["robot"].write_bytes(b"r")
            apple.direction_files(0)["baseline"].write_bytes(b"b")
            moved = supersede_for_redo(apple, "baseline")
            self.assertEqual([p.name for p in moved], ["d00_baseline.parquet"])
            self.assertTrue(apple.direction_files(0)["robot"].exists())


class ProcessGripperTest(unittest.TestCase):
    def test_commands_run_as_their_own_process(self):
        calls = []
        with patch("real_robot_exps.gripper_stack.run_gripper_command",
                   side_effect=lambda mode, **k: calls.append(mode)):
            gripper = ProcessGripper()
            gripper.send_request(False)
            gripper.send_request(True)
        self.assertEqual(calls, ["open", "close"])
        ProcessGripper(mock=True).send_request(False)  # mock: nothing sent (the guard would raise)


def _write_apple(root: Path, *, apple_visible: bool, baseline: bool, parts_ok: bool = True) -> Path:
    """A minimal but complete-looking apple folder with one direction."""
    apple = root / "A001"
    for sub in ("calib", "snapshots", "tracking", "pulls", "baseline", "compiled", "config"):
        (apple / sub).mkdir(parents=True)
    camera = np.eye(4)
    (apple / "calib" / "camera_to_base.json").write_text(json.dumps({"camera_to_base_4x4": camera.tolist()}))
    for name in ("s_A001.calib", "s_A001.samples", "report.txt", "camera_link_to_optical.json"):
        (apple / "calib" / name).write_text("x")
    snap = {"apple_pos": [0, 0, 0], "branch_pos": [0, 0, 0], "spur_pos": [0, 0, 0], "camera_frame_count": 5}
    for label in ("under_gravity", "lengthened"):
        (apple / "snapshots" / f"{label}.json").write_text(json.dumps(snap))
        (apple / "snapshots" / f"{label}.png").write_bytes(b"png")
    t0 = 1000.0
    rows = []
    for k in range(200):
        rows.append({
            "timestamp": t0 + k * 0.001, "hold_index": k // 50, "phase": (k // 25) % 2, "hold_step_idx": k,
            "joint_vel": np.zeros(7, np.float32), "ft_wrist": np.zeros(6, np.float32),
        })
    meta = {key: 0 for key in ("episode_id", "rest_reference_timestamp", "theta_rad", "phi_rad")}
    meta.update({"robot_start_pose_4x4": np.eye(4).tolist(), "robot_start_joint_pos": [0.0] * 7,
                 "pre_grasp_geometry": {}, "post_grasp_geometry": {}, "dump": {"n_holds": 4}})
    save_robot_hold_parquet(rows, apple / "pulls" / "d00_robot.parquet", meta)
    if baseline:
        base_meta = {"baseline_start_method": "joint_positions",
                     "baseline_replay_filter": {"type": "butterworth zero-phase (filtfilt)", "cutoff_hz": 5.0}}
        save_robot_hold_parquet(rows, apple / "baseline" / "d00_baseline.parquet", base_meta)
    collector = DataCollector(metadata={"coordinate_frame": "franka_base_o", "camera_to_base_4x4_used": camera.tolist()})
    for k in range(10):
        t = t0 - 0.05 + k * 0.03
        for name in ("Branch", "Spur", "Apple"):
            visible = name != "Apple" or apple_visible
            collector.update(t, name, *((0.1, 0.2, 0.3) if visible else (np.nan,) * 3), 0, 0, 0, 1)
    collector.dump(str(apple / "tracking" / "tracking_00.parquet"), metadata={"partial": False})
    radius = 0.04 if parts_ok else 0.2
    parts = {
        "apple": {"radius_m": radius, "density_kg_m3": 800.0},
        "spur": {"radius_m": 0.003, "density_kg_m3": 1200.0, "density_source": "default (not weighed)"},
        "stem": {"radius_m": 0.001, "density_kg_m3": 1000.0, "density_source": "default (not weighed)"},
        "primary": {"radius_m": 0.015},
    }
    steps = {name: {"status": "done"} for name in ("notes", "calibrate", "tags", "snapshots", "grasp", "pulls",
                                                    "baseline", "measurements")}
    (apple / "apple.json").write_text(json.dumps({"steps": steps, "calibration": {"verdict": "GOOD"}, "parts": parts}))
    return apple


class VerifyTest(unittest.TestCase):
    SESSION = {"directions": [{"index": 0, "theta": 1.0, "phi": 1.0}], "stops": 2}

    def _status(self, report, name):
        return next(s.status for s in report.sections if s.name == name)

    def test_complete_apple_has_no_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = verify_apple(_write_apple(Path(tmp), apple_visible=True, baseline=True), self.SESSION)
            for name in ("calibration", "snapshots", "pulls", "baselines"):
                self.assertEqual(self._status(report, name), "PASS", report.text())
            self.assertNotEqual(report.status, "FAIL", report.text())

    def test_hidden_apple_tag_and_missing_baseline_fail_with_redo_hints(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = verify_apple(_write_apple(Path(tmp), apple_visible=False, baseline=False), self.SESSION, "sess")
            self.assertEqual(self._status(report, "pulls"), "FAIL")
            self.assertEqual(self._status(report, "baselines"), "FAIL")
            text = report.text()
            self.assertIn("--redo grasp --redo pulls", text)
            self.assertIn("--session sess --apple A001 --redo baseline", text)

    def test_implausible_measurements_warn(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = verify_apple(_write_apple(Path(tmp), apple_visible=True, baseline=True, parts_ok=False),
                                  self.SESSION)
            self.assertIn("implausible", report.text())


if __name__ == "__main__":
    unittest.main()
