import argparse
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "at-tracking"))

from snapshot_requests import SnapshotRequests  # noqa: E402

from real_robot_exps.compile_static_sysid import (  # noqa: E402
    _correct_snapshot,
    _require_zero_tag_offsets,
    _tag_to_part_geometry,
)
from real_robot_exps.field_session import (  # noqa: E402
    STEPS,
    Apple,
    Console,
    FieldSession,
    GripperRecovered,
    Session,
    _tracking_for,
    compose_camera_to_base,
    parts_from_measurements,
)
from real_robot_exps.snapshot_geometry import (  # noqa: E402
    SnapshotError,
    request_snapshot,
    update_pre_grasp_geometry_with_snapshots,
)


def _refuse_real_gripper(*args, **kwargs):
    raise AssertionError("a test tried to send a command to the real gripper")


_GUARDS = [
    patch("real_robot_exps.field_session.run_gripper_command", side_effect=_refuse_real_gripper),
    patch("real_robot_exps.gripper_stack.run_gripper_command", side_effect=_refuse_real_gripper),
    patch("real_robot_exps.gripper_stack.kill_stray_gripper_processes", side_effect=_refuse_real_gripper),
    patch("real_robot_exps.field_session.kill_stray_gripper_processes", side_effect=_refuse_real_gripper),
]


def setUpModule():
    for guard in _GUARDS:
        guard.start()


def tearDownModule():
    for guard in _GUARDS:
        guard.stop()


def _fake_gripper(calls, fail_first=0):
    """Stand-in for field_session.run_gripper_command: records modes, fails the first N."""
    state = {"n": 0}

    def run(mode, *, timeout_s=30.0):
        state["n"] += 1
        calls.append(mode)
        if state["n"] <= fail_first:
            raise TimeoutError(f"gripper_test {mode}: no reply")
        return "Accepted"

    return run


def _pose(pos, z_axis=(0.0, 0.0, 1.0)):
    z = np.asarray(z_axis, dtype=float)
    x = np.cross([0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    pose[:3, 3] = pos
    return pose


class MeasurementTest(unittest.TestCase):
    def test_parts_use_si_units_and_measured_density(self):
        parts = parts_from_measurements({
            "apple": {"diameter_mm": 80.0, "height_mm": 80.0, "mass_g": 200.0},
            "stem": {"length_mm": 20.0, "diameter_mm": 2.0, "mass_g": None},
            "spur": {"length_mm": 100.0, "diameter_mm": 6.0, "mass_g": 3.0},
            "primary": {"diameter_mm": 30.0, "length_mm": None},
        })
        self.assertAlmostEqual(parts["apple"]["radius_m"], 0.04)
        sphere = 4.0 / 3.0 * np.pi * 0.04 ** 3
        self.assertAlmostEqual(parts["apple"]["density_kg_m3"], 0.2 / sphere)
        self.assertAlmostEqual(parts["spur"]["density_kg_m3"], 0.003 / (np.pi * 0.003 ** 2 * 0.1))
        self.assertEqual(parts["spur"]["density_source"], "measured mass / cylinder volume")
        self.assertEqual(parts["stem"]["density_kg_m3"], 1000.0)
        self.assertEqual(parts["primary"]["density_source"], "default (not weighed)")
        self.assertNotIn("length_m", parts["primary"])


class CameraToBaseTest(unittest.TestCase):
    def test_composition_maps_optical_points_into_base(self):
        base_to_link = _pose([1.0, 2.0, 0.5])
        link_to_optical = np.array([[0, 0, 1, 0.01], [-1, 0, 0, 0.02], [0, -1, 0, 0.0], [0, 0, 0, 1.0]])
        camera_to_base = compose_camera_to_base(base_to_link, link_to_optical)
        point_optical = np.array([0.0, 0.0, 1.0, 1.0])  # 1 m in front of the lens
        np.testing.assert_allclose(camera_to_base @ point_optical, base_to_link @ link_to_optical @ point_optical)


class SnapshotProtocolTest(unittest.TestCase):
    def _serve(self, server, poses, stop):
        while not stop.is_set():
            server.feed(poses, time.time())
            time.sleep(0.005)

    def test_request_returns_median_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = SnapshotRequests(Path(tmp) / "req", np.eye(4))
            poses = {"Branch": _pose([0, 0, 0]), "Spur": _pose([0, 0.1, 0]), "Apple": _pose([0, 0.2, 0])}
            stop = threading.Event()
            thread = threading.Thread(target=self._serve, args=(server, poses, stop), daemon=True)
            thread.start()
            try:
                snapshot = request_snapshot(Path(tmp) / "req", "under_gravity", Path(tmp) / "ug.json", frames=3)
            finally:
                stop.set()
                thread.join()
            self.assertEqual(snapshot["label"], "under_gravity")
            self.assertEqual(snapshot["camera_frame_count"], 3)
            np.testing.assert_allclose(snapshot["apple_pos"], [0, 0.2, 0])
            self.assertEqual(list((Path(tmp) / "req").glob("*.request.json")), [])

    def test_missing_tag_times_out_with_a_useful_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = SnapshotRequests(Path(tmp) / "req", np.eye(4))
            poses = {"Branch": _pose([0, 0, 0]), "Apple": _pose([0, 0.2, 0])}  # spur tag hidden
            stop = threading.Event()
            thread = threading.Thread(target=self._serve, args=(server, poses, stop), daemon=True)
            thread.start()
            try:
                with self.assertRaisesRegex(SnapshotError, "never saw: Spur"):
                    request_snapshot(Path(tmp) / "req", "lengthened", Path(tmp) / "l.json", frames=3, timeout_s=0.3)
            finally:
                stop.set()
                thread.join()


class TagToPartTest(unittest.TestCase):
    def test_shift_is_radius_along_tag_z(self):
        positions = {"Apple": np.array([0.5, 0.3, 0.3])}
        poses = {"Apple": _pose(positions["Apple"], z_axis=(0, 1, 0))}
        out_pos, out_pose = _tag_to_part_geometry(positions, poses, {"Apple": 0.04})
        np.testing.assert_allclose(out_pos["Apple"], [0.5, 0.34, 0.3])
        np.testing.assert_allclose(out_pose["Apple"][:3, :3], poses["Apple"][:3, :3])
        self.assertIs(_tag_to_part_geometry(positions, poses, None)[0], positions)

    def test_snapshot_correction_keeps_raw_tag_values(self):
        snapshot = {
            f"{name}_pose_4x4": _pose(pos).reshape(-1).tolist()
            for name, pos in (("branch", [0, 0, 0]), ("spur", [0, 0.1, 0]), ("apple", [0, 0.2, 0]))
        }
        corrected = _correct_snapshot(snapshot, {"Branch": 0.01, "Spur": 0.002, "Apple": 0.04}, 1.0)
        np.testing.assert_allclose(corrected["apple_pos"], [0, 0.2, 0.04])
        np.testing.assert_allclose(corrected["apple_pos_tag"], [0, 0.2, 0])
        self.assertTrue(corrected["tag_to_part_corrected"])
        geometry = update_pre_grasp_geometry_with_snapshots(
            {"parts": {"spur": {}, "stem": {}}}, lengthened_snapshot=corrected
        )
        self.assertEqual(geometry["parts"]["stem"]["connection_source"], "lengthened_snapshot")
        self.assertNotIn("snapshot", geometry)
        self.assertNotIn("settled_snapshot", geometry)

    def test_nonzero_tag_offsets_are_rejected(self):
        offset = np.eye(4)
        offset[1, 3] = 0.035
        metadata = {"tracking_config": {"objects": {"Spur": {"1": {"offset_4x4": offset.tolist()}}}}}
        with self.assertRaisesRegex(ValueError, "double count"):
            _require_zero_tag_offsets(metadata, Path("t.parquet"))


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


class WorkflowTest(unittest.TestCase):
    def _field(self, tmp, answers):
        args = argparse.Namespace(
            session="s", data_root=tmp, no_detector=True, ros_ws=tmp, record_video=False,
            mock_gripper=True, skip_calibration=True, confirm_each=False, calib_poses=3,
        )
        field = FieldSession(args, ScriptedConsole(answers))
        field.session.dir.mkdir(parents=True)
        field.session.data = {"config_path": str(REPO / "real_robot_exps" / "config.yaml"),
                              "tracking_config_path": str(REPO / "at-tracking" / "tracking_config.yaml")}
        return field

    def test_failed_step_can_be_retried_and_state_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, ["r"])
            calls = []
            for name in STEPS:
                def handler(apple, name=name):
                    calls.append(name)
                    if name == "pulls" and calls.count("pulls") == 1:
                        raise RuntimeError("tag hidden")
                setattr(field, f"step_{name}", handler)
            apple = Apple(field.session, "A001")
            self.assertTrue(field.run_apple(apple))
            self.assertEqual(calls.count("pulls"), 2)
            reloaded = Apple(Session(Path(tmp), "s"), "A001")
            self.assertTrue(reloaded.complete())
            self.assertEqual(reloaded.step("pulls")["error"], "RuntimeError: tag hidden")

    def test_calibration_releases_the_board_even_when_it_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            # no live view, hold board, suction holds, release
            field = self._field(tmp, ["n", "", "", ""])
            field.args.skip_calibration = False
            field.session.data["mock"] = True
            events = []
            field.air_on = lambda: events.append("air_on")
            field.air_off = lambda: events.append("air_off")

            def failing_calibration(apple, name, calib_dir):
                events.append("calibrate")
                raise RuntimeError("handeye_auto_calibrate exited with 1")

            field._run_calibration = failing_calibration
            apple = Apple(field.session, "A001")
            with self.assertRaisesRegex(RuntimeError, "exited with 1"):
                field.step_calibrate(apple)
            self.assertEqual(events, ["air_on", "calibrate", "air_off"])
            prompts = [line for line in field.console.output if "Enter" in line]
            self.assertIn("releases it", prompts[-1])  # released only after the operator confirms

    def test_board_is_repositioned_when_suction_does_not_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            # hold, "n" = slipping, reposition (air off), hold again, "y" = holds
            field = self._field(tmp, ["", "n", "", "", "y"])
            events = []
            field.air_on = lambda: events.append("air_on")
            field.air_off = lambda: events.append("air_off")
            field._hold_board()
            self.assertEqual(events, ["air_on", "air_off", "air_on"])

    def test_resume_starts_at_first_unfinished_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, [])
            apple = Apple(field.session, "A002")
            for name in STEPS[:4]:
                apple.mark(name, "done")
            self.assertEqual(Apple(field.session, "A002").next_step(), "grasp")
            self.assertEqual(field.session.next_apple_id(), "A003")


class RosEnvTest(unittest.TestCase):
    def test_conda_is_stripped_but_display_is_kept(self):
        from unittest.mock import patch

        from real_robot_exps.field_session import ros_env

        fake = {
            "PATH": "/home/u/anaconda3/bin:/usr/bin", "LD_LIBRARY_PATH": "/home/u/anaconda3/lib",
            "CONDA_PREFIX": "/home/u/anaconda3", "PYTHONPATH": "/x", "DISPLAY": ":1", "ROS_DOMAIN_ID": "7",
        }
        with patch.dict("os.environ", fake, clear=True):
            env = ros_env()
        self.assertNotIn("anaconda3", env["PATH"])
        for key in ("LD_LIBRARY_PATH", "CONDA_PREFIX", "PYTHONPATH"):
            self.assertNotIn(key, env)
        self.assertEqual(env["DISPLAY"], ":1")
        self.assertEqual(env["ROS_DOMAIN_ID"], "7")


class GripperCallRetryTest(unittest.TestCase):
    """_gripper_call is the single choke point close/open/air-on/air-off all go
    through, so testing it here covers the retry-with-restart behaviour for all of
    them without duplicating it per call site."""

    def _field(self, tmp, answers):
        args = argparse.Namespace(
            session="s", data_root=tmp, no_detector=True, ros_ws=tmp, record_video=False,
            mock_gripper=False, skip_calibration=True, confirm_each=False, calib_poses=3,
            no_gripper_stack=False, gripper_ssid="alejos", gripper_password="harvesting",
        )
        field = FieldSession(args, ScriptedConsole(answers))
        field.session.dir.mkdir(parents=True)
        field.session.data = {"config_path": str(REPO / "real_robot_exps" / "config.yaml"),
                              "tracking_config_path": str(REPO / "at-tracking" / "tracking_config.yaml")}
        return field

    def test_failed_close_recovers_then_asks_the_step_to_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, [])
            reasons = []
            field.recover_gripper = lambda reason: reasons.append(reason)
            with patch("real_robot_exps.field_session.run_gripper_command", _fake_gripper([], fail_first=1)):
                with self.assertRaises(GripperRecovered):
                    field._gripper_call("gripper_grab", True, "close gripper")
            self.assertEqual(len(reasons), 1)
            self.assertIn("close gripper", reasons[0])

    def test_failed_open_is_done_by_the_recovery_itself(self):
        # the recovery ends with the gripper open, so a failed open/air-off needs no step restart
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, [])
            reasons = []
            field.recover_gripper = lambda reason: reasons.append(reason)
            with patch("real_robot_exps.field_session.run_gripper_command", _fake_gripper([], fail_first=1)):
                field._gripper_call("/microROS/toggle_valve", False, "air off")
            self.assertEqual(len(reasons), 1)

    def test_without_stack_management_a_failure_just_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, [])
            field.args.no_gripper_stack = True
            field.recover_gripper = lambda reason: self.fail("must not recover")
            with patch("real_robot_exps.field_session.run_gripper_command", _fake_gripper([], fail_first=1)):
                with self.assertRaises(TimeoutError):
                    field._gripper_call("gripper_grab", True, "close gripper")

    def test_recovery_restarts_then_tests_close_and_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, ["", "y"])  # hold what the gripper holds; "did it close and open?"
            restarts, calls = [], []
            field.ensure_gripper_stack = lambda force_restart=False, confirm=True: restarts.append((force_restart, confirm)) or True
            with patch("real_robot_exps.field_session.run_gripper_command", _fake_gripper(calls)), \
                 patch("real_robot_exps.field_session.time.sleep"):
                field.recover_gripper("air on: no reply")
            self.assertEqual(restarts, [(True, False)])  # restart without the separate open prompt
            self.assertEqual(calls, ["close", "open"])

    def test_recovery_repeats_until_the_test_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            # hold; test says "n" -> restart and test again; test says "y"
            field = self._field(tmp, ["", "n", "r", "y"])
            restarts = []
            field.ensure_gripper_stack = lambda force_restart=False, confirm=True: restarts.append(1) or True
            with patch("real_robot_exps.field_session.run_gripper_command", _fake_gripper([])), \
                 patch("real_robot_exps.field_session.time.sleep"):
                field.recover_gripper("x")
            self.assertEqual(len(restarts), 2)

    def test_operator_can_give_up_on_the_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, ["", "q"])
            field.ensure_gripper_stack = lambda force_restart=False, confirm=True: False  # controller never comes up
            with self.assertRaisesRegex(RuntimeError, "could not be recovered"):
                field.recover_gripper("x")

    def test_hold_board_starts_over_after_a_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            # hold -> (air on interrupted) -> hold again -> air on -> "suction holds?" yes
            field = self._field(tmp, ["", "", "y"])
            attempts = []

            def air_on():
                attempts.append(1)
                if len(attempts) == 1:
                    raise GripperRecovered("air on was interrupted")

            field.air_on = air_on
            field._hold_board()
            self.assertEqual(len(attempts), 2)
            prompts = [line for line in field.console.output if "Hold the ChArUco board" in line]
            self.assertEqual(len(prompts), 2)  # the operator was asked to hold the board again

    def test_run_apple_restarts_the_interrupted_step_without_asking(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, [])  # no answers: a retry/skip/quit prompt would raise EOFError
            calls = []
            for name in STEPS:
                def handler(apple, name=name):
                    calls.append(name)
                    if name == "grasp" and calls.count("grasp") == 1:
                        raise GripperRecovered("close gripper interrupted")
                setattr(field, f"step_{name}", handler)
            self.assertTrue(field.run_apple(Apple(field.session, "A001")))
            self.assertEqual(calls.count("grasp"), 2)

    def test_ensure_gripper_stack_is_a_noop_in_mock_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, [])
            field.args.mock_gripper = True
            with patch("real_robot_exps.field_session.kill_stray_gripper_processes") as kill:
                field.ensure_gripper_stack()
            kill.assert_not_called()

    def test_ensure_gripper_stack_kills_relaunches_then_opens(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, [])
            order = []
            field.open_and_confirm = lambda: order.append("open")
            with patch("real_robot_exps.field_session.kill_stray_gripper_processes",
                       side_effect=lambda: order.append("kill")), \
                 patch("real_robot_exps.field_session.launch_gripper_stack",
                       side_effect=lambda *a, **k: order.append("launch") or object()), \
                 patch("real_robot_exps.field_session.gripper_stack_ready", return_value=(True, None)):
                field.ensure_gripper_stack()
            self.assertEqual(order, ["kill", "launch", "open"])

    def test_no_open_when_the_controller_does_not_come_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, [])
            field.open_and_confirm = lambda: self.fail("must not open a controller that isn't up")
            with patch("real_robot_exps.field_session.kill_stray_gripper_processes"), \
                 patch("real_robot_exps.field_session.launch_gripper_stack", return_value=object()), \
                 patch("real_robot_exps.field_session.gripper_stack_ready", return_value=(False, "timeout")):
                field.ensure_gripper_stack()

    def test_open_then_operator_confirms_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, ["y"])  # "Is the gripper released?"
            calls = []
            with patch("real_robot_exps.field_session.run_gripper_command", _fake_gripper(calls)):
                field.open_and_confirm()
            self.assertEqual(calls, ["open"])
            self.assertIn("released", field.console.output[-1])

    def test_not_released_retries_the_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, ["n", "r", "y"])  # not released -> retry -> released
            calls = []
            with patch("real_robot_exps.field_session.run_gripper_command", _fake_gripper(calls)):
                field.open_and_confirm()
            self.assertEqual(calls, ["open", "open"])

    def test_failed_open_asks_and_can_continue_without_restarting(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, ["", "c"])  # default answer is "no" after a failure -> continue
            field.ensure_gripper_stack = lambda force_restart=False: self.fail("must not restart on its own")
            calls = []
            with patch("real_robot_exps.field_session.run_gripper_command", _fake_gripper(calls, fail_first=99)):
                field.open_and_confirm()
            self.assertTrue(any("Opening the gripper failed" in line for line in field.console.output))
            self.assertIn("continued with the gripper not confirmed",
                          (Path(tmp) / "s" / "gripper_stack.log").read_text())

    def test_operator_can_ask_for_a_controller_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, ["n", "s"])
            restarts = []
            field.ensure_gripper_stack = lambda force_restart=False: restarts.append(force_restart)
            with patch("real_robot_exps.field_session.run_gripper_command", _fake_gripper([])):
                field.open_and_confirm()
            self.assertEqual(restarts, [True])

    def test_open_and_confirm_is_skipped_in_mock_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, [])  # no answers: any prompt would raise EOFError
            field.args.mock_gripper = True
            field.open_and_confirm()

    def test_end_effector_check_is_off_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = self._field(tmp, [])
            field.args.ee_check = False
            with patch("real_robot_exps.ee_profiles.require_profile") as require:
                field.check_ee(Apple(field.session, "A001"), "gripper", "grasp")
            require.assert_not_called()
            from real_robot_exps.field_session import build_parser
            self.assertFalse(build_parser().parse_args(["--session", "x"]).ee_check)
            self.assertTrue(build_parser().parse_args(["--session", "x", "--ee-check"]).ee_check)


class TrackingSelectionTest(unittest.TestCase):
    def test_picks_the_tracking_file_that_covers_the_pull(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            robot = Path(tmp) / "d00_robot.parquet"
            pq.write_table(pa.table({"timestamp": [100.0, 101.0, 102.0]}), robot)
            early, late = Path(tmp) / "tracking_00.parquet", Path(tmp) / "tracking_01.parquet"
            pq.write_table(pa.table({"timestamp": [0.0, 50.0]}), early)
            pq.write_table(pa.table({"timestamp": [99.0, 103.0]}), late)
            self.assertEqual(_tracking_for(robot, [early, late]), late)
            with self.assertRaises(ValueError):
                _tracking_for(robot, [early])


if __name__ == "__main__":
    unittest.main()
