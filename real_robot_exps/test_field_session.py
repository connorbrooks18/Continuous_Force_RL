import argparse
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

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
