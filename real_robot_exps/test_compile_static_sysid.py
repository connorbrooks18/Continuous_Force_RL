import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from real_robot_exps.compile_static_sysid import (
    _match_baseline_frames,
    compile_static_episode,
)


def _write_with_metadata(path, rows, metadata):
    table = pa.Table.from_pylist(rows)
    table = table.replace_schema_metadata({
        b"dataset_metadata": json.dumps(metadata).encode("utf-8")
    })
    pq.write_table(table, path)


class CompileStaticSysidTest(unittest.TestCase):
    def test_matches_baseline_by_frame_index_without_interpolation(self):
        source = np.arange(54, dtype=np.float64).reshape(9, 6)

        matched = _match_baseline_frames(source, 10)

        np.testing.assert_array_equal(matched[:9], source)
        np.testing.assert_array_equal(matched[9], source[-1])

    def test_rejects_large_baseline_frame_count_mismatch(self):
        source = np.zeros((8, 6), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "more than 10%"):
            _match_baseline_frames(source, 10)

    def test_compiles_legacy_branch_spur_apple_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            robot_path = tmp / "robot.parquet"
            tracking_path = tmp / "tracking.parquet"
            output_path = tmp / "unified.parquet"

            robot_rows = []
            for hold_idx, base_time in enumerate((101.0, 102.0)):
                for hold_step_idx, timestamp in enumerate((base_time, base_time + 0.1)):
                    robot_rows.append({
                        "timestamp": timestamp,
                        "hold_step_idx": hold_step_idx,
                        "hold_index": hold_idx,
                        "ft_wrist": np.arange(6, dtype=np.float32),
                        "tau_J_d": np.arange(7, dtype=np.float32) + 20,
                        "joint_pos": np.arange(7, dtype=np.float32) + 40,
                        "tcp_velocity": np.zeros(6, dtype=np.float32),
                        "action_wrench_ee": np.zeros(6, dtype=np.float32),
                        "tcp_pos": np.ones(3, dtype=np.float32),
                        "tcp_pose_4x4": np.eye(4, dtype=np.float32).reshape(-1),
                        "task_prop_gains": np.full(6, 50.0, dtype=np.float32),
                        "task_deriv_gains": np.full(6, 15.0, dtype=np.float32),
                        "target_pose_4x4": np.eye(4, dtype=np.float32).reshape(-1),
                        "hold_number": np.eye(2, dtype=np.float32)[hold_idx],
                        "direction": np.ones(1, dtype=np.float32),
                        "phase": 1,
                        "phase_name": "hold",
                        "sample_label": "hold",
                        "amplitude_m": 0.01 * (hold_idx + 1),
                        "excitation_direction": np.array([0, 1, 0], dtype=np.float32),
                    })
            _write_with_metadata(robot_path, robot_rows, {
                "episode_id": "episode-test",
                "rest_reference_timestamp": 100.0,
            })

            tracking_rows = []
            for timestamp, axis in (
                (99.9, np.array([1.0, 0.0, 0.0])),
                (100.0, np.array([1.0, 0.0, 0.0])),
                (101.0, np.array([1.0, 0.0, 0.0])),
                (101.1, np.array([1.0, 0.0, 0.0])),
                (102.0, np.array([0.0, 1.0, 0.0])),
                (102.1, np.array([0.0, 1.0, 0.0])),
            ):
                for name, scale in (("Branch", 1.0), ("Spur", 2.0), ("Apple", 3.0)):
                    pos = axis * scale
                    tracking_rows.append({
                        "timestamp": timestamp,
                        "name": name,
                        "x": pos[0], "y": pos[1], "z": pos[2],
                        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                    })
            _write_with_metadata(tracking_path, tracking_rows, {
                "coordinate_frame": "franka_base_o",
                "camera_to_base_4x4_used": np.eye(4, dtype=np.float64).tolist(),
            })

            compile_static_episode(
                robot_path,
                tracking_path,
                output_path,
                camera_frame_count=2,
                max_camera_delta_s=0.25,
                command_argv=["test"],
            )

            output = pq.read_table(output_path)
            rows = output.to_pylist()
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["episode_id"], "episode-test")
            for row in rows:
                self.assertNotIn("woody_part_start_pos", row)
                self.assertNotIn("woody_part_end_pos", row)
                self.assertNotIn("woody_bending_angles", row)
            self.assertEqual(rows[-1]["camera_frame_count"], 2)

            metadata = json.loads(
                output.schema.metadata[b"dataset_metadata"].decode("utf-8")
            )
            # tracking without a recorded selection = the old Branch/Spur/Apple layout; no chords
            self.assertEqual(metadata["topology"]["tracked_names"], ["Branch", "Spur", "Apple"])
            self.assertEqual(metadata["camera_aggregation"]["required_tracker_names"], ["Branch", "Spur", "Apple"])
            for key in ("rest_chord_vectors", "bending_definition", "rest_woody_part_start_pos"):
                self.assertNotIn(key, metadata)
            self.assertNotIn("bending_angles_rad", metadata["hold_camera_summaries"][0])
            self.assertEqual(metadata["camera_aggregation"]["requested_frame_count"], 2)
            self.assertEqual(metadata["coordinate_frame"], "franka_base_o")
            self.assertEqual(metadata["camera_to_base_4x4_used"], np.eye(4).tolist())
            self.assertIn("source_files", metadata)
            self.assertIn("source_metadata_summary", metadata)
            self.assertIn("tau_J_d", output.schema.names)
            self.assertIn("joint_pos", output.schema.names)
            self.assertIn("tcp_pose_4x4", output.schema.names)
            self.assertIn("task_prop_gains", output.schema.names)
            self.assertIn("task_deriv_gains", output.schema.names)
            self.assertIn("target_pose_4x4", output.schema.names)
            self.assertIn("branch_pose_4x4", output.schema.names)
            self.assertIn("spur_pose_4x4", output.schema.names)
            self.assertIn("apple_pose_4x4", output.schema.names)
            self.assertNotIn("woody_part_start_pos", output.schema.names)
            self.assertNotIn("woody_part_end_pos", output.schema.names)
            self.assertNotIn("woody_bending_angles", output.schema.names)
            self.assertIn("sample_label", output.schema.names)

    def test_rejects_tracking_files_not_in_base_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            robot_path = tmp / "robot.parquet"
            tracking_path = tmp / "tracking.parquet"
            output_path = tmp / "unified.parquet"

            robot_rows = [{
                "timestamp": 101.0,
                "hold_step_idx": 0,
                "hold_index": 0,
                "ft_wrist": np.zeros(6, dtype=np.float32),
                "tau_J_d": np.zeros(7, dtype=np.float32),
                "joint_pos": np.zeros(7, dtype=np.float32),
                "tcp_velocity": np.zeros(6, dtype=np.float32),
                "action_wrench_ee": np.zeros(6, dtype=np.float32),
                "tcp_pos": np.zeros(3, dtype=np.float32),
                "tcp_pose_4x4": np.eye(4, dtype=np.float32).reshape(-1),
                "task_prop_gains": np.ones(6, dtype=np.float32),
                "task_deriv_gains": np.ones(6, dtype=np.float32),
                "target_pose_4x4": np.eye(4, dtype=np.float32).reshape(-1),
                "hold_number": np.array([1.0], dtype=np.float32),
                "direction": np.array([1.0], dtype=np.float32),
                "phase": 1,
                "phase_name": "hold",
                "sample_label": "hold",
                "amplitude_m": 0.01,
                "excitation_direction": np.zeros(3, dtype=np.float32),
            }]
            _write_with_metadata(robot_path, robot_rows, {"episode_id": "episode-test"})

            tracking_rows = []
            for name in ("Branch", "Spur", "Apple"):
                tracking_rows.append({
                    "timestamp": 101.0,
                    "name": name,
                    "x": 1.0, "y": 0.0, "z": 0.0,
                    "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                })
            _write_with_metadata(tracking_path, tracking_rows, {
                "coordinate_frame": "camera_color_optical_frame",
                "camera_to_base_4x4_used": np.eye(4, dtype=np.float64).tolist(),
            })

            with self.assertRaises(ValueError):
                compile_static_episode(
                    robot_path,
                    tracking_path,
                    output_path,
                    camera_frame_count=1,
                    max_camera_delta_s=0.25,
                    command_argv=["test"],
                )


if __name__ == "__main__":
    unittest.main()


def _robot_rows():
    rows = []
    for hold_idx, base_time in enumerate((101.0, 102.0)):
        for hold_step_idx, timestamp in enumerate((base_time, base_time + 0.1)):
            rows.append({
                "timestamp": timestamp,
                "hold_step_idx": hold_step_idx,
                "hold_index": hold_idx,
                "ft_wrist": np.zeros(6, dtype=np.float32),
                "tau_J_d": np.zeros(7, dtype=np.float32),
                "joint_pos": np.zeros(7, dtype=np.float32),
                "tcp_velocity": np.zeros(6, dtype=np.float32),
                "action_wrench_ee": np.zeros(6, dtype=np.float32),
                "tcp_pos": np.ones(3, dtype=np.float32),
                "tcp_pose_4x4": np.eye(4, dtype=np.float32).reshape(-1),
                "task_prop_gains": np.full(6, 50.0, dtype=np.float32),
                "task_deriv_gains": np.full(6, 15.0, dtype=np.float32),
                "target_pose_4x4": np.eye(4, dtype=np.float32).reshape(-1),
                "hold_number": np.eye(2, dtype=np.float32)[hold_idx],
                "direction": np.ones(1, dtype=np.float32),
                "phase": 1,
                "phase_name": "hold",
                "sample_label": "hold",
                "amplitude_m": 0.01,
                "excitation_direction": np.array([0, 1, 0], dtype=np.float32),
            })
    return rows


PARTS = {
    "primary": {"radius_m": 0.02},
    "spur": {"radius_m": 0.004},
    "stem": {"radius_m": 0.001},
    "apple": {"radius_m": 0.035},
}


class TrackerSelectionCompileTest(unittest.TestCase):
    """Detector --tags selections: compile follows the tracker names in the tracking metadata."""

    def _compile(self, names, radius_shift=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        tmp = Path(directory.name)
        robot_path, tracking_path, output_path = tmp / "robot.parquet", tmp / "tracking.parquet", tmp / "out.parquet"
        _write_with_metadata(robot_path, _robot_rows(), {"episode_id": "e", "rest_reference_timestamp": 100.0})
        tracking_rows = []
        for timestamp in (99.9, 100.0, 101.0, 101.1, 102.0, 102.1):
            for index, name in enumerate(names):
                # identity rotation: tag +z is base +z, so the radius shift is along base z
                tracking_rows.append({
                    "timestamp": timestamp, "name": name,
                    "x": 0.1 * (index + 1), "y": 0.5, "z": 0.3,
                    "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                })
        # a deselected tracker's rows would never be written; an unrelated name is ignored
        tracking_rows.append({"timestamp": 100.0, "name": "Other", "x": 1.0, "y": 1.0, "z": 1.0,
                              "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0})
        _write_with_metadata(tracking_path, tracking_rows, {
            "coordinate_frame": "franka_base_o",
            "camera_to_base_4x4_used": np.eye(4).tolist(),
            "tracker_names": list(names),
            "tracking_config": {"radius_shift": radius_shift or {}},
        })
        compile_static_episode(robot_path, tracking_path, output_path, camera_frame_count=2,
                               max_camera_delta_s=0.25, parts=PARTS, command_argv=["test"])
        table = pq.read_table(output_path)
        metadata = json.loads(table.schema.metadata[b"dataset_metadata"].decode("utf-8"))
        return table, table.to_pylist(), metadata

    def test_new_tag_layout_shifts_each_tracker_by_its_part_radius(self):
        names = ["Branch", "SpurStart", "SpurEnd", "StemStart", "Apple"]
        table, rows, metadata = self._compile(names, radius_shift={"Branch": False})
        for key in ("branch", "spur_start", "spur_end", "stem_start"):
            self.assertIn(f"{key}_pose_4x4", table.schema.names)
            self.assertIn(f"{key}_pose_4x4_tag", table.schema.names)
        self.assertNotIn("spur_pose_4x4", table.schema.names)
        expected_shift = {"branch": 0.0, "spur_start": 0.004, "spur_end": 0.004, "stem_start": 0.001}
        for key, shift in expected_shift.items():
            pose = np.asarray(rows[0][f"{key}_pose_4x4"]).reshape(4, 4)
            tag = np.asarray(rows[0][f"{key}_pose_4x4_tag"]).reshape(4, 4)
            np.testing.assert_allclose(pose[:3, 3] - tag[:3, 3], [0, 0, shift], atol=1e-6, err_msg=key)
        np.testing.assert_allclose(np.asarray(rows[0]["apple_pos"]) - rows[0]["apple_pos_tag"], [0, 0, 0.035], atol=1e-6)
        correction = metadata["tag_to_part_correction"]
        self.assertEqual(correction["radius_m"]["Branch"], 0.0)
        self.assertFalse(correction["radius_shift"]["Branch"])
        self.assertEqual(correction["part_for_tracker"]["SpurEnd"], "spur")
        self.assertEqual(correction["part_for_tracker"]["StemStart"], "stem")
        self.assertEqual(metadata["topology"]["tracked_names"], names)
        self.assertNotIn("rest_chord_vectors", metadata)

    def test_selection_without_branch_compiles(self):
        table, rows, metadata = self._compile(["SpurStart", "Apple"])
        self.assertIn("spur_start_pose_4x4", table.schema.names)
        self.assertNotIn("branch_pose_4x4", table.schema.names)
        self.assertEqual(len(rows), 4)
        self.assertEqual(metadata["camera_aggregation"]["required_tracker_names"], ["SpurStart", "Apple"])

    def test_tracker_without_a_part_needs_radius_shift_off(self):
        with self.assertRaisesRegex(ValueError, "Reserved"):
            self._compile(["Reserved", "Apple"])
        table, _, _ = self._compile(["Reserved", "Apple"], radius_shift={"Reserved": False})
        self.assertIn("reserved_pose_4x4", table.schema.names)
