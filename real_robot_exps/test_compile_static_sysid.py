import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from real_robot_exps.compile_static_sysid import compile_static_episode


def _write_with_metadata(path, rows, metadata):
    table = pa.Table.from_pylist(rows)
    table = table.replace_schema_metadata({
        b"dataset_metadata": json.dumps(metadata).encode("utf-8")
    })
    pq.write_table(table, path)


class CompileStaticSysidTest(unittest.TestCase):
    def test_compiles_shared_endpoints_and_rest_relative_angles(self):
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
                        "action": np.zeros(6, dtype=np.float32),
                        "tcp_pos": np.ones(3, dtype=np.float32),
                        "tcp_pose_4x4": np.eye(4, dtype=np.float32).reshape(-1),
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
                "reference_tag_is_fruiting_base": True,
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
            np.testing.assert_allclose(rows[0]["woody_bending_angles"], np.zeros(3), atol=1e-7)
            np.testing.assert_allclose(
                rows[-1]["woody_bending_angles"], np.full(3, np.pi / 2), atol=1e-6
            )

            starts = np.asarray(rows[-1]["woody_part_start_pos"]).reshape(3, 3)
            ends = np.asarray(rows[-1]["woody_part_end_pos"]).reshape(3, 3)
            np.testing.assert_allclose(starts[0], starts[1])
            np.testing.assert_allclose(ends[0], starts[2])
            self.assertEqual(rows[-1]["camera_frame_count"], 2)

            metadata = json.loads(
                output.schema.metadata[b"dataset_metadata"].decode("utf-8")
            )
            self.assertEqual(metadata["topology"]["n_woody_parts"], 3)
            self.assertEqual(metadata["topology"]["node_order"], ["Branch", "Spur", "Apple"])
            self.assertEqual(metadata["camera_aggregation"]["requested_frame_count"], 2)
            self.assertIn("source_files", metadata)
            self.assertIn("source_metadata_summary", metadata)
            self.assertIn("tau_J_d", output.schema.names)
            self.assertIn("joint_pos", output.schema.names)
            self.assertIn("tcp_pose_4x4", output.schema.names)
            self.assertIn("target_pose_4x4", output.schema.names)
            self.assertIn("sample_label", output.schema.names)


if __name__ == "__main__":
    unittest.main()
