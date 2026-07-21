import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from real_robot_exps.viz_static_sysid import _load_plot_data, plot_static_sysid


class VizStaticSysidTest(unittest.TestCase):
    def test_builds_figure_from_unified_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = tmp / "unified.parquet"
            rows = []
            for idx, timestamp in enumerate((1.0, 1.1, 2.0, 2.1)):
                rows.append({
                    "episode_id": "episode-test",
                    "timestamp": timestamp,
                    "step_idx": idx,
                    "hold_step_idx": idx % 2,
                    "hold_index": idx // 2,
                    "ft_wrist": np.arange(6, dtype=np.float32),
                    "tau_J": np.arange(7, dtype=np.float32),
                    "tau_ext_hat_filtered": np.arange(7, dtype=np.float32) + 10,
                    "tau_J_d": np.arange(7, dtype=np.float32) + 20,
                    "gravity_torques": np.arange(7, dtype=np.float32) + 30,
                    "tcp_velocity": np.zeros(6, dtype=np.float32),
                    "action": np.zeros(6, dtype=np.float32),
                    "tcp_pos": np.ones(3, dtype=np.float32),
                    "apple_pos": np.ones(3, dtype=np.float32) * idx,
                    "woody_part_start_pos": np.arange(9, dtype=np.float32),
                    "woody_part_end_pos": np.arange(9, dtype=np.float32) + 1,
                    "woody_bending_angles": np.full(3, 0.1 * idx, dtype=np.float32),
                    "hold_number": np.eye(2, dtype=np.float32)[idx // 2],
                    "direction": np.eye(1, dtype=np.float32)[0],
                    "phase": 1,
                    "phase_name": "hold",
                    "amplitude_m": 0.01,
                    "excitation_direction": np.array([0, 1, 0], dtype=np.float32),
                    "camera_timestamp": timestamp - 0.01,
                    "robot_camera_timestamp_offset_s": 0.01,
                    "camera_window_start_timestamp": timestamp - 0.02,
                    "camera_window_end_timestamp": timestamp,
                    "camera_frame_count": 2,
                    "camera_selected_timestamps": [timestamp - 0.02, timestamp],
                    "camera_data_valid": True,
                })

            table = pa.Table.from_pylist(rows)
            table = table.replace_schema_metadata({
                b"dataset_metadata": json.dumps({"schema_name": "real_static_sysid_episode"}).encode("utf-8")
            })
            pq.write_table(table, path)

            data = _load_plot_data(path)
            fig = plot_static_sysid(data)
            self.assertEqual(len(fig.axes), 6)

    def test_builds_figure_from_arm_only_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = tmp / "arm_only.parquet"
            rows = []
            for idx, timestamp in enumerate((1.0, 1.1, 2.0)):
                rows.append({
                    "timestamp": timestamp,
                    "hold_step_idx": idx,
                    "hold_index": idx // 2,
                    "ft_wrist": np.arange(6, dtype=np.float32),
                    "tau_J": np.arange(7, dtype=np.float32),
                    "tau_ext_hat_filtered": np.arange(7, dtype=np.float32) + 10,
                    "tau_J_d": np.arange(7, dtype=np.float32) + 20,
                    "gravity_torques": np.arange(7, dtype=np.float32) + 30,
                    "tcp_velocity": np.zeros(6, dtype=np.float32),
                    "action": np.zeros(6, dtype=np.float32),
                    "tcp_pos": np.ones(3, dtype=np.float32) * idx,
                    "hold_number": np.eye(2, dtype=np.float32)[idx // 2],
                    "direction": np.eye(1, dtype=np.float32)[0],
                    "phase": 1,
                    "phase_name": "hold",
                    "amplitude_m": 0.01,
                    "excitation_direction": np.array([0, 1, 0], dtype=np.float32),
                })

            table = pa.Table.from_pylist(rows)
            table = table.replace_schema_metadata({
                b"dataset_metadata": json.dumps({"schema_name": "real_static_sysid_robot_raw"}).encode("utf-8")
            })
            pq.write_table(table, path)

            data = _load_plot_data(path)
            self.assertFalse(data.has_camera)
            fig = plot_static_sysid(data)
            self.assertEqual(len(fig.axes), 4)


if __name__ == "__main__":
    unittest.main()
