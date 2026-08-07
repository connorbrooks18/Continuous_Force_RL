import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from real_robot_exps.print_apple_tcp_base import _print_parquet_pose_report


def _pose_4x4(x: float, y: float, z: float) -> list[float]:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return pose.reshape(-1).tolist()


class PrintAppleTcpBaseTest(unittest.TestCase):
    def test_parquet_report_prints_all_pose_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.parquet"
            table = pa.Table.from_pylist([{"row_kind": None, "metadata_json": None, "timestamp": 1.23}])
            metadata = {
                "schema_name": "real_static_sysid_robot_raw",
                "pre_grasp_geometry": {
                    "snapshot": {
                        "apple_pose_4x4": _pose_4x4(3.0, 4.0, 5.0),
                        "branch_pose_4x4": _pose_4x4(1.0, 2.0, 3.0),
                        "spur_pose_4x4": _pose_4x4(2.0, 3.0, 4.0),
                        "apple_pos": [3.0, 4.0, 5.0],
                        "branch_pos": [1.0, 2.0, 3.0],
                        "spur_pos": [2.0, 3.0, 4.0],
                    },
                    "robot_snapshot": {
                        "tcp_pose_4x4": _pose_4x4(6.0, 7.0, 8.0),
                        "tcp_pos": [6.0, 7.0, 8.0],
                    },
                },
                "post_grasp_geometry": {},
            }
            table = table.replace_schema_metadata({
                b"dataset_metadata": json.dumps(metadata).encode("utf-8"),
            })
            pq.write_table(table, path)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                _print_parquet_pose_report(path)

            output = buffer.getvalue()
            self.assertIn("tcp:", output)
            self.assertIn("apple:", output)
            self.assertIn("branch:", output)
            self.assertIn("spur:", output)
            self.assertIn("Legacy diagnostics:", output)
            self.assertIn("pre_grasp_tcp_apple_distance_m:", output)
            self.assertIn("source: pre_grasp_geometry.robot_snapshot", output)
            self.assertIn("source: pre_grasp_geometry.snapshot", output)
            self.assertIn("[6. 7. 8.]", output)
            self.assertIn("[3. 4. 5.]", output)


if __name__ == "__main__":
    unittest.main()
