import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from real_robot_exps.recompile_static_sysid_batch import recompile_structure


def _write_with_metadata(path: Path, metadata: dict, rows: list[dict] | None = None) -> None:
    table = pa.Table.from_pylist(rows or [{"value": 1}])
    table = table.replace_schema_metadata(
        {b"dataset_metadata": json.dumps(metadata).encode("utf-8")}
    )
    pq.write_table(table, path)


class RecompileStaticSysidBatchTest(unittest.TestCase):
    def test_recompile_structure_pairs_robot_tracking_and_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_dir = tmp / "input"
            baseline_dir = tmp / "baselines"
            output_dir = tmp / "recompiled"
            input_dir.mkdir()
            baseline_dir.mkdir()

            robot_path = input_dir / "s11-d07_robot.parquet"
            tracking_path = input_dir / "s11-d07_tracking.parquet"
            baseline_path = baseline_dir / "s11_pull_theta1.57_phi2.36_kp100_baseline_robot.parquet"

            _write_with_metadata(
                robot_path,
                {"structure": "s11", "theta_rad": 1.57, "phi_rad": 2.36},
            )
            _write_with_metadata(
                tracking_path,
                {"coordinate_frame": "franka_base_o", "camera_to_base_4x4_used": [[1.0] * 4] * 4},
            )
            _write_with_metadata(
                baseline_path,
                {"structure": "s11", "theta_rad": 1.57, "phi_rad": 2.36},
            )

            calls: list[tuple[Path, Path, Path, float, Path | None, list[str]]] = []
            viz_calls: list[list[str]] = []

            def fake_compile(robot, tracking, output, *, camera_ema_alpha, baseline_path, command_argv):
                calls.append((
                    Path(robot),
                    Path(tracking),
                    Path(output),
                    camera_ema_alpha,
                    Path(baseline_path) if baseline_path is not None else None,
                    list(command_argv),
                ))
                output = Path(output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("compiled")
                return output

            with mock.patch(
                "real_robot_exps.recompile_static_sysid_batch.compile_static_episode",
                side_effect=fake_compile,
            ), mock.patch(
                "real_robot_exps.recompile_static_sysid_batch.subprocess.run",
                side_effect=lambda cmd, check: viz_calls.append(list(cmd)),
            ), mock.patch(
                "real_robot_exps.recompile_static_sysid_batch.sys.argv",
                ["python", "-m", "real_robot_exps.recompile_static_sysid_batch", "--structure", "s11"],
            ):
                outputs = recompile_structure(
                    "s11",
                    input_dir=input_dir,
                    baseline_dir=baseline_dir,
                    output_dir=output_dir,
                    overwrite=False,
                    require_baseline=True,
                    camera_ema_alpha=1.0,
                )

            self.assertEqual(outputs, [output_dir / "s11-d07.parquet"])
            self.assertEqual(len(calls), 1)
            robot, tracking, output, alpha, baseline, argv = calls[0]
            self.assertEqual(robot, robot_path)
            self.assertEqual(tracking, tracking_path)
            self.assertEqual(output, output_dir / "s11-d07.parquet")
            self.assertEqual(alpha, 1.0)
            self.assertEqual(baseline, baseline_path)
            self.assertEqual(argv, ["python", "-m", "real_robot_exps.recompile_static_sysid_batch", "--structure", "s11"])
            self.assertEqual(len(viz_calls), 1)
            self.assertEqual(viz_calls[0][0], sys.executable)
            self.assertEqual(viz_calls[0][2], "real_robot_exps.viz_static_sysid")
            self.assertIn("--save", viz_calls[0])

    def test_recompile_structure_requires_baseline_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_dir = tmp / "input"
            baseline_dir = tmp / "baselines"
            output_dir = tmp / "recompiled"
            input_dir.mkdir()
            baseline_dir.mkdir()

            _write_with_metadata(
                input_dir / "s11-d07_robot.parquet",
                {"structure": "s11", "theta_rad": 1.57, "phi_rad": 2.36},
            )
            _write_with_metadata(
                input_dir / "s11-d07_tracking.parquet",
                {"coordinate_frame": "franka_base_o", "camera_to_base_4x4_used": [[1.0] * 4] * 4},
            )

            with self.assertRaises(FileNotFoundError):
                recompile_structure(
                    "s11",
                    input_dir=input_dir,
                    baseline_dir=baseline_dir,
                    output_dir=output_dir,
                    overwrite=False,
                    require_baseline=True,
                    camera_ema_alpha=1.0,
                )


if __name__ == "__main__":
    unittest.main()
