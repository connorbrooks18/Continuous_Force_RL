import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from real_robot_exps.manifest_run_length import compute_manifest_duration


def _write_parquet(path: Path, timestamps: list[float]) -> None:
    table = pa.Table.from_pylist([{"timestamp": t} for t in timestamps])
    pq.write_table(table, path)


class ManifestRunLengthTest(unittest.TestCase):
    def test_computes_total_duration_from_first_and_last_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runs_dir = tmp / "runs"
            runs_dir.mkdir()

            unified_a = runs_dir / "s00-d00.parquet"
            unified_b = runs_dir / "s00-d01.parquet"
            _write_parquet(unified_a, [10.0, 12.0, 16.0])
            _write_parquet(unified_b, [20.0, 20.5, 21.0, 23.0])

            manifest = {
                "runs": [
                    {"run_id": "s00-d00", "files": {"unified": "runs/s00-d00.parquet"}},
                    {"run_id": "s00-d01", "files": {"unified": "runs/s00-d01.parquet"}},
                ]
            }
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            summary = compute_manifest_duration(manifest_path)

            self.assertEqual(summary["n_runs"], 2)
            self.assertAlmostEqual(summary["first_timestamp"], 10.0)
            self.assertAlmostEqual(summary["last_timestamp"], 23.0)
            self.assertAlmostEqual(summary["total_seconds"], 13.0)
            self.assertAlmostEqual(summary["total_minutes"], 13.0 / 60.0)

    def test_falls_back_to_robot_file_when_unified_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            robot_path = tmp / "s00-d00_robot.parquet"
            _write_parquet(robot_path, [100.0, 101.25])

            manifest = {
                "runs": [
                    {"run_id": "s00-d00", "files": {"robot": "s00-d00_robot.parquet"}},
                ]
            }
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            summary = compute_manifest_duration(manifest_path)

            self.assertAlmostEqual(summary["first_timestamp"], 100.0)
            self.assertAlmostEqual(summary["last_timestamp"], 101.25)
            self.assertAlmostEqual(summary["total_seconds"], 1.25)


if __name__ == "__main__":
    unittest.main()
