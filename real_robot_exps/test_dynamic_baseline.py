import tempfile
import unittest
from pathlib import Path

import numpy as np

from real_robot_exps.apple_pullto_static import apply_dynamic_baseline, save_robot_hold_parquet


class DynamicBaselineTest(unittest.TestCase):
    def test_interpolates_each_hold_and_preserves_raw_wrench(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.parquet"
            baseline_rows = [
                {"hold_index": 0, "hold_step_idx": 0, "ft_wrist": np.zeros(6, dtype=np.float32)},
                {"hold_index": 0, "hold_step_idx": 1, "ft_wrist": np.full(6, 2.0, dtype=np.float32)},
            ]
            save_robot_hold_parquet(baseline_rows, baseline_path, {"episode_id": "baseline"})
            collect_rows = [
                {"hold_index": 0, "hold_step_idx": idx, "ft_wrist": np.full(6, 5.0, dtype=np.float32)}
                for idx in range(3)
            ]

            metadata = apply_dynamic_baseline(collect_rows, baseline_path)

            np.testing.assert_allclose(collect_rows[0]["ft_wrist_baseline"], 0.0)
            np.testing.assert_allclose(collect_rows[1]["ft_wrist_baseline"], 1.0)
            np.testing.assert_allclose(collect_rows[2]["ft_wrist_baseline"], 2.0)
            np.testing.assert_allclose(collect_rows[1]["ft_wrist_raw"], 5.0)
            np.testing.assert_allclose(collect_rows[1]["ft_wrist"], 4.0)
            self.assertEqual(metadata["method"], "per-hold normalized-time linear interpolation")


if __name__ == "__main__":
    unittest.main()
