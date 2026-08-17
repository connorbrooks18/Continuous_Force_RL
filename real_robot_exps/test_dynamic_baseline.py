import tempfile
import unittest
from pathlib import Path

import numpy as np

from real_robot_exps.apple_pullto_static import (
    _default_baseline_path,
    _effective_manual_setup,
    _load_baseline_front_of_apple_pose,
    apply_dynamic_baseline,
    save_robot_hold_parquet,
    _validate_baseline_compatibility,
)


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

    def test_validate_baseline_compatibility_allows_setup_changes_but_requires_core_params(self):
        current = {
            "theta_rad": 1.57,
            "phi_rad": 1.57,
            "distance_m": 0.05,
            "n_holds": 5,
            "pull_start_pose_name": "manual_setup_current_tcp_pose",
            "pull_surface_pose_name": "manual_setup_current_tcp_pose",
            "robot_start_pose_4x4": np.eye(4, dtype=np.float64).tolist(),
            "dump": {"robot_info": {"kp": 100.0}},
            "pre_grasp_geometry": {
                "snapshot": {"apple_pos": [0.10, 0.20, 0.30]},
                "structure_index": 0,
                "structure_name": "structure_a",
            },
        }
        baseline = {
            "theta_rad": 1.57,
            "phi_rad": 1.57,
            "distance_m": 0.07,
            "n_holds": 4,
            "pull_start_pose_name": "some_other_pose",
            "pull_surface_pose_name": "some_other_pose",
            "robot_start_pose_4x4": (np.eye(4, dtype=np.float64) * 2.0).tolist(),
            "dump": {"robot_info": {"kp": 100.0}},
            "pre_grasp_geometry": {
                "snapshot": {"apple_pos": [0.10, 0.20, 0.30]},
                "structure_index": 99,
                "structure_name": "structure_b",
            },
        }

        _validate_baseline_compatibility(current, baseline, Path("baseline.parquet"))

        baseline["dump"]["robot_info"]["kp"] = 80.0
        with self.assertRaises(ValueError):
            _validate_baseline_compatibility(current, baseline, Path("baseline.parquet"))

        baseline["dump"]["robot_info"]["kp"] = 100.0
        baseline["theta_rad"] = 0.0
        with self.assertRaises(ValueError):
            _validate_baseline_compatibility(current, baseline, Path("baseline.parquet"))

    def test_default_baseline_path_uses_kp_suffix(self):
        self.assertEqual(
            _default_baseline_path("pull_theta1.57_phi1.57", 100.0),
            Path("pull_theta1.57_phi1.57_kp100_baseline_robot.parquet"),
        )
        self.assertEqual(
            _default_baseline_path("pull_theta1.57_phi1.57", None),
            Path("pull_theta1.57_phi1.57_baseline_robot.parquet"),
        )

    def test_effective_manual_setup_is_disabled_for_baseline_mode(self):
        self.assertTrue(_effective_manual_setup("collect", True))
        self.assertFalse(_effective_manual_setup("baseline", True))
        self.assertFalse(_effective_manual_setup("baseline", False))
        self.assertFalse(_effective_manual_setup("collect", False))

    def test_baseline_front_pose_uses_apple_translation_and_radius(self):
        fallback = np.eye(4, dtype=np.float64)
        fallback[:3, 0] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        fallback[:3, 1] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        fallback[:3, 2] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        run_metadata = {
            "pre_grasp_geometry": {
                "parts": {"apple": {"radius_m": 0.035}},
                "settled_snapshot": {"apple_pos": [0.4, 0.5, 0.6]},
            }
        }

        pose, name, radius_m, surface_pose = _load_baseline_front_of_apple_pose(run_metadata, fallback)

        np.testing.assert_allclose(pose[:3, 3], np.array([0.4, 0.465, 0.6], dtype=np.float64))
        np.testing.assert_allclose(surface_pose[:3, 3], np.array([0.4, 0.465, 0.6], dtype=np.float64))
        np.testing.assert_allclose(pose[:3, :3], fallback[:3, :3])
        self.assertEqual(name, "settled_snapshot_front_of_apple_pose")
        self.assertEqual(radius_m, 0.035)


if __name__ == "__main__":
    unittest.main()
