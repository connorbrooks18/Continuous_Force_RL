import unittest

import numpy as np

from real_robot_exps.apple_pullto_static import _load_dynamic_pull_start_pose


class DynamicPullPoseTest(unittest.TestCase):
    def test_uses_apple_radius_from_structure_metadata(self):
        fallback = np.eye(4, dtype=np.float64)
        fallback[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        run_metadata = {
            "pre_grasp_geometry": {
                "parts": {
                    "apple": {
                        "radius_m": 0.035,
                    }
                },
                "settled_snapshot": {
                    "apple_pos": [0.4, 0.5, 0.6],
                },
            }
        }

        pose, name, radius_m = _load_dynamic_pull_start_pose(run_metadata, fallback)

        np.testing.assert_allclose(pose[:3, 3], np.array([0.4, 0.465, 0.6], dtype=np.float64))
        self.assertEqual(name, "settled_snapshot_apple_center_plus_apple_radius_base_y_offset")
        self.assertEqual(radius_m, 0.035)

    def test_falls_back_when_snapshot_or_radius_missing(self):
        fallback = np.eye(4, dtype=np.float64)
        fallback[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        pose, name, radius_m = _load_dynamic_pull_start_pose({}, fallback)

        np.testing.assert_allclose(pose, fallback)
        self.assertEqual(name, "apple_pose_4x4")
        self.assertIsNone(radius_m)


if __name__ == "__main__":
    unittest.main()
