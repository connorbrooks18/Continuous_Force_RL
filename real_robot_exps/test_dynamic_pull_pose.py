import unittest

import numpy as np
import math

from real_robot_exps.apple_pullto_static import _load_dynamic_pull_start_pose


class DynamicPullPoseTest(unittest.TestCase):
    def test_uses_settled_snapshot_and_apple_radius_from_structure_metadata(self):
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

        pose, name, radius_m, surface_pose = _load_dynamic_pull_start_pose(
            run_metadata,
            fallback,
            theta=math.pi / 2.0,
            phi=math.pi / 2.0,
        )

        np.testing.assert_allclose(pose[:3, 3], np.array([0.4, 0.445, 0.6], dtype=np.float64))
        np.testing.assert_allclose(surface_pose[:3, 3], np.array([0.4, 0.465, 0.6], dtype=np.float64))
        np.testing.assert_allclose(pose[:3, 2], np.array([0.0, 1.0, 0.0], dtype=np.float64), atol=1e-6)
        np.testing.assert_allclose(surface_pose[:3, 2], np.array([0.0, 1.0, 0.0], dtype=np.float64), atol=1e-6)
        self.assertEqual(name, "settled_snapshot_apple_surface_plus_2cm_pull_direction_offset")
        self.assertEqual(radius_m, 0.035)

    def test_legacy_lengthened_snapshot_is_still_accepted(self):
        fallback = np.eye(4, dtype=np.float64)
        fallback[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        run_metadata = {
            "pre_grasp_geometry": {
                "parts": {
                    "apple": {
                        "radius_m": 0.035,
                    }
                },
                "lengthened_snapshot": {
                    "apple_pos": [0.4, 0.5, 0.6],
                },
            }
        }

        pose, name, radius_m, surface_pose = _load_dynamic_pull_start_pose(
            run_metadata,
            fallback,
            theta=math.pi / 2.0,
            phi=math.pi / 2.0,
        )

        np.testing.assert_allclose(pose[:3, 3], np.array([0.4, 0.445, 0.6], dtype=np.float64))
        np.testing.assert_allclose(surface_pose[:3, 3], np.array([0.4, 0.465, 0.6], dtype=np.float64))
        self.assertEqual(name, "lengthened_snapshot_apple_surface_plus_2cm_pull_direction_offset")
        self.assertEqual(radius_m, 0.035)

    def test_falls_back_when_snapshot_or_radius_missing(self):
        fallback = np.eye(4, dtype=np.float64)
        fallback[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        pose, name, radius_m, surface_pose = _load_dynamic_pull_start_pose({}, fallback)

        np.testing.assert_allclose(pose, fallback)
        np.testing.assert_allclose(surface_pose, fallback)
        self.assertEqual(name, "apple_pose_4x4")
        self.assertIsNone(radius_m)


if __name__ == "__main__":
    unittest.main()
