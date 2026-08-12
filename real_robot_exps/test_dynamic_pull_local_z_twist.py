import unittest

import numpy as np

from real_robot_exps.apple_pullto_static import _look_at_rotation_toward_apple


class DynamicPullFaceOnTest(unittest.TestCase):
    def test_dynamic_lineup_keeps_the_reference_apple_facing_rotation(self):
        position = np.array([0.21, 0.47, 0.18], dtype=np.float64)
        apple_center = np.array([0.58, 0.93, 0.51], dtype=np.float64)
        fallback = np.array([
            [-0.994, -0.110, 0.0],
            [0.0, 0.0, 1.0],
            [-0.110, 0.991, 0.0],
        ], dtype=np.float64)

        oriented = _look_at_rotation_toward_apple(
            position,
            apple_center,
            fallback,
            local_z_twist_deg=0.0,
        )
        oriented_with_twist_request = _look_at_rotation_toward_apple(
            position,
            apple_center,
            fallback,
            local_z_twist_deg=18.5,
        )

        np.testing.assert_allclose(oriented, fallback, atol=1e-6)
        np.testing.assert_allclose(oriented_with_twist_request, fallback, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
