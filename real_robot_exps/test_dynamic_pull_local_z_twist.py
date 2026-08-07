import math
import unittest

import numpy as np

from real_robot_exps.apple_pullto_static import _look_at_rotation_toward_apple


class DynamicPullLocalZTwistTest(unittest.TestCase):
    def test_fixed_twist_is_applied_about_local_z_only(self):
        position = np.array([0.21, 0.47, 0.18], dtype=np.float64)
        apple_center = np.array([0.58, 0.93, 0.51], dtype=np.float64)
        fallback = np.eye(3, dtype=np.float64)

        untwisted = _look_at_rotation_toward_apple(
            position,
            apple_center,
            fallback,
            local_z_twist_deg=0.0,
        )
        twisted = _look_at_rotation_toward_apple(
            position,
            apple_center,
            fallback,
            local_z_twist_deg=18.5,
        )

        forward = apple_center - position
        forward /= np.linalg.norm(forward)

        np.testing.assert_allclose(twisted[:, 2], forward, atol=1e-6)
        np.testing.assert_allclose(untwisted[:, 2], forward, atol=1e-6)

        relative = untwisted.T @ twisted
        angle = math.radians(18.5)
        expected = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        np.testing.assert_allclose(relative, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
