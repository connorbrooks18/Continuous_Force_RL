import unittest

import numpy as np

from real_robot_exps.remake_translation_matrix import solve_reference_tag_to_base_translation


class RemakeTranslationMatrixTest(unittest.TestCase):
    def test_solves_translation_from_reference_and_tcp_frames(self):
        apple_center_ref = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        tcp_pos_base = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        base_rotation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ], dtype=np.float64)

        translation = solve_reference_tag_to_base_translation(
            apple_center_ref=apple_center_ref,
            tcp_pos_base=tcp_pos_base,
            base_rotation=base_rotation,
            apple_to_tcp_distance_m=0.04,
        )

        np.testing.assert_allclose(translation, np.array([0.9, 1.74, 3.2], dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
