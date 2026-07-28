import unittest
import tempfile
from pathlib import Path

import numpy as np
import yaml

from real_robot_exps.calibrate_reference_tag_to_base import (
    _load_handeye_calibration,
    solve_reference_tag_to_base,
)


class CalibrateReferenceTagToBaseTest(unittest.TestCase):
    def test_composes_base_camera_and_camera_reference_then_inverts(self):
        camera_pose_in_base = np.array([
            [0.0, -1.0, 0.0, 0.3],
            [1.0,  0.0, 0.0, 0.4],
            [0.0,  0.0, 1.0, 0.5],
            [0.0,  0.0, 0.0, 1.0],
        ], dtype=np.float64)
        reference_tag_to_camera = np.array([
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 0.0, -1.0, 0.2],
            [0.0, 1.0, 0.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        base_to_reference, reference_to_base = solve_reference_tag_to_base(
            camera_pose_in_base=camera_pose_in_base,
            reference_tag_to_camera=reference_tag_to_camera,
        )

        expected_base_to_reference = camera_pose_in_base @ reference_tag_to_camera
        np.testing.assert_allclose(base_to_reference, expected_base_to_reference)
        np.testing.assert_allclose(reference_to_base @ base_to_reference, np.eye(4), atol=1e-8)

    def test_loads_eye_on_base_calibration_file(self):
        payload = {
            "parameters": {
                "calibration_type": "eye_on_base",
                "robot_base_frame": "fr3_link0",
                "tracking_base_frame": "camera_color_optical_frame",
            },
            "transform": {
                "translation": {"x": -0.4, "y": 0.5, "z": 0.6},
                "rotation": {"x": -0.1, "y": 0.2, "z": -0.3, "w": 0.9},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calib.calib"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            base_to_camera, meta = _load_handeye_calibration(path)

        self.assertEqual(meta["parameters"]["calibration_type"], "eye_on_base")
        np.testing.assert_allclose(base_to_camera[:3, 3], np.array([-0.4, 0.5, 0.6], dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
