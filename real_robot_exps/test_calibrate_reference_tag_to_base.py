import unittest
import tempfile
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from real_robot_exps.calibrate_camera_to_base import (
    _load_handeye_calibration,
)


class CalibrateReferenceTagToBaseTest(unittest.TestCase):
    @staticmethod
    def _quat_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
        q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
        q = q / np.linalg.norm(q)
        x, y, z, w = q
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

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
            camera_to_base, meta = _load_handeye_calibration(path)

        self.assertEqual(meta["parameters"]["calibration_type"], "eye_on_base")
        np.testing.assert_allclose(camera_to_base[:3, 3], np.array([-0.4, 0.5, 0.6], dtype=np.float64))
        np.testing.assert_allclose(
            camera_to_base[:3, :3],
            self._quat_xyzw_to_rotmat(np.array([-0.1, 0.2, -0.3, 0.9], dtype=np.float64)),
        )

    def test_script_prints_camera_to_base_by_default(self):
        payload = {
            "parameters": {
                "calibration_type": "eye_on_base",
                "robot_base_frame": "fr3_link0",
                "tracking_base_frame": "camera_color_optical_frame",
            },
            "transform": {
                "translation": {"x": -0.4, "y": 0.5, "z": 0.6},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calib.calib"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "real_robot_exps.calibrate_camera_to_base",
                    "--calib-file",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("Camera pose in base frame (camera -> base):", proc.stdout)
        self.assertIn("CAMERA_TO_BASE_4X4_DEFAULT", proc.stdout)
        self.assertIn("-0.4000000000000000", proc.stdout)


if __name__ == "__main__":
    unittest.main()
