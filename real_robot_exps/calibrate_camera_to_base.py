"""Load an eye-on-base calibration and print the camera->base transform.

The calibration file already stores the hand-eye solution as a translation and
quaternion. This helper converts that pose into a 4x4 homogeneous matrix,
prints the camera->base result, and can optionally invert it for debugging.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


def _format_matrix(mat: np.ndarray) -> str:
    return np.array2string(np.asarray(mat, dtype=np.float64), precision=8, suppress_small=True)


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    tf[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return tf


def _quat_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Quaternion norm must be non-zero")
    x, y, z, w = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _load_handeye_calibration(calib_path: Path) -> tuple[np.ndarray, dict[str, object]]:
    data = yaml.safe_load(calib_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Calibration file {calib_path} did not parse as a mapping")

    params = dict(data.get("parameters", {}) or {})
    transform = dict(data.get("transform", {}) or {})
    calib_type = str(params.get("calibration_type", ""))
    if calib_type != "eye_on_base":
        raise ValueError(
            f"Expected eye_on_base calibration in {calib_path}, got calibration_type={calib_type!r}"
        )

    translation = transform.get("translation", {})
    rotation = transform.get("rotation", {})
    base_translation = np.array([
        float(translation.get("x", 0.0)),
        float(translation.get("y", 0.0)),
        float(translation.get("z", 0.0)),
    ], dtype=np.float64)
    base_quaternion_xyzw = np.array([
        float(rotation.get("x", 0.0)),
        float(rotation.get("y", 0.0)),
        float(rotation.get("z", 0.0)),
        float(rotation.get("w", 1.0)),
    ], dtype=np.float64)
    camera_to_base = _make_transform(_quat_xyzw_to_rotmat(base_quaternion_xyzw), base_translation)
    return camera_to_base, data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the eye-on-base calibration file and print camera->base.",
    )
    parser.add_argument(
        "--calib-file",
        type=Path,
        default=Path.home() / ".ros2/easy_handeye2/calibrations/fr3_eye_on_base_calib.calib",
        help="easy_handeye2 .calib file containing the eye-on-base hand-eye solution",
    )
    parser.add_argument(
        "--calibration-transform-direction",
        choices=("camera_to_base", "base_to_camera"),
        default="camera_to_base",
        help=(
            "Interpret the .calib transform as camera_to_base (default, matching "
            "the saved calibration matrix) or base_to_camera."
        ),
    )
    args = parser.parse_args()

    raw_transform, calib_meta = _load_handeye_calibration(args.calib_file)
    calib_source = str(args.calib_file)

    print(f"Using hand-eye calibration from: {calib_source}")
    if calib_meta:
        params = dict(calib_meta.get("parameters", {}) or {})
        print(
            "Calibration frames: "
            f"robot_base_frame={params.get('robot_base_frame')}, "
            f"tracking_base_frame={params.get('tracking_base_frame')}, "
            f"calibration_type={params.get('calibration_type')}"
        )

    if args.calibration_transform_direction == "camera_to_base":
        camera_to_base = raw_transform
        base_to_camera = np.linalg.inv(raw_transform)
    else:
        base_to_camera = raw_transform
        camera_to_base = np.linalg.inv(raw_transform)

    print("\nBase -> camera (debug only):")
    print(_format_matrix(base_to_camera))
    print("\nCamera pose in base frame (camera -> base):")
    print(_format_matrix(camera_to_base))
    print("\nPaste into real_robot_exps/static_constants.py as:")
    print("CAMERA_TO_BASE_4X4_DEFAULT = np.array([")
    for row in camera_to_base:
        print("    [" + ", ".join(f"{value:.16f}" for value in row) + "],")
    print("], dtype=np.float64)")


if __name__ == "__main__":
    main()
