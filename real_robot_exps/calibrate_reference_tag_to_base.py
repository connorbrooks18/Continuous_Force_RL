"""Calibrate the reference-tag-to-base transform from a live camera snapshot.

Workflow:
    1. Load the eye-on-base hand-eye calibration as base->camera.
    2. Capture the reference AprilTag pose in the camera frame.
    3. Compose base->camera with tag->camera to get reference-tag->base.
    4. Invert that only for debugging the opposite direction.

The script prints a ready-to-paste ``REFERENCE_TAG_TO_BASE_4X4_DEFAULT`` block
for ``real_robot_exps.static_constants``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml


REFERENCE_TAG_ID = 1
TAG_SIZE_M = 0.0170


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


def _rotmat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    return quat / max(np.linalg.norm(quat), 1e-12)


def _prompt_vector(label: str, size: int) -> np.ndarray:
    while True:
        raw = input(f"{label} ({size} values, space-separated): ").strip()
        try:
            values = np.fromstring(raw, sep=" ", dtype=np.float64)
        except ValueError:
            values = np.array([], dtype=np.float64)
        if values.size == size:
            return values
        print(f"Expected {size} numbers, got {values.size}. Please try again.")


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
    base_to_camera = _make_transform(_quat_xyzw_to_rotmat(base_quaternion_xyzw), base_translation)
    return base_to_camera, data


def _init_camera(camera_fps: int, width: int, height: int, exposure: int):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, camera_fps)
    profile = pipeline.start(config)
    sensor = profile.get_device().query_sensors()[1]
    sensor.set_option(rs.option.enable_auto_exposure, 0)
    sensor.set_option(rs.option.exposure, exposure)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
    return pipeline, camera_params


def _make_detector():
    from pupil_apriltags import Detector

    return Detector(
        families="tag36h11",
        quad_decimate=1.0,
        nthreads=24,
        refine_edges=1,
        quad_sigma=0.2,
        decode_sharpening=1.0,
    )


def _capture_reference_tag_pose_camera(
    *,
    frames: int,
    timeout_s: float,
    camera_fps: int,
    width: int,
    height: int,
    exposure: int,
    decision_margin: float,
) -> np.ndarray:
    import cv2

    pipeline, camera_params = _init_camera(camera_fps, width, height, exposure)
    detector = _make_detector()
    poses = []

    try:
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline and len(poses) < int(frames):
            frameset = pipeline.wait_for_frames()
            color_frame = frameset.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=TAG_SIZE_M,
            )
            ref_tag = next(
                (
                    tag
                    for tag in detections
                    if tag.tag_id == REFERENCE_TAG_ID and tag.decision_margin > decision_margin
                ),
                None,
            )
            if ref_tag is None:
                continue
            pose = _make_transform(ref_tag.pose_R, ref_tag.pose_t.reshape(3))
            poses.append(pose)

        if not poses:
            raise RuntimeError(f"No usable reference-tag detections within {timeout_s:.1f}s")

        translations = np.stack([pose[:3, 3] for pose in poses], axis=0)
        rotations = np.stack([pose[:3, :3] for pose in poses], axis=0)
        median_translation = np.median(translations, axis=0)
        median_rotation = rotations[0]
        return _make_transform(median_rotation, median_translation)
    finally:
        pipeline.stop()


def solve_reference_tag_to_base(
    *,
    camera_pose_in_base: np.ndarray,
    reference_tag_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera_pose_in_base = np.asarray(camera_pose_in_base, dtype=np.float64).reshape(4, 4)
    reference_tag_to_camera = np.asarray(reference_tag_to_camera, dtype=np.float64).reshape(4, 4)
    reference_tag_to_base = camera_pose_in_base @ reference_tag_to_camera
    base_to_reference_tag = np.linalg.inv(reference_tag_to_base)
    return reference_tag_to_base, base_to_reference_tag


def _camera_pose_to_base_to_camera(
    *,
    pose_translation: np.ndarray,
    pose_quaternion_xyzw: np.ndarray,
    input_pose_is_camera_to_base: bool,
) -> np.ndarray:
    input_pose = _make_transform(_quat_xyzw_to_rotmat(pose_quaternion_xyzw), pose_translation)
    if input_pose_is_camera_to_base:
        return np.linalg.inv(input_pose)
    return input_pose


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a new REFERENCE_TAG_TO_BASE_4X4_DEFAULT from live reference-tag detections.",
    )
    parser.add_argument("--frames", type=int, default=5, help="Number of reference-tag detections to aggregate")
    parser.add_argument("--timeout", type=float, default=8.0, help="Maximum time to wait for detections")
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--exposure", type=int, default=100)
    parser.add_argument("--decision-margin", type=float, default=3.0)
    parser.add_argument(
        "--calib-file",
        type=Path,
        default=Path.home() / ".ros2/easy_handeye2/calibrations/fr3_eye_on_base_calib.calib",
        help="easy_handeye2 .calib file containing the eye-on-base hand-eye solution",
    )
    parser.add_argument(
        "--input-pose-is-camera-to-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Treat the entered xyz/xyzw as camera->base instead of base->camera. "
            "Default is base->camera, which matches the usual eye-on-base output."
        ),
    )
    parser.add_argument(
        "--prompt-handeye",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore --calib-file and prompt for camera pose manually",
    )
    args = parser.parse_args()

    if args.prompt_handeye:
        print("Enter the hand-eye pose manually.")
        camera_translation = _prompt_vector("Camera translation xyz", 3)
        camera_quaternion = _prompt_vector("Camera quaternion xyzw", 4)
        if args.input_pose_is_camera_to_base:
            print("Interpreting input as camera -> base, inverting to get base -> camera.")
            base_to_camera = _camera_pose_to_base_to_camera(
                pose_translation=camera_translation,
                pose_quaternion_xyzw=camera_quaternion,
                input_pose_is_camera_to_base=True,
            )
        else:
            print("Interpreting input as base -> camera, using it directly.")
            base_to_camera = _camera_pose_to_base_to_camera(
                pose_translation=camera_translation,
                pose_quaternion_xyzw=camera_quaternion,
                input_pose_is_camera_to_base=False,
            )
        calib_source = "manual prompt"
        calib_meta = {}
    else:
        base_to_camera, calib_meta = _load_handeye_calibration(args.calib_file)
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

    print("Capturing reference tag pose from live camera...")
    reference_tag_to_camera = _capture_reference_tag_pose_camera(
        frames=args.frames,
        timeout_s=args.timeout,
        camera_fps=args.camera_fps,
        width=args.camera_width,
        height=args.camera_height,
        exposure=args.exposure,
        decision_margin=args.decision_margin,
    )

    reference_tag_to_base, base_to_reference_tag = solve_reference_tag_to_base(
        camera_pose_in_base=base_to_camera,
        reference_tag_to_camera=reference_tag_to_camera,
    )

    print("\nCamera pose in base frame (base -> camera):")
    print(_format_matrix(base_to_camera))
    print("\nReference tag pose from detector (reference -> camera):")
    print(_format_matrix(reference_tag_to_camera))
    print("\nReference tag -> base:")
    print(_format_matrix(reference_tag_to_base))
    print("\nBase -> reference tag (inverse, debug only):")
    print(_format_matrix(base_to_reference_tag))
    print("\nPaste into real_robot_exps/static_constants.py as:")
    print("REFERENCE_TAG_TO_BASE_4X4_DEFAULT = np.array([")
    for row in reference_tag_to_base:
        print("    [" + ", ".join(f"{value:.16f}" for value in row) + "],")
    print("], dtype=np.float64)")

   


if __name__ == "__main__":
    main()
