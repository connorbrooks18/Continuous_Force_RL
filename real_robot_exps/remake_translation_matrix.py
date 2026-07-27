"""Solve the reference-tag-to-base translation from live detections.

Math:
    apple_base = R_ref_to_base @ apple_ref + t_ref_to_base
    tcp_base   = apple_base + [0, -d, 0]

So the translation is:
    t_ref_to_base = tcp_base - [0, -d, 0] - R_ref_to_base @ apple_ref

where:
- `apple_ref` comes from live camera detections in reference-tag coordinates
- `tcp_base` comes from the live robot state in base coordinates
- `d` is the apple-to-TCP distance in base-frame -Y

The script prints the solved translation and the full 4x4 matrix.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AT_TRACKING_ROOT = REPOSITORY_ROOT / "at-tracking"
if str(AT_TRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(AT_TRACKING_ROOT))

from real_robot_exps.static_constants import REFERENCE_TAG_TO_BASE_4X4_DEFAULT


REFERENCE_TAG_ID = 1
TAG_SIZE_M = 0.0170
APPLE_IDS = (7, 0)
APPLE_OFFSETS = (
    {"pos": [0.0, 0.0, 0.11], "rot": [[-0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, -0.7071]]},
    {"pos": [0.085, 0.0, 0.0], "rot": [[0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, 0.7071]]},
)
DEFAULT_TCP_APPLE_OFFSET_M = 0.04


def _format_vector(vec: np.ndarray) -> str:
    return np.array2string(np.asarray(vec, dtype=np.float64), precision=5, suppress_small=True)


def _format_matrix(mat: np.ndarray) -> str:
    return np.array2string(np.asarray(mat, dtype=np.float64), precision=5, suppress_small=True)


def _make_transform(rot, pos):
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = np.asarray(rot, dtype=np.float64)
    tf[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return tf


def _capture_live_apple_center_ref(
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
    import pyrealsense2 as rs
    from pupil_apriltags import Detector

    if str(AT_TRACKING_ROOT) not in sys.path:
        sys.path.insert(0, str(AT_TRACKING_ROOT))
    import Tracker  # noqa: E402

    def _init_camera(camera_fps_: int, width_: int, height_: int, exposure_: int):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width_, height_, rs.format.bgr8, camera_fps_)
        profile = pipeline.start(config)
        sensor = profile.get_device().query_sensors()[1]
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, exposure_)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        camera_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
        return pipeline, camera_params

    def _make_detector():
        return Detector(
            families="tag36h11",
            quad_decimate=1.0,
            nthreads=24,
            refine_edges=1,
            quad_sigma=0.2,
            decode_sharpening=1.0,
        )

    def _detect_tags(detector, frame, camera_params, decision_margin_):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_tags = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=TAG_SIZE_M,
        )
        allowed = {REFERENCE_TAG_ID, *APPLE_IDS}
        return {
            tag.tag_id: tag
            for tag in raw_tags
            if tag.decision_margin > decision_margin_ and tag.tag_id in allowed
        }

    def _tag_pose_in_reference(tag, ref_tag):
        r_ref_inv = ref_tag.pose_R.T
        t_ref = ref_tag.pose_t
        pos = (r_ref_inv @ (tag.pose_t - t_ref)).reshape(3)
        rot = r_ref_inv @ tag.pose_R
        return pos, rot

    detector = _make_detector()
    pipeline, camera_params = _init_camera(camera_fps, width, height, exposure)
    apple = Tracker.Tracker("Apple", APPLE_IDS, APPLE_OFFSETS)
    samples: list[np.ndarray] = []

    try:
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline and len(samples) < int(frames):
            frameset = pipeline.wait_for_frames()
            color_frame = frameset.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            tags = _detect_tags(detector, frame, camera_params, decision_margin)
            if REFERENCE_TAG_ID not in tags:
                continue

            ref_tag = tags[REFERENCE_TAG_ID]
            tags_in_ref = {
                tag_id: {"pos": pos, "rot": rot}
                for tag_id, tag in tags.items()
                if tag_id != REFERENCE_TAG_ID
                for pos, rot in [_tag_pose_in_reference(tag, ref_tag)]
            }
            apple.updatePose(tags_in_ref)
            if apple.pose is None:
                continue
            samples.append(np.asarray(apple.pose["pos"], dtype=np.float64).reshape(3))

        if not samples:
            raise RuntimeError(f"No usable apple detections within {timeout_s:.1f}s")
        return np.median(np.stack(samples, axis=0), axis=0)
    finally:
        pipeline.stop()


def _extract_tcp_base_position(robot_snapshot) -> np.ndarray:
    tcp_pos = getattr(robot_snapshot, "O_T_EE", None)
    if tcp_pos is None:
        raise ValueError("robot snapshot is missing O_T_EE")
    tcp_pos = np.asarray(tcp_pos, dtype=np.float64).reshape(-1)
    if tcp_pos.size != 16:
        raise ValueError(f"O_T_EE must contain 16 values, got {tcp_pos.size}")
    return np.array([tcp_pos[12], tcp_pos[13], tcp_pos[14]], dtype=np.float64)


def solve_reference_tag_to_base_translation(
    *,
    apple_center_ref: np.ndarray,
    tcp_pos_base: np.ndarray,
    base_rotation: np.ndarray,
    apple_to_tcp_distance_m: float = DEFAULT_TCP_APPLE_OFFSET_M,
) -> np.ndarray:
    apple_center_ref = np.asarray(apple_center_ref, dtype=np.float64).reshape(3)
    tcp_pos_base = np.asarray(tcp_pos_base, dtype=np.float64).reshape(3)
    base_rotation = np.asarray(base_rotation, dtype=np.float64).reshape(3, 3)
    offset_base = np.array([0.0, -float(apple_to_tcp_distance_m), 0.0], dtype=np.float64)
    return tcp_pos_base - offset_base - base_rotation @ apple_center_ref


def main() -> None:
    import yaml
    import pylibfranka

    parser = argparse.ArgumentParser(description="Solve the live reference-tag-to-base translation")
    parser.add_argument(
        "--config",
        type=str,
        default="real_robot_exps/config.yaml",
        help="Path to the robot config YAML",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of complete camera frames to aggregate before solving",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="Maximum time to wait for enough camera frames",
    )
    parser.add_argument(
        "--apple-to-tcp-distance-m",
        type=float,
        default=DEFAULT_TCP_APPLE_OFFSET_M,
        help="Distance from apple center to TCP in base -Y, in meters",
    )
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--exposure", type=int, default=100)
    parser.add_argument("--decision-margin", type=float, default=3.0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    robot_cfg = config["robot"]
    robot = pylibfranka.Robot(robot_cfg["ip"])
    try:
        robot.set_EE(robot_cfg.get("NE_T_EE", [
            0.7071, -0.7071, 0.0, 0.0,
            0.7071, 0.7071, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.1034, 1.0,
        ]))
        robot.set_K(robot_cfg.get("EE_T_K", [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]))
        robot_snapshot = robot.read_once()

        apple_center_ref = _capture_live_apple_center_ref(
            frames=args.frames,
            timeout_s=args.timeout,
            camera_fps=args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
            exposure=args.exposure,
            decision_margin=args.decision_margin,
        )
        tcp_pos_base = _extract_tcp_base_position(robot_snapshot)

        base_rotation = np.asarray(REFERENCE_TAG_TO_BASE_4X4_DEFAULT, dtype=np.float64)[:3, :3]
        translation = solve_reference_tag_to_base_translation(
            apple_center_ref=apple_center_ref,
            tcp_pos_base=tcp_pos_base,
            base_rotation=base_rotation,
            apple_to_tcp_distance_m=args.apple_to_tcp_distance_m,
        )

        ref_to_base = np.asarray(REFERENCE_TAG_TO_BASE_4X4_DEFAULT, dtype=np.float64).copy()
        ref_to_base[:3, 3] = translation

        print("apple_center_ref_m =", _format_vector(apple_center_ref))
        print("tcp_pos_base_m =", _format_vector(tcp_pos_base))
        print("apple_to_tcp_distance_m =", f"{args.apple_to_tcp_distance_m:.5f}")
        print("reference_tag_to_base_translation_m =", _format_vector(translation))
        print("reference_tag_to_base_4x4:")
        print(_format_matrix(ref_to_base))
    finally:
        robot.stop()


if __name__ == "__main__":
    main()
