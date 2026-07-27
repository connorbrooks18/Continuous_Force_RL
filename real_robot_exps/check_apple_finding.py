"""Diagnostic helper for apple finding and frame transforms.

This script prints, for one or more live samples:
- the reference-tag-to-base transform used,
- the raw apple pose in the camera frame,
- the apple pose in the reference-tag frame,
- the apple pose in the Franka base frame,
- the robot TCP pose, if robot access is enabled.

It is intended for troubleshooting frame mismatches in the dynamic apple
pull-start logic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AT_TRACKING_ROOT = REPOSITORY_ROOT / "at-tracking"
if str(AT_TRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(AT_TRACKING_ROOT))

import Tracker  # noqa: E402

from real_robot_exps.apple_pullto_static import (  # noqa: E402
    APPLE_POS_M,
    APPLE_ROT_M,
    CLOSE_PULL_ROLL_FORWARD_DEG,
    HOME_POS_M,
    HOME_ROT_M,
    PRE_GRASP_APPLE_APPROACH_AXIS_BASE,
    USE_CLOSE_PULL_START_POSE,
    _load_dynamic_pull_start_pose,
    _translation_only_pose,
)
from real_robot_exps.pro_robot_interface import FrankaInterface, make_ee_target_pose_from_matrix  # noqa: E402
from real_robot_exps.static_constants import REFERENCE_TAG_TO_BASE_4X4_DEFAULT  # noqa: E402


TAG_SIZE_M = 0.0170
REFERENCE_TAG_ID = 1
APPLE_IDS = (7, 0)
APPLE_OFFSETS = (
    {"pos": [0.0, 0.0, 0.11], "rot": [[-0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, -0.7071]]},
    {"pos": [0.085, 0.0, 0.0], "rot": [[0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, 0.7071]]},
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def _pose_to_quat_xyzw(pose_4x4: np.ndarray) -> np.ndarray:
    return R.from_matrix(np.asarray(pose_4x4, dtype=np.float64)[:3, :3]).as_quat()


def _format_vec(vec: np.ndarray, precision: int = 5) -> str:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{v:+.{precision}f}" for v in vec) + "]"


def _format_mat(mat: np.ndarray, precision: int = 5) -> str:
    return np.array2string(
        np.asarray(mat, dtype=np.float64),
        precision=precision,
        suppress_small=True,
    )


def _init_camera(camera_fps: int, width: int, height: int, exposure: int):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, camera_fps)
    profile = pipeline.start(config)
    color_sensor = profile.get_device().query_sensors()[1]
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    color_sensor.set_option(rs.option.exposure, exposure)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
    return pipeline, camera_params


def _make_detector():
    return Detector(
        families="tag36h11",
        quad_decimate=1.0,
        nthreads=12,
        refine_edges=1,
        quad_sigma=0.2,
        decode_sharpening=1.0,
    )


def _detect_tags(detector, frame, camera_params, decision_margin: float):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=TAG_SIZE_M,
    )
    allowed = {REFERENCE_TAG_ID, *APPLE_IDS}
    return {
        tag.tag_id: tag
        for tag in tags
        if tag.decision_margin > decision_margin and tag.tag_id in allowed
    }


def _tag_pose_in_reference(tag, ref_tag):
    r_ref_inv = ref_tag.pose_R.T
    t_ref = ref_tag.pose_t
    pos = (r_ref_inv @ (tag.pose_t - t_ref)).reshape(3)
    rot = r_ref_inv @ tag.pose_R
    return pos, rot


def _tag_to_apple_transform(tag_id: int) -> np.ndarray:
    if tag_id == APPLE_IDS[0]:
        offset = APPLE_OFFSETS[0]
    elif tag_id == APPLE_IDS[1]:
        offset = APPLE_OFFSETS[1]
    else:
        raise KeyError(f"Unsupported apple tag id: {tag_id}")
    return _make_transform(offset["rot"], offset["pos"])


def _apple_pose_from_tag(tag) -> np.ndarray:
    tag_to_apple = _tag_to_apple_transform(int(tag.tag_id))
    return _make_transform(tag.pose_R, tag.pose_t) @ tag_to_apple


def _fuse_poses(poses: list[np.ndarray]) -> np.ndarray | None:
    if not poses:
        return None
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(poses[0], dtype=np.float64)[:3, :3]
    out[:3, 3] = np.median(
        np.stack([np.asarray(p, dtype=np.float64)[:3, 3] for p in poses], axis=0),
        axis=0,
    )
    return out


def _snapshot_tcp_pose(robot: FrankaInterface, stabilize_s: float = 1.0):
    robot.start_torque_mode()
    try:
        time.sleep(float(stabilize_s))
        snap = robot.get_state_snapshot()
    finally:
        robot.end_control()

    tcp_pose = np.eye(4, dtype=np.float64)
    tcp_pose[:3, :3] = R.from_quat(snap.ee_quat.detach().cpu().numpy()).as_matrix()
    tcp_pose[:3, 3] = snap.ee_pos.detach().cpu().numpy()
    return snap, tcp_pose


def _load_run_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _print_sample(sample: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(f"Sample timestamp: {sample['timestamp']:.6f}")
    print("Reference tag -> base transform used:")
    print(_format_mat(sample["reference_tag_to_base_4x4"]))
    print(f"Translation used: {_format_vec(sample['reference_tag_to_base_4x4'][:3, 3])}")
    print(f"Base-frame apple position: {_format_vec(sample['apple_pos_base'])}")
    print(f"Camera-frame apple position: {_format_vec(sample['apple_pos_camera'])}")
    print(f"Reference-frame apple position: {_format_vec(sample['apple_pos_reference'])}")
    print("Apple pose in camera frame:")
    print(_format_mat(sample["apple_pose_camera_4x4"]))
    print("Apple pose in reference frame:")
    print(_format_mat(sample["apple_pose_reference_4x4"]))
    print("Apple pose in base frame:")
    print(_format_mat(sample["apple_pose_base_4x4"]))
    print(f"Dynamic pull pose name: {sample['dynamic_pull_pose_name']}")
    print("Dynamic pull pose 4x4:")
    print(_format_mat(sample["dynamic_pull_pose_4x4"]))
    print("Pull start pose for reset 4x4:")
    print(_format_mat(sample["pull_start_pose_for_reset_4x4"]))
    if sample.get("tcp_pose_4x4") is not None:
        print(f"TCP position: {_format_vec(sample['tcp_pos'])}")
        print("TCP pose 4x4:")
        print(_format_mat(sample["tcp_pose_4x4"]))
        print(f"Base-frame apple minus TCP delta: {_format_vec(sample['apple_pos_base'] - sample['tcp_pos'])}")
    if sample.get("per_tag"):
        print("Per-tag apple poses in camera frame:")
        for item in sample["per_tag"]:
            print(
                f"  tag {item['tag_id']}: apple_pos_camera={_format_vec(item['apple_pos_camera'])} "
                f"apple_pos_reference={_format_vec(item['apple_pos_reference'])}"
            )


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}

    apple_pos_base = np.median(np.stack([s["apple_pos_base"] for s in samples], axis=0), axis=0)
    apple_pos_camera = np.median(np.stack([s["apple_pos_camera"] for s in samples], axis=0), axis=0)
    tcp_pos = np.median(np.stack([s["tcp_pos"] for s in samples if s.get("tcp_pos") is not None], axis=0), axis=0) if any(s.get("tcp_pos") is not None for s in samples) else None
    return {
        "sample_count": len(samples),
        "apple_pos_base": apple_pos_base,
        "apple_pos_camera": apple_pos_camera,
        "tcp_pos": tcp_pos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("real_robot_exps/config.yaml"))
    parser.add_argument("--run-metadata-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--samples", type=int, default=1, help="Number of valid samples to collect.")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--decision-margin", type=float, default=3.0)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--exposure", type=int, default=100)
    parser.add_argument("--robot", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    real_config = _load_config(args.config)
    run_metadata = _load_run_metadata(args.run_metadata_file)
    ref_to_base = np.asarray(
        real_config.get("tracking", {}).get(
            "reference_tag_to_base_4x4",
            REFERENCE_TAG_TO_BASE_4X4_DEFAULT,
        ),
        dtype=np.float64,
    ).reshape(4, 4)

    print("=" * 78)
    print("APPLE FINDING CHECK")
    print("=" * 78)
    print(f"Config: {args.config}")
    print("reference_tag_to_base_4x4:")
    print(_format_mat(ref_to_base))
    print(f"reference_tag_to_base translation: {_format_vec(ref_to_base[:3, 3])}")
    print(f"USE_CLOSE_PULL_START_POSE: {bool(USE_CLOSE_PULL_START_POSE)}")
    print(f"CLOSE_PULL_ROLL_FORWARD_DEG: {float(CLOSE_PULL_ROLL_FORWARD_DEG):.2f}")
    print(f"PRE_GRASP_APPLE_APPROACH_AXIS_BASE: {_format_vec(PRE_GRASP_APPLE_APPROACH_AXIS_BASE)}")
    print("APPLE fallback pose used by pullto:")
    print(_format_mat(make_ee_target_pose_from_matrix(APPLE_POS_M, APPLE_ROT_M)))
    print("HOME pose used by pullto:")
    print(_format_mat(make_ee_target_pose_from_matrix(HOME_POS_M, HOME_ROT_M)))

    robot = None
    if args.robot:
        print("\nConnecting to robot...")
        robot = FrankaInterface(real_config, device=args.device)
        _, tcp_pose = _snapshot_tcp_pose(robot)
    else:
        tcp_pose = None

    pipeline, camera_params = _init_camera(
        args.camera_fps, args.camera_width, args.camera_height, args.exposure
    )
    detector = _make_detector()
    apple_tracker = Tracker.Tracker("Apple", ids=APPLE_IDS, id_offsets=list(APPLE_OFFSETS))

    samples: list[dict[str, Any]] = []
    deadline = time.time() + float(args.timeout)

    try:
        print("\nCapturing camera samples...")
        while len(samples) < int(args.samples) and time.time() < deadline:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            tags = _detect_tags(detector, frame, camera_params, args.decision_margin)
            if REFERENCE_TAG_ID not in tags:
                continue

            ref_tag = tags[REFERENCE_TAG_ID]
            tags_in_ref = {
                tag_id: {"pos": pos, "rot": rot}
                for tag_id, tag in tags.items()
                if tag_id != REFERENCE_TAG_ID
                for pos, rot in [_tag_pose_in_reference(tag, ref_tag)]
            }
            apple_tracker.updatePose(tags_in_ref)

            per_tag: list[dict[str, Any]] = []
            apple_camera_poses: list[np.ndarray] = []
            apple_reference_poses: list[np.ndarray] = []
            for tag_id in APPLE_IDS:
                if tag_id not in tags:
                    continue
                tag = tags[tag_id]
                apple_pose_camera = _apple_pose_from_tag(tag)
                apple_pose_ref = _make_transform(*_tag_pose_in_reference(tag, ref_tag)[::-1]) @ _tag_to_apple_transform(tag_id)
                apple_camera_poses.append(apple_pose_camera)
                apple_reference_poses.append(apple_pose_ref)
                per_tag.append({
                    "tag_id": int(tag_id),
                    "apple_pos_camera": apple_pose_camera[:3, 3].copy(),
                    "apple_pos_reference": apple_pose_ref[:3, 3].copy(),
                })

            if not apple_camera_poses or not apple_reference_poses or apple_tracker.pose is None:
                continue

            apple_pose_camera = _fuse_poses(apple_camera_poses)
            apple_pose_reference = np.asarray(
                _make_transform(apple_tracker.pose["rot"], apple_tracker.pose["pos"]),
                dtype=np.float64,
            )
            apple_pose_base = ref_to_base @ apple_pose_reference
            dynamic_pull_pose_4x4, dynamic_pull_pose_name = _load_dynamic_pull_start_pose(
                run_metadata,
                make_ee_target_pose_from_matrix(APPLE_POS_M, APPLE_ROT_M),
            )
            if tcp_pose is not None and "settled_snapshot_apple_pose" in dynamic_pull_pose_name:
                current_snap = robot.get_state_snapshot()
                pull_start_pose_for_reset = _translation_only_pose(dynamic_pull_pose_4x4, current_snap.ee_quat)
            else:
                pull_start_pose_for_reset = np.asarray(dynamic_pull_pose_4x4, dtype=np.float64)

            sample = {
                "timestamp": float(time.time()),
                "reference_tag_to_base_4x4": ref_to_base.copy(),
                "apple_pose_camera_4x4": apple_pose_camera.copy(),
                "apple_pose_reference_4x4": apple_pose_reference.copy(),
                "apple_pose_base_4x4": apple_pose_base.copy(),
                "apple_pos_camera": apple_pose_camera[:3, 3].copy(),
                "apple_pos_reference": apple_pose_reference[:3, 3].copy(),
                "apple_pos_base": apple_pose_base[:3, 3].copy(),
                "dynamic_pull_pose_4x4": dynamic_pull_pose_4x4.copy(),
                "dynamic_pull_pose_name": dynamic_pull_pose_name,
                "pull_start_pose_for_reset_4x4": pull_start_pose_for_reset.copy(),
                "per_tag": per_tag,
                "tcp_pose_4x4": tcp_pose.copy() if tcp_pose is not None else None,
                "tcp_pos": tcp_pose[:3, 3].copy() if tcp_pose is not None else None,
            }
            samples.append(sample)
            _print_sample(sample)

        if not samples:
            raise RuntimeError(
                f"No valid samples captured in {args.timeout:.1f}s. "
                "Make sure the reference tag and apple tags are visible."
            )

        summary = _summary(samples)
        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        print(f"Samples: {summary['sample_count']}")
        print(f"Median apple position (camera frame): {_format_vec(summary['apple_pos_camera'])}")
        print(f"Median apple position (base frame): {_format_vec(summary['apple_pos_base'])}")
        if summary["tcp_pos"] is not None:
            print(f"Median TCP position: {_format_vec(summary['tcp_pos'])}")
            print(f"Median apple minus TCP delta: {_format_vec(summary['apple_pos_base'] - summary['tcp_pos'])}")

        if args.output is not None:
            payload = {
                "config_path": str(args.config),
                "reference_tag_to_base_4x4": ref_to_base.tolist(),
                "samples": [
                    {
                        **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sample.items() if k != "per_tag"},
                        "per_tag": [
                            {
                                "tag_id": item["tag_id"],
                                "apple_pos_camera": item["apple_pos_camera"].tolist(),
                                "apple_pos_reference": item["apple_pos_reference"].tolist(),
                            }
                            for item in sample["per_tag"]
                        ],
                    }
                    for sample in samples
                ],
                "summary": {
                    "sample_count": summary["sample_count"],
                    "apple_pos_camera": summary["apple_pos_camera"].tolist(),
                    "apple_pos_base": summary["apple_pos_base"].tolist(),
                    "tcp_pos": None if summary["tcp_pos"] is None else summary["tcp_pos"].tolist(),
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(f"\nWrote JSON report to {args.output}")

    finally:
        pipeline.stop()
        if robot is not None:
            robot.shutdown()


if __name__ == "__main__":
    main()
