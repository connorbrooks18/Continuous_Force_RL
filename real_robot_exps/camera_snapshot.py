"""Capture a short AprilTag-based structure snapshot in the Franka base frame.

This is a lightweight, headless utility used by runner.py before a pull run.
It samples a few complete camera frames, aggregates the tracked Branch/Spur/Apple
poses, transforms them into the Franka base frame, and returns a single snapshot
dictionary that can be embedded into run metadata.
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
from pupil_apriltags import Detector

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AT_TRACKING_ROOT = REPOSITORY_ROOT / "at-tracking"
if str(AT_TRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(AT_TRACKING_ROOT))

import Tracker  # noqa: E402

from real_robot_exps.frame_transforms import median_pose_4x4, transform_pose_to_base
from real_robot_exps.static_constants import CAMERA_TO_BASE_4X4_DEFAULT


TAG_SIZE_M = 0.0170
REFERENCE_TAG_ID = 1
TRACKED_NAMES = ("Branch", "Spur", "Apple")


def _quat_xyzw_from_rotmat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    return quat / max(np.linalg.norm(quat), 1e-12)


def _tracker_set() -> list[Any]:
    apple_offsets = [
        {"pos": [0, 0.0, 0.11], "rot": [[-0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, -0.7071]]},
        {"pos": [0.085, 0.00, 0.0], "rot": [[0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, 0.7071]]},
    ]
    spur_offsets = [
        {"pos": [0.0, 0.01, 0.03], "rot": np.eye(3)},
        {"pos": [0.0, 0.01, 0.03], "rot": [[0, 0, -1], [0, 1, 0], [1, 0, 0]]},
        {"pos": [0.0, 0.01, 0.03], "rot": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]},
    ]
    branch_offsets = [{"pos": [0, -0.03, 0.03], "rot": np.eye(3)}]
    return [
        Tracker.Tracker("Branch", ids=(2,), id_offsets=branch_offsets),
        Tracker.Tracker("Spur", ids=(3, 4, 5), id_offsets=spur_offsets),
        Tracker.Tracker("Apple", ids=(7, 0), id_offsets=apple_offsets),
    ]


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


def _detect_valid_tags(detector, frame, camera_params, decision_margin: float):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=TAG_SIZE_M,
    )
    allowed = {0, 1, 2, 3, 4, 5, 7}
    return {
        tag.tag_id: tag
        for tag in tags
        if tag.decision_margin > decision_margin and tag.tag_id in allowed
    }


def _rpy_deg_from_vector(vec: np.ndarray) -> list[float]:
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return [0.0, 0.0, 0.0]
    v = vec / norm
    yaw = float(np.degrees(np.arctan2(v[1], v[0])))
    pitch = float(np.degrees(np.arctan2(-v[2], np.hypot(v[0], v[1]))))
    return [0.0, pitch, yaw]


def _build_snapshot(
    sample_poses_base: dict[str, list[np.ndarray]],
    timestamps: list[float],
    camera_to_base: np.ndarray,
    reference_tag_to_base: np.ndarray | None,
) -> dict[str, Any]:
    median_poses = {}
    median_positions = {}
    for name in TRACKED_NAMES:
        pose_stack = np.stack(sample_poses_base[name], axis=0)
        pose = np.median(pose_stack, axis=0)
        pose[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        median_poses[name] = pose
        median_positions[name] = pose[:3, 3].copy()

    branch = median_positions["Branch"]
    spur = median_positions["Spur"]
    apple = median_positions["Apple"]
    starts = np.stack([branch, branch, spur], axis=0)
    ends = np.stack([spur, apple, apple], axis=0)

    return {
        "timestamp": float(np.median(np.asarray(timestamps, dtype=np.float64))),
        "camera_selected_timestamps": [float(t) for t in timestamps],
        "camera_frame_count": int(len(timestamps)),
        "camera_to_base_4x4": np.asarray(camera_to_base, dtype=np.float64).tolist(),
        "reference_tag_to_base_4x4_used": (
            np.asarray(reference_tag_to_base, dtype=np.float64).tolist()
            if reference_tag_to_base is not None
            else None
        ),
        "apple_pos": apple.tolist(),
        "apple_pose_4x4": median_poses["Apple"].reshape(-1).tolist(),
        "apple_quat_xyzw": _quat_xyzw_from_rotmat(median_poses["Apple"][:3, :3]).tolist(),
        "branch_pos": branch.tolist(),
        "branch_pose_4x4": median_poses["Branch"].reshape(-1).tolist(),
        "spur_pos": spur.tolist(),
        "spur_pose_4x4": median_poses["Spur"].reshape(-1).tolist(),
        "woody_part_start_pos": starts.reshape(-1).tolist(),
        "woody_part_end_pos": ends.reshape(-1).tolist(),
        "woody_bending_angles": [0.0, 0.0, 0.0],
    }


def capture_structure_snapshot(
    *,
    camera_to_base_4x4: np.ndarray | None = None,
    min_complete_frames: int = 5,
    timeout_s: float = 8.0,
    camera_fps: int = 15,
    width: int = 1280,
    height: int = 720,
    exposure: int = 100,
    decision_margin: float = 3.0,
) -> dict[str, Any]:
    camera_to_base = np.asarray(
        CAMERA_TO_BASE_4X4_DEFAULT if camera_to_base_4x4 is None else camera_to_base_4x4,
        dtype=np.float64,
    ).reshape(4, 4)
    trackers = _tracker_set()
    detector = _make_detector()
    pipeline, camera_params = _init_camera(camera_fps, width, height, exposure)
    sample_poses_base = {name: [] for name in TRACKED_NAMES}
    reference_tag_base_samples: list[np.ndarray] = []
    timestamps: list[float] = []

    try:
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline and len(timestamps) < int(min_complete_frames):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            tag_dict = _detect_valid_tags(detector, frame, camera_params, decision_margin)
            tags_in_camera = {
                tag_id: {
                    "pos": np.asarray(tag.pose_t, dtype=np.float64).reshape(3),
                    "rot": np.asarray(tag.pose_R, dtype=np.float64),
                }
                for tag_id, tag in tag_dict.items()
            }
            for tracker in trackers:
                tracker.updatePose(tags_in_camera)
            if any(tracker.pose is None for tracker in trackers):
                continue
            if REFERENCE_TAG_ID in tag_dict:
                ref_tag = tag_dict[REFERENCE_TAG_ID]
                reference_tag_base_samples.append(
                    transform_pose_to_base(
                        {"pos": np.asarray(ref_tag.pose_t, dtype=np.float64).reshape(3), "rot": np.asarray(ref_tag.pose_R, dtype=np.float64)},
                        camera_to_base=camera_to_base,
                    )
                )
            now = time.time()
            for tracker in trackers:
                pose_base = transform_pose_to_base(tracker.pose, camera_to_base=camera_to_base)
                sample_poses_base[tracker.name].append(pose_base)
            timestamps.append(now)
        if len(timestamps) < int(min_complete_frames):
            raise RuntimeError(
                f"Only captured {len(timestamps)} complete frames in {timeout_s:.1f}s; "
                f"need at least {min_complete_frames}"
            )
        return _build_snapshot(
            sample_poses_base,
            timestamps,
            camera_to_base,
            median_pose_4x4(reference_tag_base_samples),
        )
    finally:
        pipeline.stop()


def update_pre_grasp_geometry_with_snapshots(
    pre_grasp_geometry: dict[str, Any],
    *,
    settled_snapshot: dict[str, Any],
    lengthened_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = json.loads(json.dumps(pre_grasp_geometry))
    out["settled_snapshot"] = settled_snapshot
    out["lengthened_snapshot"] = lengthened_snapshot or {}
    out["snapshot"] = settled_snapshot

    parts = out.setdefault("parts", {})
    if "primary" in parts:
        parts["primary"]["connection_rpy_deg"] = [0.0, 0.0, 0.0]
    if lengthened_snapshot and all(key in lengthened_snapshot for key in ("branch_pos", "spur_pos", "apple_pos")):
        branch = np.asarray(lengthened_snapshot["branch_pos"], dtype=np.float64)
        spur = np.asarray(lengthened_snapshot["spur_pos"], dtype=np.float64)
        apple = np.asarray(lengthened_snapshot["apple_pos"], dtype=np.float64)
        spur_vec = spur - branch
        stem_vec = apple - spur
        if "spur" in parts:
            parts["spur"]["connection_rpy_deg"] = _rpy_deg_from_vector(spur_vec)
            parts["spur"]["connection_source"] = "lengthened_snapshot"
        if "stem" in parts:
            parts["stem"]["connection_rpy_deg"] = _rpy_deg_from_vector(stem_vec)
            parts["stem"]["connection_source"] = "lengthened_snapshot"
    if "apple" in parts:
        parts["apple"]["connection_rpy_deg"] = [0.0, 0.0, 0.0]
        parts["apple"]["connection_source"] = "settled_snapshot"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-second", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()
    if args.output_second is not None:
        settled, lengthened = capture_structure_snapshot_pair(
            min_complete_frames=args.frames,
            timeout_s=args.timeout,
        )
        args.output.write_text(json.dumps(settled, indent=2, sort_keys=True), encoding="utf-8")
        args.output_second.write_text(json.dumps(lengthened, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote snapshot to {args.output}")
        print(f"Wrote snapshot to {args.output_second}")
    else:
        snapshot = capture_structure_snapshot(
            min_complete_frames=args.frames,
            timeout_s=args.timeout,
        )
        args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote snapshot to {args.output}")


if __name__ == "__main__":
    main()
