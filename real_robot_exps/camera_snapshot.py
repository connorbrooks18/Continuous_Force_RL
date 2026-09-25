"""Standalone structure snapshot for the lab tools (runner.py, print_apple_tcp_base.py).

The field session does NOT use this: it keeps one detector running and takes
every snapshot through its request directory (see snapshot_geometry.request_snapshot),
so only one process ever opens the RealSense. This module opens the camera itself,
so it must not run while a detector is running.

Tags, sizes and offsets come from at-tracking/tracking_config.yaml, the same
source the detector uses.
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

from snapshot_requests import TRACKED_NAMES, build_snapshot  # noqa: E402
from tracking_config import DEFAULT_CONFIG_PATH, load_tracking_config  # noqa: E402

from real_robot_exps.frame_transforms import transform_pose_to_base
from real_robot_exps.snapshot_geometry import (  # noqa: F401  (re-exported for older callers)
    rpy_deg_from_vector as _rpy_deg_from_vector,
    update_pre_grasp_geometry_with_snapshots,
)
from real_robot_exps.static_constants import CAMERA_TO_BASE_4X4_DEFAULT


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


def _detect_valid_tags(detector, frame, camera_params, config) -> dict[int, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=config.default_tag_size_m,
    )
    allowed = set(config.allowed_ids())
    sizes = config.tag_sizes()
    valid = {}
    for tag in tags:
        if tag.decision_margin <= config.decision_margin or tag.tag_id not in allowed:
            continue
        size = sizes.get(tag.tag_id, config.default_tag_size_m)
        if size != config.default_tag_size_m:
            tag.pose_t = tag.pose_t * (size / config.default_tag_size_m)
        valid[tag.tag_id] = tag
    return valid


def capture_structure_snapshot(
    *,
    camera_to_base_4x4: np.ndarray | None = None,
    tracking_config_path: Path | str = DEFAULT_CONFIG_PATH,
    min_complete_frames: int = 5,
    timeout_s: float = 8.0,
    camera_fps: int = 15,
    width: int = 1280,
    height: int = 720,
    exposure: int = 100,
) -> dict[str, Any]:
    camera_to_base = np.asarray(
        CAMERA_TO_BASE_4X4_DEFAULT if camera_to_base_4x4 is None else camera_to_base_4x4,
        dtype=np.float64,
    ).reshape(4, 4)
    config = load_tracking_config(tracking_config_path)
    trackers = config.build_trackers()
    detector = Detector(
        families=config.tag_family,
        quad_decimate=1.0,
        nthreads=12,
        refine_edges=1,
        quad_sigma=0.2,
        decode_sharpening=1.0,
    )
    pipeline, camera_params = _init_camera(camera_fps, width, height, exposure)
    sample_poses_base = {name: [] for name in TRACKED_NAMES}
    timestamps: list[float] = []

    try:
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline and len(timestamps) < int(min_complete_frames):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            tag_dict = _detect_valid_tags(detector, frame, camera_params, config)
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
            now = time.time()
            for tracker in trackers:
                sample_poses_base[tracker.name].append(
                    transform_pose_to_base(tracker.pose, camera_to_base=camera_to_base)
                )
            timestamps.append(now)
        if len(timestamps) < int(min_complete_frames):
            raise RuntimeError(
                f"Only captured {len(timestamps)} complete frames in {timeout_s:.1f}s; "
                f"need at least {min_complete_frames}"
            )
        snapshot = build_snapshot(sample_poses_base, timestamps, camera_to_base)
        snapshot["source"] = "camera_snapshot_standalone"
        snapshot["tracking_config"] = config.to_metadata()
        return snapshot
    finally:
        pipeline.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--camera-to-base", type=Path, default=None,
                        help="JSON camera(optical)->base 4x4; defaults to static_constants")
    parser.add_argument("--tracking-config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    camera_to_base = None
    if args.camera_to_base is not None:
        from real_robot_exps.frame_transforms import load_camera_to_base
        camera_to_base = load_camera_to_base(args.camera_to_base)
    snapshot = capture_structure_snapshot(
        camera_to_base_4x4=camera_to_base,
        tracking_config_path=args.tracking_config,
        min_complete_frames=args.frames,
        timeout_s=args.timeout,
    )
    args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote snapshot to {args.output}")


if __name__ == "__main__":
    main()
