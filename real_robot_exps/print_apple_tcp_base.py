"""Print live apple, spur, branch, and TCP poses in the Franka base frame.

This is a small diagnostic helper for checking whether the camera-derived
apple / spur / branch poses and the robot TCP pose agree in the same frame.

Default behavior:
- capture one fresh camera snapshot for the object poses
- refresh one robot state snapshot for the TCP pose
- print all poses in base frame, rounded for readability

Use --watch to repeat at a fixed interval.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from real_robot_exps.camera_snapshot import capture_structure_snapshot
from real_robot_exps.pro_robot_interface import FrankaInterface


def _format_pos_m(value) -> str:
    vec = np.asarray(value, dtype=np.float64).reshape(3)
    return np.array2string(vec, precision=3, suppress_small=True)


def _format_quat_xyzw(value) -> str:
    quat = np.asarray(value, dtype=np.float64).reshape(4)
    return np.array2string(quat, precision=4, suppress_small=True)


def _quat_xyzw_from_rotmat(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    return quat / max(np.linalg.norm(quat), 1e-12)


def _pose_components_from_4x4(pose_4x4) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose_4x4, dtype=np.float64).reshape(4, 4)
    return pose[:3, 3].copy(), _quat_xyzw_from_rotmat(pose[:3, :3])


def _load_config(path: Path, overrides: list[str]) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be 'key=value', got: {override}")
        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")
        parent = config
        for key in keys[:-1]:
            parent = parent[key]
        try:
            value = int(value_str)
        except ValueError:
            try:
                value = float(value_str)
            except ValueError:
                if value_str.lower() == "true":
                    value = True
                elif value_str.lower() == "false":
                    value = False
                else:
                    value = value_str
        parent[keys[-1]] = value
    return config


def _extract_tcp_pose_base(snap) -> tuple[np.ndarray, np.ndarray]:
    tcp_pos = np.asarray(snap.ee_pos.detach().cpu().numpy(), dtype=np.float64).reshape(3)
    tcp_quat_wxyz = np.asarray(snap.ee_quat.detach().cpu().numpy(), dtype=np.float64).reshape(4)
    tcp_quat_xyzw = np.array([tcp_quat_wxyz[1], tcp_quat_wxyz[2], tcp_quat_wxyz[3], tcp_quat_wxyz[0]], dtype=np.float64)
    return tcp_pos, tcp_quat_xyzw


def _read_parquet_metadata(path: Path) -> dict:
    payload = (pq.read_schema(path).metadata or {}).get(b"dataset_metadata")
    return json.loads(payload.decode("utf-8")) if payload else {}


def _vector_distance(a, b) -> tuple[np.ndarray, float]:
    a_vec = np.asarray(a, dtype=np.float64).reshape(3)
    b_vec = np.asarray(b, dtype=np.float64).reshape(3)
    delta = a_vec - b_vec
    return delta, float(np.linalg.norm(delta))


def _extract_grasp_distance(
    metadata: dict,
    *,
    phase: str,
    fallback_apple_pos=None,
) -> tuple[np.ndarray | None, float | None, dict | None]:
    phase_geometry = dict(metadata.get(f"{phase}_grasp_geometry", {}) or {})
    snapshot = dict(
        phase_geometry.get("snapshot", {})
        or phase_geometry.get("lengthened_snapshot", {})
        or phase_geometry.get("settled_snapshot", {})
        or {}
    )
    robot_snapshot = dict(phase_geometry.get("robot_snapshot", {}) or {})

    # The pre-grasp camera snapshot is intentionally taken before the arm
    # approaches. Do not pair it with the later pull-start TCP snapshot.
    tcp_pos = phase_geometry.get("tcp_pos", snapshot.get("tcp_pos"))
    if phase == "post" and tcp_pos is None:
        tcp_pos = robot_snapshot.get("tcp_pos")
    apple_pos = phase_geometry.get("apple_pos", snapshot.get("apple_pos"))
    if apple_pos is None:
        apple_pos = fallback_apple_pos
    if tcp_pos is None or apple_pos is None:
        return None, None, None

    delta, distance = _vector_distance(tcp_pos, apple_pos)
    return delta, distance, phase_geometry


def _print_parquet_distances(parquet_path: Path) -> None:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    metadata = _read_parquet_metadata(parquet_path)
    if not metadata:
        raise ValueError(f"No dataset metadata found in {parquet_path}")

    pre_delta, pre_distance, pre_geo = _extract_grasp_distance(metadata, phase="pre")
    fallback_apple = None
    if pre_geo is not None:
        pre_snapshot = dict(pre_geo.get("snapshot", {}) or pre_geo.get("lengthened_snapshot", {}) or {})
        pre_settled = dict(pre_geo.get("settled_snapshot", {}) or {})
        fallback_apple = pre_geo.get(
            "apple_pos",
            pre_snapshot.get("apple_pos", pre_settled.get("apple_pos")),
        )
    post_delta, post_distance, post_geo = _extract_grasp_distance(
        metadata,
        phase="post",
        fallback_apple_pos=fallback_apple,
    )

    print(f"parquet: {parquet_path}")
    if pre_delta is not None and pre_distance is not None:
        print(f"pre_grasp_tcp_minus_apple_base_m: {_format_pos_m(pre_delta)}")
        print(f"pre_grasp_tcp_apple_distance_m:   {pre_distance:.6f}")
    else:
        print("pre_grasp_tcp_apple_distance_m:   <unavailable>")

    if post_delta is not None and post_distance is not None:
        print(f"post_grasp_tcp_minus_apple_base_m: {_format_pos_m(post_delta)}")
        print(f"post_grasp_tcp_apple_distance_m:   {post_distance:.6f}")
    else:
        print("post_grasp_tcp_apple_distance_m:   <unavailable>")

    def _pose_name(phase_geo: dict | None, fallback: str = "unknown") -> str:
        if not phase_geo:
            return fallback
        robot_snapshot = dict(phase_geo.get("robot_snapshot", {}) or {})
        return str(
            phase_geo.get("pull_start_pose_name")
            or robot_snapshot.get("pull_start_pose_name")
            or phase_geo.get("robot_snapshot_pull_start_pose_name")
            or fallback
        )

    if pre_geo is not None:
        print(f"pre_grasp_pose_name: {_pose_name(pre_geo)}")
    if post_geo is not None:
        print(f"post_grasp_pose_name: {_pose_name(post_geo)}")


def _capture_pair(
    robot: FrankaInterface,
    *,
    camera_timeout_s: float,
    camera_frames: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, dict, float, float]:
    robot_capture_start = time.time()
    robot.refresh_state_snapshot()
    robot_snap = robot.get_state_snapshot()
    tcp_pos_base, tcp_quat_xyzw = _extract_tcp_pose_base(robot_snap)
    robot_capture_end = time.time()

    try:
        camera_snapshot = capture_structure_snapshot(
            min_complete_frames=camera_frames,
            timeout_s=camera_timeout_s,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "Device or resource busy" in message or "errno=16" in message:
            raise RuntimeError(
                "RealSense camera is already in use by another process. "
                "Stop any runner, detector, TF publisher, RViz camera consumer, "
                "or other RealSense process before running this diagnostic."
            ) from exc
        raise
    object_poses = {
        "apple": _pose_components_from_4x4(camera_snapshot["apple_pose_4x4"]),
        "spur": _pose_components_from_4x4(camera_snapshot["spur_pose_4x4"]),
        "branch": _pose_components_from_4x4(camera_snapshot["branch_pose_4x4"]),
    }
    return object_poses, tcp_pos_base, tcp_quat_xyzw, camera_snapshot, robot_capture_start, robot_capture_end


def _print_pose_block(name: str, pos_base: np.ndarray, quat_xyzw: np.ndarray) -> None:
    print(f"{name}_pos_base_m:   {_format_pos_m(pos_base)}")
    print(f"{name}_quat_xyzw:    {_format_quat_xyzw(quat_xyzw)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("real_robot_exps/config.yaml"))
    parser.add_argument("--camera-frames", type=int, default=1, help="Number of camera frames to median for the apple")
    parser.add_argument("--camera-timeout", type=float, default=8.0, help="Timeout for the camera snapshot")
    parser.add_argument("--parquet", type=Path, default=None, help="Read an existing parquet file and print pre/post grasp TCP-apple distances")
    parser.add_argument("--watch", action=argparse.BooleanOptionalAction, default=False, help="Keep printing repeatedly")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between prints in watch mode")
    parser.add_argument("--override", action="append", default=[], help="Override config values (repeatable)")
    args = parser.parse_args()

    if args.parquet is not None:
        _print_parquet_distances(args.parquet)
        return

    config = _load_config(args.config, args.override)
    robot = FrankaInterface(config, device="cpu")

    try:
        if not args.watch:
            object_poses, tcp_pos_base, tcp_quat_xyzw, camera_snapshot, robot_start, robot_end = _capture_pair(
                robot,
                camera_timeout_s=args.camera_timeout,
                camera_frames=args.camera_frames,
            )
            _print_pose_block("apple", *object_poses["apple"])
            _print_pose_block("spur", *object_poses["spur"])
            _print_pose_block("branch", *object_poses["branch"])
            _print_pose_block("tcp", tcp_pos_base, tcp_quat_xyzw)
            print(f"camera_snapshot_timestamp_s: {camera_snapshot['timestamp']:.3f}")
            print(f"robot_snapshot_window_s:     {robot_start:.3f} -> {robot_end:.3f}")
            print(f"tcp_minus_apple_base_m: {_format_pos_m(tcp_pos_base - object_poses['apple'][0])}")
            return

        print("Watching apple, spur, branch, and TCP poses in base frame. Ctrl-C to stop.")
        while True:
            start = time.time()
            object_poses, tcp_pos_base, tcp_quat_xyzw, camera_snapshot, robot_start, robot_end = _capture_pair(
                robot,
                camera_timeout_s=args.camera_timeout,
                camera_frames=args.camera_frames,
            )
            now = time.time()
            print(
                f"t={now:.3f}  "
                f"apple_pos_base_m={_format_pos_m(object_poses['apple'][0])}  "
                f"apple_quat_xyzw={_format_quat_xyzw(object_poses['apple'][1])}  "
                f"tcp_pos_base_m={_format_pos_m(tcp_pos_base)}  "
                f"tcp_quat_xyzw={_format_quat_xyzw(tcp_quat_xyzw)}"
            )
            print(
                f"       spur_pos_base_m={_format_pos_m(object_poses['spur'][0])}  "
                f"spur_quat_xyzw={_format_quat_xyzw(object_poses['spur'][1])}  "
                f"branch_pos_base_m={_format_pos_m(object_poses['branch'][0])}  "
                f"branch_quat_xyzw={_format_quat_xyzw(object_poses['branch'][1])}"
            )
            print(
                f"       camera_snapshot_timestamp_s={camera_snapshot['timestamp']:.3f}  "
                f"robot_snapshot_window_s={robot_start:.3f}->{robot_end:.3f}"
            )
            print(f"       tcp_minus_apple_base_m={_format_pos_m(tcp_pos_base - object_poses['apple'][0])}")
            remaining = args.interval - (time.time() - start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
