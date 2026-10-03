"""Compile separately recorded robot and AprilTag data into one episode Parquet.

Both inputs use Unix wall-clock seconds from ``time.time()``. Robot measurements
remain at the robot policy rate. The tracking Parquet is expected to already be
expressed in the Franka base frame, so the compiler only aligns timestamps and
aggregates geometry; it does not apply any camera-to-base calibration.

Usage:
    python -m real_robot_exps.compile_static_sysid \
        --robot pull_theta2.36_phi1.57_raw_robot.parquet \
        --tracking output.parquet \
        --output pull_theta2.36_phi1.57_unified.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from real_robot_exps.snapshot_geometry import update_pre_grasp_geometry_with_snapshots
from real_robot_exps.static_constants import CAMERA_TO_BASE_4X4_DEFAULT

SCHEMA_NAME = "real_static_sysid_episode"
SCHEMA_VERSION = "1.1.0"
# Trackers of tracking files written before the detector recorded its selection.
TRACKED_NAMES = ("Branch", "Spur", "Apple")


def tracker_key(name: str) -> str:
    """Tracker name -> column/snapshot key prefix: 'SpurStart' -> 'spur_start'.

    Same rule as at-tracking's ``tracking_config.snake_key``.
    """
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name)).replace(" ", "_").lower()


def tracked_names_from_metadata(metadata: dict[str, Any]) -> tuple[str, ...]:
    """The trackers a tracking file holds (the detector's ``--tags`` selection)."""
    names = tuple(str(name) for name in (metadata.get("tracker_names") or TRACKED_NAMES))
    if "Apple" not in names:
        raise ValueError(f"Tracking has no Apple tracker (trackers: {list(names)})")
    return names


def radius_shift_from_metadata(metadata: dict[str, Any]) -> dict[str, bool]:
    """Per-tracker ``radius_shift`` from the tracking config; absent means True."""
    flags = (metadata.get("tracking_config") or {}).get("radius_shift") or {}
    return {str(name): bool(flag) for name, flag in flags.items()}


def _read_dataset_metadata(path: Path) -> dict[str, Any]:
    raw = pq.read_schema(path).metadata or {}
    payload = raw.get(b"dataset_metadata")
    if payload is None:
        return {}
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid dataset_metadata JSON in {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "modified_timestamp": float(stat.st_mtime),
        "sha256": _sha256(path),
    }


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_tracking_frames(path: Path, names=TRACKED_NAMES) -> pd.DataFrame:
    tracking = pq.read_table(path).to_pandas()
    expected = ["timestamp", "name", "x", "y", "z", "qx", "qy", "qz", "qw"]
    if list(tracking.columns) != expected:
        if len(tracking.columns) == len(expected):
            # Backward compatibility with the old integer-column DataCollector.
            tracking.columns = expected
        else:
            raise ValueError(f"Unexpected tracking columns: {list(tracking.columns)}")

    tracking = tracking[tracking["name"].isin(names)].copy()
    xyz = tracking[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(xyz.to_numpy()).all(axis=1)
    # The tracker uses exactly [0,0,0] as its missing-detection sentinel.
    valid &= ~np.isclose(xyz.to_numpy(), 0.0).all(axis=1)
    tracking = tracking.loc[valid].copy()
    tracking[["x", "y", "z"]] = xyz.loc[valid]
    tracking["timestamp"] = pd.to_numeric(tracking["timestamp"], errors="coerce")
    tracking = tracking[np.isfinite(tracking["timestamp"])].copy()

    if tracking.empty:
        raise ValueError(f"Tracking input contains no valid {'/'.join(names)} frames")
    return tracking.sort_values("timestamp").reset_index(drop=True)


class _TrackingFrameIndex:
    """Pre-grouped complete tracking frames for fast timestamp lookup."""

    def __init__(self, frames: pd.DataFrame, names=TRACKED_NAMES) -> None:
        self.names = tuple(names)
        groups = frames.groupby("timestamp", sort=True)
        timestamps: list[float] = []
        grouped_frames: list[pd.DataFrame] = []
        for timestamp, group in groups:
            if set(str(name) for name in group["name"].tolist()) >= set(self.names):
                timestamps.append(float(timestamp))
                grouped_frames.append(group.sort_values("name"))
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        self.frames = grouped_frames
        if not len(self.timestamps):
            raise ValueError("Tracking input contains no complete valid marker frames")


def _build_tracking_index(frames: pd.DataFrame, names=TRACKED_NAMES) -> _TrackingFrameIndex:
    return _TrackingFrameIndex(frames, names)


def _require_tracking_frame_base(metadata: dict[str, Any], path: Path) -> None:
    frame = str(metadata.get("coordinate_frame", "")).strip()
    if frame not in {"franka_base_o", "franka_base_o_frame"}:
        raise ValueError(
            f"Tracking input {path} must already be expressed in Franka base frame; "
            f"got coordinate_frame={frame!r}"
        )


def _load_camera_to_base(metadata: dict[str, Any], path: Path) -> np.ndarray:
    camera_to_base = metadata.get("camera_to_base_4x4_used", CAMERA_TO_BASE_4X4_DEFAULT)
    camera_to_base = np.asarray(camera_to_base, dtype=np.float64)
    if camera_to_base.shape != (4, 4) or not np.isfinite(camera_to_base).all():
        raise ValueError(
            f"Tracking input {path} has invalid camera_to_base_4x4_used metadata"
        )
    if abs(float(np.linalg.det(camera_to_base[:3, :3]))) < 1e-10:
        raise ValueError(
            f"Tracking input {path} has a non-invertible camera_to_base_4x4_used rotation"
        )
    return camera_to_base


def _select_frames(
    frames: _TrackingFrameIndex | pd.DataFrame,
    *,
    center: float,
    count: int,
    max_delta_s: float,
    interval: tuple[float, float] | None = None,
    prefer_before: bool = False,
) -> pd.DataFrame:
    # Keep accepting a DataFrame for compatibility with direct callers/tests,
    # but the compiler passes the precomputed index below.
    index = frames if isinstance(frames, _TrackingFrameIndex) else _build_tracking_index(frames)
    candidate_indices = np.arange(len(index.timestamps), dtype=np.int64)
    if interval is not None:
        start, end = interval
        candidate_indices = candidate_indices[
            (index.timestamps >= float(start)) & (index.timestamps <= float(end))
        ]
    if prefer_before:
        before = index.timestamps[candidate_indices] <= float(center)
        if np.any(before):
            candidate_indices = candidate_indices[before]
    if not len(candidate_indices):
        raise ValueError(
            f"No complete camera frames within {max_delta_s:.3f}s of timestamp {center:.6f}"
        )

    deltas = np.abs(index.timestamps[candidate_indices] - float(center))
    valid = deltas <= float(max_delta_s)
    candidate_indices = candidate_indices[valid]
    deltas = deltas[valid]
    if not len(candidate_indices):
        raise ValueError(
            f"No complete camera frames within {max_delta_s:.3f}s of timestamp {center:.6f}"
        )
    order = np.argsort(deltas, kind="stable")[: int(count)]
    selected_indices = np.sort(candidate_indices[order])
    selected = pd.concat(
        [index.frames[int(idx)] for idx in selected_indices],
        ignore_index=True,
    )
    if selected.empty:
        raise ValueError(
            f"No complete camera frames within {max_delta_s:.3f}s of timestamp {center:.6f}"
        )
    return selected


def _complete_timestamps_in_interval(
    frames: _TrackingFrameIndex | pd.DataFrame,
    interval: tuple[float, float],
) -> np.ndarray:
    index = frames if isinstance(frames, _TrackingFrameIndex) else _build_tracking_index(frames)
    start, end = interval
    mask = (index.timestamps >= float(start)) & (index.timestamps <= float(end))
    return index.timestamps[mask].copy()


def _nearest_timestamp(target: float, candidates: np.ndarray) -> float:
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1)
    if candidates.size == 0:
        return float("nan")
    idx = int(np.argmin(np.abs(candidates - float(target))))
    return float(candidates[idx])


def _median_positions(frames: pd.DataFrame, names=TRACKED_NAMES) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for name in names:
        subset = frames[frames["name"] == name][["x", "y", "z"]]
        if subset.empty:
            raise ValueError(f"Tracking selection is missing {name}")
        positions[name] = np.median(subset.to_numpy(dtype=np.float64), axis=0)
    return positions


def _quat_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Cannot convert zero-length quaternion to rotation matrix")
    x, y, z, w = q / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / np.linalg.norm(q)


def _make_transform(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_xyzw_to_rotmat(quat_xyzw)
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


# Tracker name prefix -> structure part whose measured radius separates the tracked
# point (on the surface) from the part's centre (apple) or woody axis (branch, spur,
# stem). 'SpurStart' and 'SpurEnd' both use the spur, 'StemStart' the stem.
TRACKER_PART_PREFIXES = (("Branch", "primary"), ("Spur", "spur"), ("Stem", "stem"), ("Apple", "apple"))


def part_for_tracker(name: str) -> str | None:
    for prefix, part in TRACKER_PART_PREFIXES:
        if str(name).startswith(prefix):
            return part
    return None
# AprilTag pose convention (apriltag_pose / pupil_apriltags): the tag frame's +z
# points into the tag, away from the camera, i.e. into the object it is stuck on.
TAG_INTO_SURFACE_SIGN = 1.0


def _identity_tracking_geometry(
    positions: dict[str, np.ndarray],
    poses: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return positions, poses


def _tag_to_part_geometry(
    positions: dict[str, np.ndarray],
    poses: dict[str, np.ndarray],
    radii_m: dict[str, float] | None,
    sign: float = TAG_INTO_SURFACE_SIGN,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Move each tracked point from the tag on the surface to the part centre/axis.

    ``pos_part = pos_tag + sign * r_part * z_tag`` (tag z in base frame). The
    rotation is kept, so poses stay tag-aligned. Without radii this is a no-op.
    """
    if not radii_m:
        return positions, poses
    out_positions: dict[str, np.ndarray] = {}
    out_poses: dict[str, np.ndarray] = {}
    for name, pos in positions.items():
        pose = np.asarray(poses[name], dtype=np.float64).reshape(4, 4).copy()
        radius = float(radii_m.get(name, 0.0))
        shifted = np.asarray(pos, dtype=np.float64) + float(sign) * radius * pose[:3, 2]
        pose[:3, 3] = shifted
        out_positions[name] = shifted
        out_poses[name] = pose
    return out_positions, out_poses


def _merge_measured_parts(stored: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    """Measured values win; collection-time fields (e.g. connection angles) are kept."""
    merged = json.loads(json.dumps(stored))
    for name, values in measured.items():
        entry = merged.setdefault(name, {})
        entry.update(values)
        entry["geometry_source"] = "measured"
    return merged


def _part_radii_from_parts(
    parts: dict[str, Any],
    names=TRACKED_NAMES,
    radius_shift: dict[str, bool] | None = None,
) -> dict[str, float]:
    """Radius to add for each tracker; 0 for trackers with ``radius_shift: false``."""
    radius_shift = radius_shift or {}
    radii = {}
    missing = []
    for tracker in names:
        if not radius_shift.get(tracker, True):
            radii[tracker] = 0.0
            continue
        part = part_for_tracker(tracker)
        if part is None:
            raise ValueError(
                f"Tracker {tracker!r} maps to no structure part (names must start with "
                f"{', '.join(prefix for prefix, _ in TRACKER_PART_PREFIXES)}), or set radius_shift: false"
            )
        radius = (parts.get(part) or {}).get("radius_m")
        if radius is None or not np.isfinite(float(radius)) or float(radius) < 0.0:
            missing.append(f"{part}.radius_m")
            continue
        radii[tracker] = float(radius)
    if missing:
        raise ValueError(f"Measured parts are missing {', '.join(sorted(set(missing)))}")
    return radii


# Convention: a tag offset in tracking_config.yaml only ever takes the tracked
# point from the tag centre to the part *surface* (e.g. a 3D-printed clip's contact
# point). Compile always adds the measured radius from there to the part's axis or
# centre, for every part, whether or not the tag has an offset -- except for
# trackers whose config says radius_shift: false (their radius is 0 here).


def _correct_snapshot(snapshot: dict[str, Any], radii_m: dict[str, float], sign: float) -> dict[str, Any]:
    """Apply the tag->part shift to a stored snapshot, keeping raw values as *_tag.

    Every tracker in ``radii_m`` whose ``<key>_pose_4x4`` is in the snapshot is
    corrected; snapshots without the apple are returned unchanged.
    """
    present = [name for name in radii_m if f"{tracker_key(name)}_pose_4x4" in snapshot]
    if not snapshot or "Apple" not in present:
        return snapshot
    out = dict(snapshot)
    for stale in ("woody_part_start_pos", "woody_part_end_pos", "woody_bending_angles"):
        out.pop(stale, None)  # chords of raw tag positions (older snapshots)
    positions, poses = {}, {}
    for tracker in present:
        key = tracker_key(tracker)
        pose = np.asarray(snapshot[f"{key}_pose_4x4"], dtype=np.float64).reshape(4, 4)
        poses[tracker] = pose
        positions[tracker] = pose[:3, 3].copy()
        out[f"{key}_pos_tag"] = pose[:3, 3].tolist()
        out[f"{key}_pose_4x4_tag"] = pose.reshape(-1).tolist()
    positions, poses = _tag_to_part_geometry(positions, poses, radii_m, sign)
    for tracker in present:
        key = tracker_key(tracker)
        out[f"{key}_pos"] = positions[tracker].tolist()
        out[f"{key}_pose_4x4"] = poses[tracker].reshape(-1).tolist()
    out["tag_to_part_corrected"] = True
    return out


def _as_list(value: Any, *, dtype=np.float32) -> list[float]:
    return np.asarray(value, dtype=dtype).reshape(-1).tolist()


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == 0:
        return values
    out = values.copy()
    for idx in range(1, out.shape[0]):
        out[idx] = alpha * out[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def _update_pose_translation(flat_pose: list[float], pos_xyz: np.ndarray) -> list[float]:
    pose = np.asarray(flat_pose, dtype=np.float64).reshape(4, 4).copy()
    pose[:3, 3] = np.asarray(pos_xyz, dtype=np.float64).reshape(3)
    return pose.reshape(-1).tolist()


def _unified_schema(n_holds: int, n_directions: int, woody_keys=("branch", "spur")) -> pa.Schema:
    """Arrow schema with enforced dimensions for every model-facing vector."""
    vector = lambda size: pa.list_(pa.float32(), int(size))
    return pa.schema([
        pa.field("episode_id", pa.string()),
        pa.field("timestamp", pa.float64(), metadata={b"unit": b"Unix seconds"}),
        pa.field("step_idx", pa.int32()),
        pa.field("hold_step_idx", pa.int32()),
        pa.field("hold_index", pa.int32()),
        pa.field("ft_wrist", vector(6), metadata={b"frame": b"robot EE/body"}),
        pa.field("ft_wrist_raw", vector(6), metadata={b"frame": b"robot EE/body"}),
        pa.field("ft_wrist_baseline", vector(6), metadata={b"frame": b"robot EE/body"}),
        pa.field("tau_J_d", vector(7), metadata={b"unit": b"N m", b"source": b"RobotState.tau_J_d"}),
        pa.field("joint_pos", vector(7), metadata={b"unit": b"rad", b"source": b"RobotState.q"}),
        pa.field("tcp_velocity", vector(6)),
        pa.field("action_wrench_ee", vector(6), metadata={b"semantics": b"commanded EE twist"}),
        pa.field("tcp_pos", vector(3)),
        pa.field("tcp_pose_4x4", vector(16), metadata={b"frame": b"franka_base_o"}),
        pa.field("task_prop_gains", vector(6), metadata={b"semantics": b"pose proportional gains"}),
        pa.field("task_deriv_gains", vector(6), metadata={b"semantics": b"pose derivative gains"}),
        pa.field("apple_pos", vector(3), metadata={b"frame": b"franka_base_o"}),
        pa.field("apple_pose_4x4", vector(16), metadata={b"frame": b"franka_base_o"}),
        *[pa.field(f"{key}_pose_4x4", vector(16), metadata={b"frame": b"franka_base_o"}) for key in woody_keys],
        pa.field("apple_pos_tag", vector(3), metadata={b"frame": b"franka_base_o", b"semantics": b"raw tag centre"}),
        pa.field("apple_pose_4x4_tag", vector(16), metadata={b"frame": b"franka_base_o", b"semantics": b"raw tag pose"}),
        *[
            pa.field(f"{key}_pose_4x4_tag", vector(16), metadata={b"frame": b"franka_base_o", b"semantics": b"raw tag pose"})
            for key in woody_keys
        ],
        pa.field("hold_number", vector(n_holds), metadata={b"encoding": b"one_hot"}),
        pa.field("direction", vector(n_directions), metadata={b"encoding": b"one_hot"}),
        pa.field("phase", pa.int8(), metadata={b"encoding": b"moving=0, hold=1"}),
        pa.field("phase_name", pa.string()),
        pa.field("sample_label", pa.string()),
        pa.field("amplitude_m", pa.float32()),
        pa.field("target_pose_4x4", vector(16), metadata={b"frame": b"franka_base_o"}),
        pa.field("excitation_direction", vector(3)),
        pa.field("camera_timestamp", pa.float64()),
        pa.field("robot_camera_timestamp_offset_s", pa.float64()),
        pa.field("camera_window_start_timestamp", pa.float64()),
        pa.field("camera_window_end_timestamp", pa.float64()),
        pa.field("camera_frame_count", pa.int16()),
        pa.field("camera_selected_timestamps", pa.list_(pa.float64())),
        pa.field("camera_data_valid", pa.bool_()),
    ])


def _match_baseline_frames(
    source_ft: np.ndarray,
    target_frame_count: int,
    *,
    max_relative_difference: float = 0.10,
) -> np.ndarray:
    """Match baseline samples to regular samples by frame index.

    The recordings are expected to have nearly the same frame rate. Keep
    samples at their original indices, truncate an overlong baseline, and
    repeat its final sample only when the regular run has a few extra frames.
    """
    source_ft = np.asarray(source_ft, dtype=np.float64)
    if source_ft.ndim != 2 or source_ft.shape[1] != 6 or len(source_ft) == 0:
        raise ValueError("Baseline wrench data must be a non-empty array of shape (n, 6)")
    if target_frame_count < 1:
        raise ValueError("Regular recording must contain at least one frame")

    relative_difference = abs(len(source_ft) - target_frame_count) / max(
        len(source_ft), target_frame_count
    )
    if relative_difference > max_relative_difference:
        raise ValueError(
            f"Baseline and regular frame counts differ by more than "
            f"{max_relative_difference:.0%}: "
            f"baseline={len(source_ft)}, regular={target_frame_count}"
        )
    else:
        print(
            f"Baseline and regular frame counts differ by {relative_difference}: "
            f"baseline={len(source_ft)}, regular={target_frame_count}"
        )

    matched = source_ft[:target_frame_count]
    if len(matched) < target_frame_count:
        matched = np.pad(
            matched,
            ((0, target_frame_count - len(matched)), (0, 0)),
            mode="edge",
        )
    return matched


def compile_static_episode(
    robot_path: Path | str,
    tracking_path: Path | str,
    output_path: Path | str,
    *,
    camera_frame_count: int = 5,
    max_camera_delta_s: float = 1.0,
    camera_ema_alpha: float = 1.0,
    baseline_path: Path | str | None = None,
    parts: dict[str, Any] | None = None,
    tag_to_part_sign: float = TAG_INTO_SURFACE_SIGN,
    command_argv: list[str] | None = None,
) -> Path:
    """Align robot rows with camera frames and write one unified episode.

    ``parts`` are the measured structure parts (``primary``/``spur``/``stem``/
    ``apple`` with ``radius_m`` etc.). When given, they replace the parts stored
    at collection time, and every tracked tag position is moved by the part
    radius to the part centre/axis (see ``_tag_to_part_geometry``). Without
    ``parts`` tag positions are used as recorded (old lab data).
    """
    robot_path = Path(robot_path)
    tracking_path = Path(tracking_path)
    output_path = Path(output_path)
    if int(camera_frame_count) < 1:
        raise ValueError("camera_frame_count must be >= 1")
    if not (0.0 < float(camera_ema_alpha) <= 1.0):
        raise ValueError("camera_ema_alpha must be in (0, 1].")

    robot_table = pq.read_table(robot_path)
    robot_rows_all = robot_table.to_pylist()
    robot_rows = [row for row in robot_rows_all if str(row.get("row_kind", "data")) != "metadata"]
    if not robot_rows:
        raise ValueError("Robot input contains no hold rows")
    baseline_applied = False
    if baseline_path is not None:
        baseline_path = Path(baseline_path)
        baseline_rows = [
            row for row in pq.read_table(baseline_path).to_pylist()
            if str(row.get("row_kind", "data")) != "metadata"
        ]
        baseline_by_part = {}
        for row in baseline_rows:
            # hold_index identifies pull/hold part 1, 2, ...; numeric phase
            # separates the moving and holding portions within that part.
            key = (int(row.get("hold_index", 0)), int(row.get("phase", 1)))
            baseline_by_part.setdefault(key, []).append(row)

        for hold_idx in sorted({int(row["hold_index"]) for row in robot_rows}):
            hold_rows = [row for row in robot_rows if int(row["hold_index"]) == hold_idx]
            phases = []
            for row in hold_rows:
                phase = int(row["phase"])
                if phase not in phases:
                    phases.append(phase)
            for phase in phases:
                current = [row for row in hold_rows if int(row["phase"]) == phase]
                source = sorted(
                    baseline_by_part.get((hold_idx, phase), []),
                    key=lambda row: int(row.get("hold_step_idx", 0)),
                )
                if not source:
                    raise ValueError(
                        "Baseline has no rows for "
                        f"hold_index={hold_idx}, phase={phase}"
                    )
                source_ft = np.asarray(
                    [row["ft_wrist"] for row in source],
                    dtype=np.float64,
                )
                matched = _match_baseline_frames(source_ft, len(current), max_relative_difference=.50)
                for row, values in zip(current, matched):
                    raw = np.asarray(
                        row.get("ft_wrist_raw", row["ft_wrist"]), dtype=np.float32
                    )
                    row["ft_wrist_raw"] = raw
                    row["ft_wrist_baseline"] = values.astype(np.float32)
                    row["ft_wrist"] = (raw.astype(np.float64) - values).astype(np.float32)
        baseline_applied = True
    required_robot_fields = {
        "timestamp", "hold_index", "ft_wrist", "tau_J_d", "joint_pos",
        "tcp_velocity", "action_wrench_ee", "tcp_pos", "tcp_pose_4x4", "target_pose_4x4",
        "task_prop_gains", "task_deriv_gains",
        "hold_number", "direction", "phase", "excitation_direction",
    }
    missing = required_robot_fields - set(robot_rows[0])
    if missing:
        raise ValueError(f"Robot input is missing required fields: {sorted(missing)}")

    robot_metadata = _read_dataset_metadata(robot_path)
    if robot_rows_all and str(robot_rows_all[0].get("row_kind", "")) == "metadata":
        embedded = robot_rows_all[0].get("metadata_json")
        if embedded:
            try:
                robot_metadata = json.loads(embedded)
            except json.JSONDecodeError:
                pass
    if baseline_applied:
        robot_metadata["dynamic_baseline"] = {
            **dict(robot_metadata.get("dynamic_baseline", {}) or {}),
            "role": "corrected_collect_run",
            "applied": True,
            "application_stage": "compile_static_episode",
            "source_path": str(baseline_path.resolve()),
            "source_sha256": _sha256(baseline_path),
        }
    tracking_metadata = _read_dataset_metadata(tracking_path)
    names = tracked_names_from_metadata(tracking_metadata)
    woody_names = [name for name in names if name != "Apple"]
    radius_shift = radius_shift_from_metadata(tracking_metadata)
    part_radii = None
    if parts is not None:
        part_radii = _part_radii_from_parts(parts, names, radius_shift)
    _require_tracking_frame_base(tracking_metadata, tracking_path)
    camera_to_base_4x4 = _load_camera_to_base(tracking_metadata, tracking_path)
    camera_frames = _load_tracking_frames(tracking_path, names)
    camera_index = _build_tracking_index(camera_frames, names)

    def _tag_geometry(selected: pd.DataFrame):
        positions_tag = _median_positions(selected, names)
        poses_tag = {}
        for name in names:
            pose_rows = selected[selected["name"] == name][["qx", "qy", "qz", "qw"]].to_numpy()
            quat = np.median(pose_rows.astype(np.float64), axis=0)
            poses_tag[name] = _make_transform(positions_tag[name], quat)
        return positions_tag, poses_tag

    rest_timestamp = float(
        robot_metadata.get(
            "rest_reference_timestamp",
            min(float(row["timestamp"]) for row in robot_rows),
        )
    )
    rest_frames = _select_frames(
        camera_index,
        center=rest_timestamp,
        count=int(camera_frame_count),
        max_delta_s=float(max_camera_delta_s),
        prefer_before=True,
    )
    rest_positions_tag, rest_poses_tag = _tag_geometry(rest_frames)
    rest_positions, rest_poses = _tag_to_part_geometry(
        rest_positions_tag, rest_poses_tag, part_radii, tag_to_part_sign
    )

    hold_indices = sorted({int(row["hold_index"]) for row in robot_rows if int(row["hold_index"]) >= 0})
    hold_camera_summaries: list[dict[str, Any]] = []
    for hold_idx in hold_indices:
        hold_rows = [row for row in robot_rows if int(row["hold_index"]) == hold_idx]
        timestamps = np.asarray([float(row["timestamp"]) for row in hold_rows])
        start = float(timestamps.min())
        end = float(timestamps.max())
        center = float((start + end) / 2.0)
        hold_camera_timestamps = _complete_timestamps_in_interval(camera_index, (start, end))
        selected = _select_frames(
            camera_index,
            center=center,
            count=int(camera_frame_count),
            max_delta_s=float(max_camera_delta_s),
            interval=(start, end),
        )
        selected_timestamps = selected["timestamp"].astype(float).tolist()
        camera_center = float(np.median(selected_timestamps))
        unique_selected_timestamps = sorted(set(selected_timestamps))
        hold_camera_summaries.append({
            "hold_index": hold_idx,
            "robot_start_timestamp": start,
            "robot_end_timestamp": end,
            "robot_midpoint_timestamp": center,
            "complete_camera_timestamps": hold_camera_timestamps.tolist(),
            "selected_camera_timestamps": selected_timestamps,
            "camera_median_timestamp": camera_center,
            "camera_frame_count": len(unique_selected_timestamps),
        })

    episode_id = str(robot_metadata.get("episode_id", ""))
    output_rows: list[dict[str, Any]] = []
    for step_idx, robot_row in enumerate(robot_rows):
        hold_idx = int(robot_row["hold_index"])
        timestamp = float(robot_row["timestamp"])
        selected = _select_frames(
            camera_index,
            center=timestamp,
            count=int(camera_frame_count),
            max_delta_s=float(max_camera_delta_s),
        )
        positions_tag, poses_tag = _tag_geometry(selected)
        positions, poses = _tag_to_part_geometry(positions_tag, poses_tag, part_radii, tag_to_part_sign)
        selected_timestamps = selected["timestamp"].astype(float).tolist()
        camera_timestamp = float(np.median(selected_timestamps))
        unique_selected_timestamps = sorted(set(selected_timestamps))
        output_rows.append({
            "episode_id": episode_id,
            "timestamp": timestamp,
            "step_idx": int(step_idx),
            "hold_step_idx": int(robot_row.get("hold_step_idx", step_idx)),
            "hold_index": hold_idx,
            "ft_wrist": _as_list(robot_row["ft_wrist"]),
            "ft_wrist_raw": _as_list(robot_row.get("ft_wrist_raw", robot_row["ft_wrist"])),
            "ft_wrist_baseline": _as_list(robot_row.get("ft_wrist_baseline", np.zeros(6))),
            "tau_J_d": _as_list(robot_row["tau_J_d"]),
            "joint_pos": _as_list(robot_row["joint_pos"]),
            "tcp_velocity": _as_list(robot_row["tcp_velocity"]),
            "action_wrench_ee": _as_list(robot_row["action_wrench_ee"]),
            "tcp_pos": _as_list(robot_row["tcp_pos"]),
            "tcp_pose_4x4": _as_list(robot_row["tcp_pose_4x4"]),
            "task_prop_gains": _as_list(robot_row["task_prop_gains"]),
            "task_deriv_gains": _as_list(robot_row["task_deriv_gains"]),
            "apple_pos": _as_list(positions["Apple"]),
            "apple_pose_4x4": _as_list(poses["Apple"]),
            **{f"{tracker_key(name)}_pose_4x4": _as_list(poses[name]) for name in woody_names},
            "apple_pos_tag": _as_list(positions_tag["Apple"]),
            "apple_pose_4x4_tag": _as_list(poses_tag["Apple"]),
            **{f"{tracker_key(name)}_pose_4x4_tag": _as_list(poses_tag[name]) for name in woody_names},
            "hold_number": _as_list(robot_row["hold_number"]),
            "direction": _as_list(robot_row["direction"]),
            "phase": int(robot_row["phase"]),
            "phase_name": str(robot_row.get("phase_name", "hold")),
            "sample_label": str(robot_row.get("sample_label", robot_row.get("phase_name", "hold"))),
            "amplitude_m": float(robot_row.get("amplitude_m", math.nan)),
            "target_pose_4x4": _as_list(robot_row["target_pose_4x4"]),
            "excitation_direction": _as_list(robot_row["excitation_direction"]),
            "camera_timestamp": camera_timestamp,
            "robot_camera_timestamp_offset_s": timestamp - camera_timestamp,
            "camera_window_start_timestamp": min(unique_selected_timestamps),
            "camera_window_end_timestamp": max(unique_selected_timestamps),
            "camera_frame_count": len(unique_selected_timestamps),
            "camera_selected_timestamps": selected_timestamps,
            "camera_data_valid": True,
        })

    n_holds = len(output_rows[0]["hold_number"])
    n_directions = len(output_rows[0]["direction"])

    if float(camera_ema_alpha) < 1.0:
        apple_positions = np.asarray([row["apple_pos"] for row in output_rows], dtype=np.float64)
        smoothed_apple = _ema(apple_positions, float(camera_ema_alpha))
        for idx, row in enumerate(output_rows):
            row["apple_pos"] = _as_list(smoothed_apple[idx])
            row["apple_pose_4x4"] = _update_pose_translation(row["apple_pose_4x4"], smoothed_apple[idx])

    table = pa.Table.from_pylist(
        output_rows,
        schema=_unified_schema(n_holds, n_directions, [tracker_key(name) for name in woody_names]),
    )
    # post_grasp_geometry is what the robot recorded after the grasp settled
    # (robot state + camera_snapshot); it is never synthesised from pull rows.
    post_grasp_geometry = dict(robot_metadata.get("post_grasp_geometry", {}) or {})
    first_row = output_rows[0]
    pull_start_tracking = {
        "timestamp": first_row["timestamp"],
        "hold_index": first_row["hold_index"],
        "hold_step_idx": first_row["hold_step_idx"],
        "tcp_pos": first_row["tcp_pos"],
        "tcp_pose_4x4": first_row["tcp_pose_4x4"],
        "target_pose_4x4": first_row["target_pose_4x4"],
        "apple_pos": first_row["apple_pos"],
        "apple_pose_4x4": first_row["apple_pose_4x4"],
    }
    rest_snapshot_during_run = {
        "timestamp": rest_timestamp,
        "tcp_pos": robot_rows[0]["tcp_pos"],
        "tcp_pose_4x4": robot_rows[0]["tcp_pose_4x4"],
        "target_pose_4x4": robot_rows[0]["target_pose_4x4"],
        **{f"{tracker_key(name)}_pos": np.asarray(rest_positions[name]).tolist() for name in names},
        **{f"{tracker_key(name)}_pose_4x4": rest_poses[name].reshape(-1).tolist() for name in names},
        "camera_selected_timestamps": rest_frames["timestamp"].astype(float).tolist(),
        "camera_frame_count": len(rest_frames),
    }
    pre_grasp_geometry = dict(robot_metadata.get("pre_grasp_geometry", {}) or {})
    tag_to_part_correction: dict[str, Any] = {"applied": False}
    if parts is not None:
        pre_grasp_geometry["parts"] = _merge_measured_parts(pre_grasp_geometry.get("parts") or {}, parts)
        for key in ("under_gravity_snapshot", "lengthened_snapshot"):
            pre_grasp_geometry[key] = _correct_snapshot(
                dict(pre_grasp_geometry.get(key) or {}), part_radii, tag_to_part_sign
            )
        if post_grasp_geometry.get("camera_snapshot"):
            post_grasp_geometry["camera_snapshot"] = _correct_snapshot(
                dict(post_grasp_geometry["camera_snapshot"]), part_radii, tag_to_part_sign
            )
        # Connection angles were computed from raw tag positions at collection
        # time; recompute them from the corrected lengthened snapshot.
        pre_grasp_geometry = update_pre_grasp_geometry_with_snapshots(pre_grasp_geometry)
        tag_to_part_correction = {
            "applied": True,
            "formula": "pos_part = pos_tag + sign * radius_m * z_tag (tag z in base frame)",
            "sign": float(tag_to_part_sign),
            "sign_convention": "+1: AprilTag +z points into the tag, i.e. into the part",
            "radius_m": {name: part_radii[name] for name in names},
            "radius_shift": {name: bool(radius_shift.get(name, True)) for name in names},
            "tag_offset_convention": (
                "tracked point = tag centre + tracking_config offset, which ends on the part "
                "surface; the measured radius is then added along the tracked frame's +z for every "
                "tracker except those with radius_shift: false (radius 0)"
            ),
            "part_for_tracker": {name: part_for_tracker(name) for name in names},
            "raw_fields": ["apple_pos_tag", "apple_pose_4x4_tag"]
            + [f"{tracker_key(name)}_pose_4x4_tag" for name in woody_names],
        }
    pre_grasp_geometry["rest_snapshot_during_run"] = rest_snapshot_during_run
    metadata_dump = {
        **robot_metadata,
        "source_files": {
            "robot": _source_info(robot_path),
            "tracking": _source_info(tracking_path),
        },
        "source_metadata_summary": {
            "robot_dump": robot_metadata.get("dump", {}),
            "pre_grasp_geometry": robot_metadata.get("pre_grasp_geometry", {}),
            "post_grasp_geometry": robot_metadata.get("post_grasp_geometry", {}),
            "tracking_coordinate_frame": tracking_metadata.get("coordinate_frame"),
        },
        "field_layout": {
            "ft_wrist": {"dim": 6, "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]},
            "ft_wrist_raw": {
                "dim": 6, "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                "description": "uncorrected measured/model-estimated wrench before dynamic baseline subtraction",
            },
            "ft_wrist_baseline": {
                "dim": 6, "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                "description": "time-varying unloaded baseline subtracted from ft_wrist_raw",
            },
            "tau_J_d": {
                "dim": 7, "order": [f"joint_{i}" for i in range(1, 8)], "unit": "N m",
                "description": "commanded/desired link-side joint torque without gravity",
            },
            "joint_pos": {
                "dim": 7, "order": [f"joint_{i}" for i in range(1, 8)], "unit": "rad",
                "description": "measured joint positions",
            },
            "tcp_velocity": {"dim": 6, "order": ["vx", "vy", "vz", "wx", "wy", "wz"]},
            "action_wrench_ee": {
                "dim": 6,
                "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                "description": "per-frame pose-control wrench computed from pose error and twist",
            },
            "tcp_pos": {"dim": 3, "order": ["x", "y", "z"]},
            "tcp_pose_4x4": {"dim": 16, "reshape": [4, 4]},
            "task_prop_gains": {
                "dim": 6,
                "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                "description": "pose proportional gains used to compute action",
            },
            "task_deriv_gains": {
                "dim": 6,
                "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                "description": "pose derivative gains used to compute action",
            },
            "target_pose_4x4": {"dim": 16, "reshape": [4, 4]},
            "apple_pos": {"dim": 3, "order": ["x", "y", "z"]},
            "apple_pose_4x4": {"dim": 16, "reshape": [4, 4]},
            **{f"{tracker_key(name)}_pose_4x4": {"dim": 16, "reshape": [4, 4], "tracker": name}
               for name in woody_names},
            "hold_number": {"dim": n_holds, "encoding": "one_hot"},
            "direction": {"dim": n_directions, "encoding": "one_hot"},
            "phase": {"dim": 1, "encoding": {"moving": 0, "hold": 1}},
            "excitation_direction": {"dim": 3, "description": "unit pull direction"},
        },
        "pre_grasp_geometry": {
            **pre_grasp_geometry,
        },
        "post_grasp_geometry": post_grasp_geometry,
        "pull_start_tracking": pull_start_tracking,
        "tag_to_part_correction": tag_to_part_correction,
        "row_count": len(output_rows),
        "hold_count": len(hold_indices),
        "compiler": {
            "module": "real_robot_exps.compile_static_sysid",
            "command_argv": list(command_argv if command_argv is not None else sys.argv),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "pyarrow_version": pa.__version__,
            "repository_git_commit": _git_commit(Path(__file__).resolve().parents[1]),
            "tracking_git_commit": _git_commit(Path(__file__).resolve().parents[1] / "at-tracking"),
        },
    }
    metadata_dump.update({
        "episode_id": episode_id,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": "franka_base_o",
        "data_frame": "franka_base_o_frame",
        "position_unit": "m",
        "angle_unit": "rad",
        "timestamp_clock": "Unix wall clock from time.time() on the shared host",
        "timestamp_unit": "seconds",
        "camera_to_base_4x4_used": camera_to_base_4x4.tolist(),
        "topology": {
            "tracked_names": list(names),
            "column_key_for_tracker": {name: tracker_key(name) for name in names},
            "tracked_tag_ids": tracking_metadata.get("tracked_tag_ids"),
        },
        "rest_reference_timestamp": rest_timestamp,
        "rest_selected_camera_timestamps": rest_frames["timestamp"].astype(float).tolist(),
        "camera_aggregation": {
            "method": "coordinate-wise median of nearest complete valid frames",
            "requested_frame_count": int(camera_frame_count),
            "max_camera_delta_s": float(max_camera_delta_s),
            "ema_alpha": float(camera_ema_alpha),
            "missing_pose_sentinel": [0.0, 0.0, 0.0],
            "required_tracker_names": list(names),
            "rest_selection": "nearest frames at or before rest_reference_timestamp",
            "hold_selection": "nearest frames within robot hold interval when available",
        },
        "hold_camera_summaries": hold_camera_summaries,
    })
    compilation_metadata = metadata_dump
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[b"dataset_metadata"] = json.dumps(
        compilation_metadata, sort_keys=True, default=str
    ).encode("utf-8")
    table = table.replace_schema_metadata(schema_metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, type=Path, help="Raw robot hold Parquet")
    parser.add_argument("--tracking", required=True, type=Path, help="Raw tracking Parquet")
    parser.add_argument("--baseline", type=Path, help="Post-run joint-velocity baseline Parquet")
    parser.add_argument("--output", required=True, type=Path, help="Unified episode Parquet")
    parser.add_argument("--camera-frames", type=int, default=5, help="Camera frames per estimate")
    parser.add_argument(
        "--max-camera-delta",
        type=float,
        default=1.0,
        help="Maximum allowed camera-to-reference time difference in seconds",
    )
    parser.add_argument(
        "--camera-ema-alpha",
        type=float,
        default=1.0,
        help="EMA alpha for smoothing camera geometry; 1.0 disables smoothing",
    )
    parser.add_argument(
        "--parts-json",
        type=Path,
        default=None,
        help="Measured parts JSON ({primary,spur,stem,apple: {radius_m, length_m, mass_kg, ...}} "
        "or a field apple.json with a 'parts' key); enables the tag->part radius correction "
        "(radius needed only for parts of tracked tags with radius_shift)",
    )
    args = parser.parse_args()
    parts = None
    if args.parts_json is not None:
        payload = json.loads(args.parts_json.read_text(encoding="utf-8"))
        parts = payload.get("parts", payload)
    output = compile_static_episode(
        args.robot,
        args.tracking,
        args.output,
        camera_frame_count=args.camera_frames,
        max_camera_delta_s=args.max_camera_delta,
        camera_ema_alpha=args.camera_ema_alpha,
        baseline_path=args.baseline,
        parts=parts,
        command_argv=sys.argv,
    )
    print(f"Wrote unified static system-ID episode to {output}")


if __name__ == "__main__":
    main()
