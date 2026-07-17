"""Compile separately recorded robot and AprilTag data into one episode Parquet.

Both inputs use Unix wall-clock seconds from ``time.time()``. Robot measurements
remain at the robot policy rate. A small median-filtered camera estimate is
attached to every robot row in the corresponding static hold.

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


SCHEMA_NAME = "real_static_sysid_episode"
SCHEMA_VERSION = "1.0.0"
TRACKED_NAMES = ("Branch", "Spur", "Apple")
WOODY_PART_NAMES = ("Branch", "Spur", "Apple")


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


def _load_tracking_frames(path: Path) -> pd.DataFrame:
    tracking = pq.read_table(path).to_pandas()
    expected = ["timestamp", "name", "x", "y", "z", "qx", "qy", "qz", "qw"]
    if list(tracking.columns) != expected:
        if len(tracking.columns) == len(expected):
            # Backward compatibility with the old integer-column DataCollector.
            tracking.columns = expected
        else:
            raise ValueError(f"Unexpected tracking columns: {list(tracking.columns)}")

    tracking = tracking[tracking["name"].isin(TRACKED_NAMES)].copy()
    xyz = tracking[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(xyz.to_numpy()).all(axis=1)
    # The tracker uses exactly [0,0,0] as its missing-detection sentinel.
    valid &= ~np.isclose(xyz.to_numpy(), 0.0).all(axis=1)
    tracking = tracking.loc[valid].copy()
    tracking[["x", "y", "z"]] = xyz.loc[valid]
    tracking["timestamp"] = pd.to_numeric(tracking["timestamp"], errors="coerce")
    tracking = tracking[np.isfinite(tracking["timestamp"])].copy()

    rows: list[dict[str, Any]] = []
    for timestamp, group in tracking.groupby("timestamp", sort=True):
        by_name = {str(row["name"]): row for _, row in group.iterrows()}
        if not all(name in by_name for name in TRACKED_NAMES):
            continue
        row: dict[str, Any] = {"timestamp": float(timestamp)}
        for name in TRACKED_NAMES:
            row[name] = np.array(
                [by_name[name]["x"], by_name[name]["y"], by_name[name]["z"]],
                dtype=np.float64,
            )
        rows.append(row)
    if not rows:
        raise ValueError("Tracking input contains no complete valid Branch/Spur/Apple frames")
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _select_frames(
    frames: pd.DataFrame,
    *,
    center: float,
    count: int,
    max_delta_s: float,
    interval: tuple[float, float] | None = None,
    prefer_before: bool = False,
) -> pd.DataFrame:
    candidates = frames
    if interval is not None:
        start, end = interval
        in_interval = frames[(frames["timestamp"] >= start) & (frames["timestamp"] <= end)]
        if not in_interval.empty:
            candidates = in_interval
    if prefer_before:
        before = candidates[candidates["timestamp"] <= center]
        if not before.empty:
            candidates = before
    candidates = candidates.assign(
        _abs_delta=(candidates["timestamp"] - float(center)).abs()
    )
    candidates = candidates[candidates["_abs_delta"] <= float(max_delta_s)]
    selected = candidates.nsmallest(int(count), "_abs_delta").sort_values("timestamp")
    if selected.empty:
        raise ValueError(
            f"No complete camera frames within {max_delta_s:.3f}s of timestamp {center:.6f}"
        )
    return selected.drop(columns=["_abs_delta"])


def _median_positions(frames: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        name: np.median(np.stack(frames[name].to_list(), axis=0), axis=0)
        for name in TRACKED_NAMES
    }


def _endpoints(
    positions: dict[str, np.ndarray], fruiting_base_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    branch = positions["Branch"]
    spur = positions["Spur"]
    apple = positions["Apple"]
    starts = np.stack([fruiting_base_pos, branch, spur], axis=0)
    ends = np.stack([branch, spur, apple], axis=0)
    return starts, ends, ends - starts


def _chord_deflections(chords: np.ndarray, rest_chords: np.ndarray) -> np.ndarray:
    chord_norms = np.linalg.norm(chords, axis=1)
    rest_norms = np.linalg.norm(rest_chords, axis=1)
    if np.any(chord_norms < 1e-12) or np.any(rest_norms < 1e-12):
        raise ValueError("Cannot compute bending angle for a zero-length woody chord")
    cosine = np.sum(chords * rest_chords, axis=1) / (chord_norms * rest_norms)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _as_list(value: Any, *, dtype=np.float32) -> list[float]:
    return np.asarray(value, dtype=dtype).reshape(-1).tolist()


def _unified_schema(n_holds: int, n_directions: int) -> pa.Schema:
    """Arrow schema with enforced dimensions for every model-facing vector."""
    vector = lambda size: pa.list_(pa.float32(), int(size))
    return pa.schema([
        pa.field("episode_id", pa.string()),
        pa.field("timestamp", pa.float64(), metadata={b"unit": b"Unix seconds"}),
        pa.field("step_idx", pa.int32()),
        pa.field("hold_step_idx", pa.int32()),
        pa.field("hold_index", pa.int32()),
        pa.field("ft_wrist", vector(6), metadata={b"frame": b"robot EE/body"}),
        pa.field("tau_J", vector(7), metadata={b"unit": b"N m", b"source": b"RobotState.tau_J"}),
        pa.field("tau_ext_hat_filtered", vector(7), metadata={b"unit": b"N m", b"source": b"RobotState.tau_ext_hat_filtered"}),
        pa.field("tau_J_d", vector(7), metadata={b"unit": b"N m", b"source": b"RobotState.tau_J_d"}),
        pa.field("tcp_velocity", vector(6)),
        pa.field("action", vector(6), metadata={b"semantics": b"commanded EE twist"}),
        pa.field("tcp_pos", vector(3)),
        pa.field("apple_pos", vector(3), metadata={b"frame": b"reference_apriltag"}),
        pa.field("woody_part_start_pos", vector(9)),
        pa.field("woody_part_end_pos", vector(9)),
        pa.field("woody_bending_angles", vector(3), metadata={b"unit": b"rad"}),
        pa.field("hold_number", vector(n_holds), metadata={b"encoding": b"one_hot"}),
        pa.field("direction", vector(n_directions), metadata={b"encoding": b"one_hot"}),
        pa.field("phase", pa.int8(), metadata={b"encoding": b"moving=0, hold=1"}),
        pa.field("phase_name", pa.string()),
        pa.field("amplitude_m", pa.float32()),
        pa.field("excitation_direction", vector(3)),
        pa.field("camera_timestamp", pa.float64()),
        pa.field("robot_camera_timestamp_offset_s", pa.float64()),
        pa.field("camera_window_start_timestamp", pa.float64()),
        pa.field("camera_window_end_timestamp", pa.float64()),
        pa.field("camera_frame_count", pa.int16()),
        pa.field("camera_selected_timestamps", pa.list_(pa.float64())),
        pa.field("camera_data_valid", pa.bool_()),
    ])


def compile_static_episode(
    robot_path: Path | str,
    tracking_path: Path | str,
    output_path: Path | str,
    *,
    camera_frame_count: int = 5,
    max_camera_delta_s: float = 1.0,
    fruiting_base_pos: np.ndarray | None = None,
    command_argv: list[str] | None = None,
) -> Path:
    robot_path = Path(robot_path)
    tracking_path = Path(tracking_path)
    output_path = Path(output_path)
    if int(camera_frame_count) < 1:
        raise ValueError("camera_frame_count must be >= 1")

    robot_table = pq.read_table(robot_path)
    robot_rows = robot_table.to_pylist()
    if not robot_rows:
        raise ValueError("Robot input contains no hold rows")
    required_robot_fields = {
        "timestamp", "hold_index", "ft_wrist", "tau_J", "tau_ext_hat_filtered",
        "tau_J_d", "tcp_velocity", "action",
        "tcp_pos", "hold_number", "direction", "phase", "excitation_direction",
    }
    missing = required_robot_fields - set(robot_rows[0])
    if missing:
        raise ValueError(f"Robot input is missing required fields: {sorted(missing)}")

    robot_metadata = _read_dataset_metadata(robot_path)
    tracking_metadata = _read_dataset_metadata(tracking_path)
    camera_frames = _load_tracking_frames(tracking_path)

    fruiting_base_was_explicit = fruiting_base_pos is not None
    if fruiting_base_pos is None:
        # Current assumption: the reference AprilTag origin is the fruiting base.
        # TODO: replace this fallback with a calibrated tag-to-base translation.
        fruiting_base_pos = np.asarray(
            tracking_metadata.get("fruiting_base_pos", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
    fruiting_base_pos = np.asarray(fruiting_base_pos, dtype=np.float64).reshape(3)

    rest_timestamp = float(
        robot_metadata.get(
            "rest_reference_timestamp",
            min(float(row["timestamp"]) for row in robot_rows),
        )
    )
    rest_frames = _select_frames(
        camera_frames,
        center=rest_timestamp,
        count=int(camera_frame_count),
        max_delta_s=float(max_camera_delta_s),
        prefer_before=True,
    )
    rest_positions = _median_positions(rest_frames)
    rest_starts, rest_ends, rest_chords = _endpoints(rest_positions, fruiting_base_pos)

    hold_indices = sorted({int(row["hold_index"]) for row in robot_rows})
    hold_geometry: dict[int, dict[str, Any]] = {}
    hold_camera_summaries: list[dict[str, Any]] = []
    for hold_idx in hold_indices:
        hold_rows = [row for row in robot_rows if int(row["hold_index"]) == hold_idx]
        timestamps = np.asarray([float(row["timestamp"]) for row in hold_rows])
        start = float(timestamps.min())
        end = float(timestamps.max())
        center = float((start + end) / 2.0)
        selected = _select_frames(
            camera_frames,
            center=center,
            count=int(camera_frame_count),
            max_delta_s=float(max_camera_delta_s),
            interval=(start, end),
        )
        positions = _median_positions(selected)
        starts, ends, chords = _endpoints(positions, fruiting_base_pos)
        bending = _chord_deflections(chords, rest_chords)
        selected_timestamps = selected["timestamp"].astype(float).tolist()
        camera_center = float(np.median(selected_timestamps))
        geometry = {
            "apple_pos": positions["Apple"],
            "woody_part_start_pos": starts.reshape(-1),
            "woody_part_end_pos": ends.reshape(-1),
            "woody_bending_angles": bending,
            "camera_selected_timestamps": selected_timestamps,
            "camera_timestamp": camera_center,
            "camera_window_start_timestamp": min(selected_timestamps),
            "camera_window_end_timestamp": max(selected_timestamps),
            "camera_frame_count": len(selected_timestamps),
        }
        hold_geometry[hold_idx] = geometry
        hold_camera_summaries.append({
            "hold_index": hold_idx,
            "robot_start_timestamp": start,
            "robot_end_timestamp": end,
            "robot_midpoint_timestamp": center,
            "selected_camera_timestamps": selected_timestamps,
            "camera_median_timestamp": camera_center,
            "bending_angles_rad": bending.tolist(),
        })

    episode_id = str(robot_metadata.get("episode_id", ""))
    output_rows: list[dict[str, Any]] = []
    for step_idx, robot_row in enumerate(robot_rows):
        hold_idx = int(robot_row["hold_index"])
        geometry = hold_geometry[hold_idx]
        timestamp = float(robot_row["timestamp"])
        output_rows.append({
            "episode_id": episode_id,
            "timestamp": timestamp,
            "step_idx": int(step_idx),
            "hold_step_idx": int(robot_row.get("hold_step_idx", step_idx)),
            "hold_index": hold_idx,
            "ft_wrist": _as_list(robot_row["ft_wrist"]),
            "tau_J": _as_list(robot_row["tau_J"]),
            "tau_ext_hat_filtered": _as_list(robot_row["tau_ext_hat_filtered"]),
            "tau_J_d": _as_list(robot_row["tau_J_d"]),
            "tcp_velocity": _as_list(robot_row["tcp_velocity"]),
            "action": _as_list(robot_row["action"]),
            "tcp_pos": _as_list(robot_row["tcp_pos"]),
            "apple_pos": _as_list(geometry["apple_pos"]),
            "woody_part_start_pos": _as_list(geometry["woody_part_start_pos"]),
            "woody_part_end_pos": _as_list(geometry["woody_part_end_pos"]),
            "woody_bending_angles": _as_list(geometry["woody_bending_angles"]),
            "hold_number": _as_list(robot_row["hold_number"]),
            "direction": _as_list(robot_row["direction"]),
            "phase": int(robot_row["phase"]),
            "phase_name": str(robot_row.get("phase_name", "hold")),
            "amplitude_m": float(robot_row.get("amplitude_m", math.nan)),
            "excitation_direction": _as_list(robot_row["excitation_direction"]),
            "camera_timestamp": float(geometry["camera_timestamp"]),
            "robot_camera_timestamp_offset_s": timestamp - float(geometry["camera_timestamp"]),
            "camera_window_start_timestamp": float(geometry["camera_window_start_timestamp"]),
            "camera_window_end_timestamp": float(geometry["camera_window_end_timestamp"]),
            "camera_frame_count": int(geometry["camera_frame_count"]),
            "camera_selected_timestamps": geometry["camera_selected_timestamps"],
            "camera_data_valid": True,
        })

    n_holds = len(output_rows[0]["hold_number"])
    n_directions = len(output_rows[0]["direction"])
    table = pa.Table.from_pylist(
        output_rows,
        schema=_unified_schema(n_holds, n_directions),
    )
    compilation_metadata = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "episode_id": episode_id,
        "timestamp_clock": "Unix wall clock from time.time() on the shared host",
        "timestamp_unit": "seconds",
        "coordinate_frame": "reference_apriltag",
        "position_unit": "m",
        "angle_unit": "rad",
        "topology": {
            "node_order": ["fruiting_base", "Branch", "Spur", "Apple"],
            "junction_names": list(WOODY_PART_NAMES),
            "n_woody_parts": 3,
            "start_nodes": ["fruiting_base", "Branch", "Spur"],
            "end_nodes": ["Branch", "Spur", "Apple"],
            "shared_endpoints": True,
        },
        "fruiting_base_pos": fruiting_base_pos.tolist(),
        "fruiting_base_source": (
            "explicit compiler argument"
            if fruiting_base_was_explicit
            else "tracking metadata; currently reference-tag origin"
        ),
        "bending_definition": (
            "Per-part unsigned chord deflection from frame-0/rest: "
            "acos(clip(dot(chord_t,chord_0)/(|chord_t||chord_0|),-1,1))"
        ),
        "rest_reference_timestamp": rest_timestamp,
        "rest_selected_camera_timestamps": rest_frames["timestamp"].astype(float).tolist(),
        "rest_woody_part_start_pos": rest_starts.reshape(-1).tolist(),
        "rest_woody_part_end_pos": rest_ends.reshape(-1).tolist(),
        "rest_chord_vectors": rest_chords.reshape(-1).tolist(),
        "camera_aggregation": {
            "method": "coordinate-wise median of nearest complete valid frames",
            "requested_frame_count": int(camera_frame_count),
            "max_camera_delta_s": float(max_camera_delta_s),
            "missing_pose_sentinel": [0.0, 0.0, 0.0],
            "required_tracker_names": list(TRACKED_NAMES),
            "rest_selection": "nearest frames at or before rest_reference_timestamp",
            "hold_selection": "nearest frames within robot hold interval when available",
        },
        "hold_camera_summaries": hold_camera_summaries,
        "field_layout": {
            "ft_wrist": {"dim": 6, "order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]},
            "tau_J": {
                "dim": 7, "order": [f"joint_{i}" for i in range(1, 8)], "unit": "N m",
                "description": "measured link-side joint torque sensor signals",
            },
            "tau_ext_hat_filtered": {
                "dim": 7, "order": [f"joint_{i}" for i in range(1, 8)], "unit": "N m",
                "description": "low-pass filtered external torque estimate; excludes configured EE/load and robot dynamics",
            },
            "tau_J_d": {
                "dim": 7, "order": [f"joint_{i}" for i in range(1, 8)], "unit": "N m",
                "description": "desired link-side joint torque without gravity",
            },
            "tcp_velocity": {"dim": 6, "order": ["vx", "vy", "vz", "wx", "wy", "wz"]},
            "action": {"dim": 6, "order": ["vx", "vy", "vz", "wx", "wy", "wz"]},
            "tcp_pos": {"dim": 3, "order": ["x", "y", "z"]},
            "apple_pos": {"dim": 3, "order": ["x", "y", "z"]},
            "woody_part_start_pos": {"dim": 9, "reshape": [3, 3]},
            "woody_part_end_pos": {"dim": 9, "reshape": [3, 3]},
            "woody_bending_angles": {"dim": 3, "part_order": list(WOODY_PART_NAMES)},
            "hold_number": {"dim": n_holds, "encoding": "one_hot"},
            "direction": {"dim": n_directions, "encoding": "one_hot"},
            "phase": {"dim": 1, "encoding": {"moving": 0, "hold": 1}},
            "excitation_direction": {"dim": 3, "description": "unit pull direction"},
        },
        "row_count": len(output_rows),
        "hold_count": len(hold_indices),
        "source_files": {
            "robot": _source_info(robot_path),
            "tracking": _source_info(tracking_path),
        },
        "source_metadata": {
            "robot": robot_metadata,
            "tracking": tracking_metadata,
        },
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
    parser.add_argument("--output", required=True, type=Path, help="Unified episode Parquet")
    parser.add_argument("--camera-frames", type=int, default=5, help="Camera frames per estimate")
    parser.add_argument(
        "--max-camera-delta",
        type=float,
        default=1.0,
        help="Maximum allowed camera-to-reference time difference in seconds",
    )
    parser.add_argument(
        "--fruiting-base-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Override fruiting base in the reference-tag frame",
    )
    args = parser.parse_args()
    output = compile_static_episode(
        args.robot,
        args.tracking,
        args.output,
        camera_frame_count=args.camera_frames,
        max_camera_delta_s=args.max_camera_delta,
        fruiting_base_pos=args.fruiting_base_pos,
        command_argv=sys.argv,
    )
    print(f"Wrote unified static system-ID episode to {output}")


if __name__ == "__main__":
    main()
