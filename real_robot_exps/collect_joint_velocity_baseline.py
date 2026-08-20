"""Collect an unloaded baseline by replaying an actual run's joint velocities.

The input is the raw robot Parquet from ``apple_pullto_static``.  Its
``joint_vel`` rows are replayed in order at the robot's 1 kHz control rate;
the baseline therefore follows the actual motion frame-for-frame instead of
reconstructing it from poses or hold indices.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from real_robot_exps.apple_pullto_static import save_robot_hold_parquet
from real_robot_exps.pro_robot_interface import FrankaInterface


def load_episode_metadata(path: Path) -> dict:
    schema_metadata = pq.read_schema(path).metadata or {}
    payload = schema_metadata.get(b"dataset_metadata")
    if payload is not None:
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid dataset metadata in {path}") from exc

    rows = pq.read_table(path).to_pylist()
    for row in rows:
        if str(row.get("row_kind", "data")) == "metadata":
            metadata_json = row.get("metadata_json")
            if metadata_json:
                try:
                    return json.loads(metadata_json)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid metadata_json in {path}") from exc
    return {}


def load_joint_velocity_frames(path: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows = [
        row for row in pq.read_table(path).to_pylist()
        if str(row.get("row_kind", "data")) != "metadata"
    ]
    if not rows:
        raise ValueError(f"Actual run contains no robot rows: {path}")
    missing = [idx for idx, row in enumerate(rows) if row.get("joint_vel") is None]
    if missing:
        raise ValueError(
            f"Actual run is missing joint_vel on {len(missing)} rows; "
            "collect a new run with 1000 Hz control_rate_hz."
        )
    velocities = np.asarray([row["joint_vel"] for row in rows], dtype=np.float64)
    if velocities.ndim != 2 or velocities.shape[1] != 7:
        raise ValueError(f"Expected joint_vel shape [n, 7], got {velocities.shape}")
    if not np.isfinite(velocities).all():
        raise ValueError("Actual run contains non-finite joint velocities")
    timestamps = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.shape[0] != velocities.shape[0]:
        raise ValueError("Actual run timestamps are missing or malformed")
    if not np.isfinite(timestamps).all():
        raise ValueError("Actual run contains non-finite timestamps")
    return rows, velocities, timestamps


def resample_joint_velocity_frames(
    rows: list[dict], velocities: np.ndarray, timestamps: np.ndarray, rate_hz: float
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Resample irregularly collected state velocities onto a fixed-rate grid."""
    if rate_hz <= 0.0 or not np.isfinite(rate_hz):
        raise ValueError(f"Invalid replay rate: {rate_hz}")
    source_t = timestamps - timestamps[0]
    keep = np.concatenate(([True], np.diff(source_t) > 0.0))
    source_t = source_t[keep]
    velocities = velocities[keep]
    rows = [row for row, valid in zip(rows, keep) if valid]
    if len(source_t) < 2:
        return rows, velocities, source_t

    dt = 1.0 / rate_hz
    replay_t = np.arange(0.0, source_t[-1] + 0.5 * dt, dt, dtype=np.float64)
    replay_v = np.column_stack([
        np.interp(replay_t, source_t, velocities[:, joint])
        for joint in range(velocities.shape[1])
    ])
    source_indices = np.searchsorted(source_t, replay_t, side="right").clip(1, len(rows)) - 1
    replay_rows = [rows[int(index)] for index in source_indices]
    return replay_rows, replay_v, replay_t


def load_start_pose(metadata: dict, actual_robot_path: Path) -> np.ndarray:
    pose = metadata.get("robot_start_pose_4x4")
    if pose is None:
        raise ValueError(
            f"Actual run metadata in {actual_robot_path} does not include robot_start_pose_4x4"
        )
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(
            f"Invalid robot_start_pose_4x4 in {actual_robot_path}: expected finite 4x4 matrix"
        )
    return pose


def _force_in_sim_frame(pose_flat: np.ndarray, force_base: np.ndarray) -> np.ndarray:
    pose_flat = np.asarray(pose_flat, dtype=np.float64)
    force_base = np.asarray(force_base, dtype=np.float64)
    # rotation = np.asarray([
    #     [pose_flat[0], pose_flat[4], pose_flat[8]],
    #     [pose_flat[1], pose_flat[5], pose_flat[9]],
    #     [pose_flat[2], pose_flat[6], pose_flat[10]],
    # ])
    # force_body = np.zeros(6, dtype=np.float64)
    # force_body[:3] = rotation.T @ force_base[:3]
    # force_body[3:] = rotation.T @ force_base[3:]
    return -force_base


def collect_baseline(
    actual_robot_path: Path | str,
    output_path: Path | str,
    config_path: Path | str,
    *,
    device: str = "cpu",
    metadata: dict | None = None,
) -> Path:
    actual_robot_path = Path(actual_robot_path)
    output_path = Path(output_path)
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    metadata = load_episode_metadata(actual_robot_path)
    actual_rows, source_velocities, source_timestamps = load_joint_velocity_frames(actual_robot_path)
    replay_rows, velocities, replay_timestamps = resample_joint_velocity_frames(
        actual_rows,
        source_velocities,
        source_timestamps,
        float(config["robot"].get("control_rate_hz", 1000.0)),
    )
    start_pose_4x4 = load_start_pose(metadata, actual_robot_path)

    robot = FrankaInterface(config, device=device)
    trajectory = None
    try:
        print(f"Resetting baseline run to recorded start pose from {actual_robot_path.name}...")
        robot.reset_to_start_pose(start_pose_4x4)
        replay_rate_hz = float(config["robot"].get("control_rate_hz", 1000.0))
        print(f"Replaying {len(velocities)} timestamp-interpolated joint-velocity frames at "
              f"{replay_rate_hz:g} Hz...")
        # Preload the trajectory into the comm process.  The robot-side 1 kHz
        # loop interpolates it independently; the Python process only samples
        # force/state and cannot stall the motion command stream.
        robot.start_joint_velocity_mode(velocities, replay_rate_hz)
        for idx in range(len(velocities)):
            robot.wait_for_policy_step()
    finally:
        try:
            robot.end_control()
        except Exception as exc:
            print(f"[BASELINE] Ignoring end_control failure after baseline replay: {exc}")
        trajectory = robot.get_last_trajectory()
        robot.shutdown()
    if trajectory is None or len(trajectory.get("timestamp", [])) == 0:
        raise RuntimeError("Joint velocity replay returned no robot-side frames")

    rows = []
    replay_count = len(replay_rows)
    replay_rate_hz = float(config["robot"].get("control_rate_hz", 1000.0))
    for index, timestamp in enumerate(trajectory["timestamp"]):
        replay_index = min(int(round(index * replay_rate_hz / 1000.0)), replay_count - 1)
        source = replay_rows[replay_index]
        pose = trajectory["O_T_EE"][index]
        ft_raw = _force_in_sim_frame(pose, trajectory["ft_wrist_raw"][index])
        ft_filtered = _force_in_sim_frame(pose, trajectory["ft_wrist_filtered"][index])
        phase_value = int(source.get("phase", 1))
        phase_name = str(source.get("phase_name", "joint_velocity_replay"))
        sample_label = str(source.get("sample_label", "baseline"))
        rows.append({
            "timestamp": float(timestamp),
            "hold_step_idx": int(source.get("hold_step_idx", replay_index)),
            "hold_index": int(source.get("hold_index", 0)),
            "phase": phase_value,
            "phase_name": phase_name,
            "sample_label": sample_label,
            "ft_wrist": ft_filtered.astype(np.float32),
            "ft_wrist_raw": ft_raw.astype(np.float32),
            "tau_J": trajectory["tau_J"][index],
            "tau_ext_hat_filtered": trajectory["tau_ext_hat_filtered"][index],
            "tau_J_d": trajectory["tau_J_d"][index],
            "joint_pos": trajectory["joint_pos"][index],
            "joint_vel": trajectory["joint_vel"][index],
            "joint_vel_cmd": trajectory["joint_vel_cmd"][index],
            "joint_vel_target": trajectory["joint_vel_target"][index],
            "tcp_pos": np.asarray(pose[12:15], dtype=np.float32),
            "tcp_pose_4x4": np.asarray(pose, dtype=np.float32),
        })

    baseline_metadata = {
        "collection_mode": "joint_velocity_replay_baseline",
        "baseline_source_robot_path": str(actual_robot_path.resolve()),
        "baseline_source_frame_count": int(len(source_velocities)),
        "baseline_source_timestamps": [float(value) for value in source_timestamps.tolist()],
        "baseline_replayed_frame_count": int(len(rows)),
        "baseline_rate_hz": float(config["robot"].get("control_rate_hz", 1000.0)),
        "baseline_replay_field": "joint_vel",
        "baseline_command_field": "joint_vel_cmd",
        "baseline_replay_semantics": (
            "timestamp-interpolated actual-run joint velocities on a fixed control-rate grid "
            "with comm-side jerk limiting"
        ),
        "baseline_start_pose_4x4": np.asarray(start_pose_4x4, dtype=np.float64).tolist(),
        **(metadata or {}),
    }
    save_robot_hold_parquet(rows, output_path, baseline_metadata)
    print(f"Wrote joint-velocity baseline to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-robot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("real_robot_exps/config.yaml"))
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    metadata = {}
    if args.metadata:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    from real_robot_exps.gripper_test import GripperClient
    gc = GripperClient()
    gc.send_request(True)
    collect_baseline(args.actual_robot, args.output, args.config, metadata=metadata)
    gc.send_request(False)


if __name__ == "__main__":
    main()
