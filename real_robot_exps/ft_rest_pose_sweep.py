"""Record force/torque at rest in a few nearby poses.

This script is meant to diagnose wrench zeroing, frame convention, and
pose-dependent offsets without introducing a pull motion. It follows the
same robot control path as ``apple_pullto_static`` as closely as possible:

- load the same gains
- use the same FrankaInterface
- move to a small set of nearby poses
- hold still at each pose
- record the same robot-side step rows into Parquet

Usage:
    python -m real_robot_exps.ft_rest_pose_sweep \
        --config real_robot_exps/config.yaml \
        --output ft_rest_pose_sweep.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from real_robot_exps.apple_pullto_static import (
    hold_and_record,
    load_gains_from_config,
)
from real_robot_exps.pro_robot_interface import FrankaInterface, make_ee_target_pose_from_matrix


def _save_robot_hold_parquet(rows, filename, metadata):
    table = pa.Table.from_pylist(rows)
    file_metadata = dict(metadata)
    file_metadata.setdefault("schema_name", "real_ft_rest_pose_sweep_robot_raw")
    file_metadata.setdefault("schema_version", "1.0.0")
    file_metadata.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
    file_metadata.setdefault("timestamp_clock", "Unix wall clock from time.time()")
    file_metadata.setdefault("timestamp_unit", "seconds")
    file_metadata.setdefault("host", socket.gethostname())
    file_metadata.setdefault("platform", platform.platform())
    file_metadata.setdefault("python_version", platform.python_version())
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[b"dataset_metadata"] = json.dumps(
        file_metadata, sort_keys=True, default=str
    ).encode("utf-8")
    table = table.replace_schema_metadata(schema_metadata)
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output)
    return output


def _resolve_pose_sets(config: dict[str, object], offset_m: float):
    robot_cfg = config.get("robot", {})
    if not isinstance(robot_cfg, dict):
        robot_cfg = {}

    home_pos = np.asarray(robot_cfg.get("ft_home_pos", [0.0, 0.85, 0.42]), dtype=np.float64)
    home_rot = np.asarray(robot_cfg.get("ft_home_rot", [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), dtype=np.float64)

    apple_pos = np.asarray(robot_cfg.get("ft_apple_pos", [0.0, 0.9262, 0.41]), dtype=np.float64)
    apple_rot = np.asarray(robot_cfg.get("ft_apple_rot", [[-0.994, -0.110, 0.0], [0.0, 0.0, 1.0], [-0.11, 0.991, 0.0]]), dtype=np.float64)

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    poses = [
        ("home", home_pos, home_rot),
        ("home_up", home_pos + offset_m * up, home_rot),
        ("home_forward", home_pos + offset_m * forward, home_rot),
        ("apple", apple_pos, apple_rot),
        ("apple_up", apple_pos + offset_m * up, apple_rot),
        ("apple_forward", apple_pos + offset_m * forward, apple_rot),
    ]
    return poses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="real_robot_exps/config.yaml", help="Real robot config path")
    parser.add_argument("--output", type=Path, default=Path("ft_rest_pose_sweep.parquet"), help="Output Parquet path")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--hold-seconds", type=float, default=3.0, help="Seconds to hold each pose")
    parser.add_argument("--offset-cm", type=float, default=2.0, help="Small offset from home/apple in cm")
    parser.add_argument("--kp", type=int, default=80, help="Position controller proportional gain")
    parser.add_argument("--debug", action="store_true", help="Print more diagnostics")
    parser.add_argument("--mock", action="store_true", help="Use the mock robot interface")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as stream:
        real_config = yaml.safe_load(stream)

    if args.mock:
        real_config.setdefault("robot", {})
        real_config["robot"]["use_mock"] = True

    gains = load_gains_from_config(real_config, args.device)
    gains = dict(gains)
    gains["task_prop_gains"] = torch.tensor([args.kp, args.kp, args.kp, 30, 30, 30], device=args.device, dtype=torch.float32)
    gains["task_deriv_gains"] = torch.tensor([1.75 * math.sqrt(v) for v in [args.kp, args.kp, args.kp, 30, 30, 30]], device=args.device, dtype=torch.float32)

    robot = FrankaInterface(real_config, device=args.device)
    robot_cfg = real_config.get("robot", {})
    if not isinstance(robot_cfg, dict):
        robot_cfg = {}

    home_pose = make_ee_target_pose_from_matrix(
        np.asarray(robot_cfg.get("ft_home_pos", [0.0, 0.85, 0.42]), dtype=np.float64),
        np.asarray(robot_cfg.get("ft_home_rot", [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), dtype=np.float64),
    )
    apple_pose = make_ee_target_pose_from_matrix(
        np.asarray(robot_cfg.get("ft_apple_pos", [0.0, 0.9262, 0.41]), dtype=np.float64),
        np.asarray(robot_cfg.get("ft_apple_rot", [[-0.994, -0.110, 0.0], [0.0, 0.0, 1.0], [-0.11, 0.991, 0.0]]), dtype=np.float64),
    )

    offset_m = float(args.offset_cm) / 100.0
    poses = _resolve_pose_sets(real_config, offset_m)
    episode_id = str(uuid4())
    collection_start_timestamp = time.time()

    print("Moving to home pose...")
    robot.reset_to_start_pose(home_pose)
    default_dof_pos = robot.get_state_snapshot().joint_pos.clone()

    robot_rows = []
    pose_summaries = []

    for idx, (label, pos, rot) in enumerate(poses):
        target_pose = make_ee_target_pose_from_matrix(pos, rot)
        print(f"Holding pose {idx + 1}/{len(poses)}: {label}")
        robot.reset_to_start_pose(target_pose)
        robot.start_torque_mode()
        snap = robot.get_state_snapshot()
        start_timestamp = time.time()

        data = hold_and_record(
            robot,
            gains,
            snap.ee_pos.clone(),
            snap.ee_quat.clone(),
            default_dof_pos,
            duration_sec=float(args.hold_seconds),
            device=args.device,
            record_rows=robot_rows,
            hold_number=idx,
            n_holds=len(poses),
            direction_idx=0,
            n_directions=1,
            excitation_direction=np.zeros(3, dtype=np.float32),
            amplitude_m=0.0,
        )
        hold_mean = np.mean(data, axis=0).tolist() if len(data) else [float("nan")] * 6
        hold_std = np.std(data, axis=0).tolist() if len(data) else [float("nan")] * 6
        hold_rows = [row for row in robot_rows if row["hold_index"] == idx]
        torque_means = {
            field: (
                np.mean(np.asarray([row[field] for row in hold_rows], dtype=np.float64), axis=0).tolist()
                if hold_rows else [float("nan")] * 7
            )
            for field in ("tau_J", "tau_ext_hat_filtered", "tau_J_d")
        }
        robot.end_control()
        pose_summaries.append({
            "hold_index": idx,
            "label": label,
            "target_pos": np.asarray(pos, dtype=np.float64).tolist(),
            "target_rot": np.asarray(rot, dtype=np.float64).tolist(),
            "start_timestamp": start_timestamp,
            "n_samples": int(len(data)),
            "wrench_mean": hold_mean,
            "wrench_std": hold_std,
            "tau_J_mean": torque_means["tau_J"],
            "tau_ext_hat_filtered_mean": torque_means["tau_ext_hat_filtered"],
            "tau_J_d_mean": torque_means["tau_J_d"],
        })

    robot.reset_to_start_pose(home_pose)

    robot_metadata = {
        "episode_id": episode_id,
        "collection_start_timestamp": collection_start_timestamp,
        "collection_end_timestamp": time.time(),
        "collection_mode": "ft_rest_pose_sweep",
        "rest_pose_count": len(poses),
        "rest_hold_duration_s": float(args.hold_seconds),
        "rest_pose_offset_m": offset_m,
        "rest_pose_offset_cm": float(args.offset_cm),
        "pose_summaries": pose_summaries,
        "home_pose_4x4": np.asarray(home_pose).tolist(),
        "apple_pose_4x4": np.asarray(apple_pose).tolist(),
        "position_unit": "m",
        "force_unit": "N",
        "torque_unit": "N*m",
        "joint_torque_fields": {
            "order": [f"joint_{i}" for i in range(1, 8)],
            "order_direction": "base-to-end-effector",
            "tau_J": "measured link-side joint torque sensor signals",
            "tau_ext_hat_filtered": "low-pass filtered external torque estimate; excludes configured EE/load and robot dynamics",
            "tau_J_d": "desired link-side joint torques without gravity",
        },
        "note": "Rest-pose wrench sweep with no pull motion; intended for zero/bias debugging.",
    }
    saved_path = _save_robot_hold_parquet(robot_rows, args.output, robot_metadata)
    print(f"Wrote rest pose sweep data to {saved_path}")
    robot.shutdown()


if __name__ == "__main__":
    main()
