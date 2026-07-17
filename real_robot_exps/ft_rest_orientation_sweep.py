"""Record seven unloaded, stationary orientations at the apple pose.

The TCP position remains fixed at the configured apple position. The seven
recorded stops are the reference orientation followed by +/- ``--angle-deg``
about each local EE axis.

Usage:
    python -m real_robot_exps.ft_rest_orientation_sweep \
        --config real_robot_exps/config.yaml \
        --output ft_rest_orientation_sweep.parquet
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml

from real_robot_exps.apple_pullto_static import hold_and_record, load_gains_from_config
from real_robot_exps.ft_rest_pose_sweep import _save_robot_hold_parquet
from real_robot_exps.pro_robot_interface import FrankaInterface, make_ee_target_pose_from_matrix


def _axis_rotation(axis: str, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"Unknown rotation axis: {axis}")


def _reference_poses(config: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    robot_cfg = config.get("robot", {})
    if not isinstance(robot_cfg, dict):
        robot_cfg = {}
    return [
        (
            "home",
            np.asarray(robot_cfg.get("ft_home_pos", [0.0, 0.85, 0.42]), dtype=np.float64),
            np.asarray(robot_cfg.get("ft_home_rot", [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), dtype=np.float64),
        ),
        (
            "apple",
            np.asarray(robot_cfg.get("ft_apple_pos", [0.0, 0.9262, 0.41]), dtype=np.float64),
            np.asarray(robot_cfg.get("ft_apple_rot", [[-0.994, -0.110, 0.0], [0.0, 0.0, 1.0], [-0.11, 0.991, 0.0]]), dtype=np.float64),
        ),
    ]


def _orientation_targets(config: dict, angle_deg: float) -> list[dict]:
    angle_rad = math.radians(float(angle_deg))
    _, position, reference_rotation = next(
        pose for pose in _reference_poses(config) if pose[0] == "apple"
    )
    targets = [{
        "label": "apple_reference",
        "reference_name": "apple",
        "position": position,
        "reference_rotation": reference_rotation,
        "target_rotation": reference_rotation,
        "local_axis": "none",
        "angle_deg": 0.0,
    }]
    for axis in ("x", "y", "z"):
        for sign, sign_name in ((1.0, "plus"), (-1.0, "minus")):
            signed_angle = sign * angle_rad
            targets.append({
                "label": f"apple_{axis}_{sign_name}_{angle_deg:g}deg",
                "reference_name": "apple",
                "position": position,
                "reference_rotation": reference_rotation,
                # Post-multiplication applies the perturbation in the local EE frame.
                "target_rotation": reference_rotation @ _axis_rotation(axis, signed_angle),
                "local_axis": axis,
                "angle_deg": math.degrees(signed_angle),
            })
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="real_robot_exps/config.yaml")
    parser.add_argument("--output", type=Path, default=Path("ft_rest_orientation_sweep.parquet"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--angle-deg", type=float, default=45.0)
    parser.add_argument("--kp", type=int, default=80)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.angle_deg <= 45.0:
        raise ValueError("--angle-deg must be greater than 0 and no more than 45 degrees")
    if args.hold_seconds <= 0.0:
        raise ValueError("--hold-seconds must be positive")

    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.mock:
        config.setdefault("robot", {})["use_mock"] = True

    gains = dict(load_gains_from_config(config, args.device))
    proportional = [args.kp, args.kp, args.kp, 30, 30, 30]
    gains["task_prop_gains"] = torch.tensor(proportional, device=args.device, dtype=torch.float32)
    gains["task_deriv_gains"] = torch.tensor(
        [1.75 * math.sqrt(value) for value in proportional],
        device=args.device,
        dtype=torch.float32,
    )

    targets = _orientation_targets(config, args.angle_deg)
    references = {name: (pos, rot) for name, pos, rot in _reference_poses(config)}
    home_pos, home_rot = references["home"]
    home_pose = make_ee_target_pose_from_matrix(home_pos, home_rot)

    robot = FrankaInterface(config, device=args.device)
    print("Moving to home reference pose...")
    robot.reset_to_start_pose(home_pose)
    default_dof_pos = robot.get_state_snapshot().joint_pos.clone()

    episode_id = str(uuid4())
    collection_start_timestamp = time.time()
    rows: list[dict] = []
    summaries: list[dict] = []

    try:
        for hold_index, target in enumerate(targets):
            target_pose = make_ee_target_pose_from_matrix(
                target["position"], target["target_rotation"]
            )

            print(f"Holding orientation {hold_index + 1}/{len(targets)}: {target['label']}")
            robot.reset_to_start_pose(target_pose)
            robot.start_torque_mode()
            snapshot = robot.get_state_snapshot()
            start_timestamp = time.time()

            wrench = hold_and_record(
                robot,
                gains,
                snapshot.ee_pos.clone(),
                snapshot.ee_quat.clone(),
                default_dof_pos,
                duration_sec=args.hold_seconds,
                device=args.device,
                record_rows=rows,
                hold_number=hold_index,
                n_holds=len(targets),
                direction_idx=0,
                n_directions=1,
                excitation_direction=np.zeros(3, dtype=np.float32),
                amplitude_m=0.0,
            )
            robot.end_control()

            hold_rows = [row for row in rows if row["hold_index"] == hold_index]
            mean = lambda field: np.mean(
                np.asarray([row[field] for row in hold_rows], dtype=np.float64), axis=0
            ).tolist()
            summaries.append({
                "hold_index": hold_index,
                "label": target["label"],
                "reference_name": target["reference_name"],
                "local_rotation_axis": target["local_axis"],
                "rotation_angle_deg": target["angle_deg"],
                "target_pos": target["position"].tolist(),
                "reference_rot": target["reference_rotation"].tolist(),
                "target_rot": target["target_rotation"].tolist(),
                "start_timestamp": start_timestamp,
                "n_samples": len(hold_rows),
                "wrench_mean": np.mean(wrench, axis=0).tolist(),
                "wrench_std": np.std(wrench, axis=0).tolist(),
                "tau_J_mean": mean("tau_J"),
                "tau_ext_hat_filtered_mean": mean("tau_ext_hat_filtered"),
                "tau_J_d_mean": mean("tau_J_d"),
                "gravity_torques_mean": mean("gravity_torques"),
            })

        robot.reset_to_start_pose(home_pose)
        metadata = {
            "schema_name": "real_ft_rest_orientation_sweep_robot_raw",
            "schema_version": "1.0.0",
            "episode_id": episode_id,
            "collection_mode": "ft_rest_orientation_sweep",
            "collection_start_timestamp": collection_start_timestamp,
            "collection_end_timestamp": time.time(),
            "hold_duration_s": args.hold_seconds,
            "rotation_angle_deg": args.angle_deg,
            "rotation_frame": "local EE frame",
            "rotation_composition": "R_target = R_reference @ R_local_delta",
            "approach_strategy": "move directly through the seven ordered apple-pose orientations",
            "orientation_count": len(targets),
            "pose_summaries": summaries,
            "position_unit": "m",
            "angle_unit": "degrees in metadata; radians in rotation matrices",
            "force_unit": "N",
            "torque_unit": "N*m",
            "note": "Seven-stop unloaded stationary orientation sweep at the fixed apple TCP position.",
        }
        output = _save_robot_hold_parquet(rows, args.output, metadata)
        print(f"Wrote orientation sweep data to {output}")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
