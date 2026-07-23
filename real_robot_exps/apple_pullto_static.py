"""Quasi-static apple-pull system-ID collection.

Records robot state during fixed holds and optionally applies a matched
unloaded dynamic wrench baseline. Camera detection is intentionally kept in a
separate process; an existing tracking Parquet can be compiled afterward.

Usage:
    python -m real_robot_exps.apple_pullto_static --mode collect
"""

import argparse
import hashlib
import json
import math
import platform
import socket
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
warnings.filterwarnings("ignore")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from real_robot_exps.pro_robot_interface import FrankaInterface, make_ee_target_pose, make_ee_target_pose_from_matrix
from real_robot_exps.hybrid_controller import (
    ControlTargets, get_euler_xyz, compute_pose_error, compute_pose_task_wrench,
)


CONVERGE_THRESHOLD = 1e-4  # 0.1mm
CONVERGE_FRAMES = 10
MAX_STEPS = 500             # ~33s at 15Hz safety cap
MOVE_DISTANCE = 0.02     # 5cm

# One-line switch for unloaded/system-identification trials closer to the robot.
# False uses the normal apple pose; True uses CLOSE_PULL_START_POSITION_M.
USE_CLOSE_PULL_START_POSE = False
CLOSE_PULL_START_POSITION_M = np.array([0.0, 0.7, 0.35], dtype=np.float64)
CLOSE_PULL_ROLL_FORWARD_DEG = 20.0

# Baseline mode records an unloaded wrench profile. When this is True, collect
# mode subtracts the matching profile point-by-point within each static hold.
USE_DYNAMIC_BASELINE_CORRECTION = True


def _quat_wxyz_to_rotmat(quat: torch.Tensor) -> np.ndarray:
    """Convert a wxyz quaternion tensor to a 3x3 rotation matrix."""
    q = np.asarray(quat.detach().cpu().numpy(), dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _tcp_pose_4x4_from_snapshot(snap) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _quat_wxyz_to_rotmat(snap.ee_quat)
    pose[:3, 3] = snap.ee_pos.detach().cpu().numpy().astype(np.float64, copy=False)
    return pose


def _flat_float32(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _append_robot_sample(
    record_rows,
    *,
    timestamp: float,
    hold_step_idx: int,
    hold_index: int,
    phase: int,
    phase_name: str,
    sample_label: str,
    amplitude_m: float,
    hold_one_hot: np.ndarray,
    direction_one_hot: np.ndarray,
    excitation_direction: np.ndarray,
    snap,
    action: np.ndarray | None = None,
):
    if record_rows is None:
        return
    record_rows.append({
        "timestamp": float(timestamp),
        "hold_step_idx": int(hold_step_idx),
        "hold_index": int(hold_index),
        "hold_number": _flat_float32(hold_one_hot.copy()),
        "direction_index": int(np.argmax(direction_one_hot)) if direction_one_hot.size else 0,
        "direction": _flat_float32(direction_one_hot.copy()),
        "phase": int(phase),
        "phase_name": str(phase_name),
        "sample_label": str(sample_label),
        "amplitude_m": float(amplitude_m),
        "ft_wrist": _flat_float32(snap.force_torque.cpu().numpy()),
        "ft_wrist_raw": _flat_float32(snap.force_torque.cpu().numpy()),
        "tau_J_d": _flat_float32(snap.tau_J_d.cpu().numpy()),
        "joint_pos": _flat_float32(snap.joint_pos.cpu().numpy()),
        "tcp_velocity": _flat_float32(np.concatenate([
            snap.ee_linvel.cpu().numpy(),
            snap.ee_angvel.cpu().numpy(),
        ])),
        "action": _flat_float32(np.zeros(6, dtype=np.float32) if action is None else np.asarray(action, dtype=np.float32).reshape(6)),
        "tcp_pos": _flat_float32(snap.ee_pos.cpu().numpy()),
        "tcp_pose_4x4": _flat_float32(_tcp_pose_4x4_from_snapshot(snap)),
        "excitation_direction": _flat_float32(excitation_direction.copy()),
    })


def load_gains_from_config(real_config: dict, device: str = "cpu") -> dict:
    """Load controller gains from real robot config control_gains section.

    Returns dict with all tensors needed to build ControlTargets.
    """
    gains_cfg = real_config.get('control_gains', {})

    if 'task_prop_gains' not in gains_cfg:
        raise RuntimeError("control_gains.task_prop_gains not found in config")
    if 'task_deriv_gains' not in gains_cfg:
        raise RuntimeError("control_gains.task_deriv_gains not found in config")

    task_prop_gains = torch.tensor(gains_cfg['task_prop_gains'], device=device, dtype=torch.float32)
    task_deriv_gains = torch.tensor(gains_cfg['task_deriv_gains'], device=device, dtype=torch.float32)
    kp_null = gains_cfg.get('kp_null', 0.0)
    kd_null = gains_cfg.get('kd_null', 0.0)

    if gains_cfg.get('singularity_damping_enabled', False):
        singularity_damping = gains_cfg.get('singularity_damping_lambda', 0.01)
    else:
        singularity_damping = 0.0

    partial_inertia_decoupling = gains_cfg.get('partial_inertia_decoupling', False)
    sep_ori = gains_cfg.get('sep_ori', False)
    pose_ki = torch.tensor(gains_cfg.get('pose_ki',  [2.0, 2.0, 2.0, 0.0, 0.0, 0.0]), device=device, dtype=torch.float32)
    pose_integral_clamp = gains_cfg.get('pose_integral_clamp', 50.0)
    pose_integral_reset_on_target = gains_cfg.get('pose_integral_reset_on_target', True)

    # Mutual exclusion check
    if partial_inertia_decoupling and sep_ori:
        raise RuntimeError(
            "partial_inertia_decoupling and sep_ori are mutually exclusive — "
            "set only one to true in control_gains"
        )

    # print(f"  task_prop_gains:  {task_prop_gains.tolist()}")
    # print(f"  task_deriv_gains: {task_deriv_gains.tolist()}")
    # print(f"  kp_null: {kp_null}, kd_null: {kd_null}")
    # print(f"  singularity_damping: {singularity_damping}")
    # print(f"  partial_inertia_decoupling: {partial_inertia_decoupling}")
    # print(f"  sep_ori: {sep_ori}")

    return {
        'task_prop_gains': task_prop_gains,
        'task_deriv_gains': task_deriv_gains,
        'kp_null': kp_null,
        'kd_null': kd_null,
        'singularity_damping': singularity_damping,
        'partial_inertia_decoupling': partial_inertia_decoupling,
        'sep_ori': sep_ori,
        'pose_ki': pose_ki,
        'pose_integral_clamp': pose_integral_clamp,
        'pose_integral_reset_on_target': pose_integral_reset_on_target
    }


def build_position_targets(
    gains: dict,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    default_dof_pos: torch.Tensor,
    device: str = "cpu",
) -> ControlTargets:
    """Build ControlTargets for pure position control to a fixed target.

    Sets goal_position = target_pos so bounds constraint doesn't interfere.
    """
    # pos_bounds set large enough to never clamp
    pos_bounds = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)

    return ControlTargets(
        target_pos=target_pos,
        target_quat=target_quat,
        target_force=torch.zeros(6, device=device),
        sel_matrix=torch.zeros(6, device=device),
        task_prop_gains=gains['task_prop_gains'],
        task_deriv_gains=gains['task_deriv_gains'],
        force_kp=torch.zeros(6, device=device),
        force_di_wrench=torch.zeros(6, device=device),
        default_dof_pos=default_dof_pos,
        kp_null=gains['kp_null'],
        kd_null=gains['kd_null'],
        pos_bounds=pos_bounds,
        goal_position=target_pos,
        ctrl_mode="force_only",
        singularity_damping=gains['singularity_damping'],
        partial_inertia_decoupling=gains['partial_inertia_decoupling'],
        sep_ori=gains['sep_ori'],
        pose_ki=gains['pose_ki'],
        pose_integral_clamp=gains['pose_integral_clamp'],
        pose_integral_reset_on_target=gains['pose_integral_reset_on_target']
    )


def _quat_to_rpy_deg(quat: torch.Tensor) -> list:
    """Convert quaternion (w,x,y,z) to [roll, pitch, yaw] in degrees."""
    roll, pitch, yaw = get_euler_xyz(quat)
    return [math.degrees(roll.item()), math.degrees(pitch.item()), math.degrees(yaw.item())]


def run_move(
    robot: FrankaInterface,
    gains: dict,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    default_dof_pos: torch.Tensor,
    label: str,
    device: str = "cpu",
    prnt: bool = True,
    manage_control: bool = True,
    *,
    record_rows=None,
    hold_index: int = -1,
    hold_number: int = -1,
    n_holds: int = 1,
    direction_idx: int = 0,
    n_directions: int = 1,
    excitation_direction=None,
    amplitude_m: float = 0.0,
    sample_label: str = "move",
) -> dict:
    """Run torque control until robot converges to target_pos.

    Returns dict with position/orientation start, target, achieved, errors, and steps.
    """
    targets = build_position_targets(gains, target_pos, target_quat, default_dof_pos, device)
    excitation_direction = np.asarray(
        excitation_direction if excitation_direction is not None else np.zeros(3),
        dtype=np.float32,
    ).reshape(3)
    hold_one_hot = np.zeros(int(n_holds), dtype=np.float32)
    if 0 <= int(hold_number) < int(n_holds):
        hold_one_hot[int(hold_number)] = 1.0
    direction_one_hot = np.zeros(int(n_directions), dtype=np.float32)
    if 0 <= int(direction_idx) < int(n_directions):
        direction_one_hot[int(direction_idx)] = 1.0

    if manage_control:
        robot.start_torque_mode()

    snap = robot.get_state_snapshot()
    start_pos = snap.ee_pos.clone()
    start_quat = snap.ee_quat.clone()
    prev_pos = snap.ee_pos.clone()
    converge_count = 0

    for step in range(MAX_STEPS):
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        timestamp = time.time()
        robot.check_safety(snap)

        robot.set_control_targets(targets)
        _append_robot_sample(
            record_rows,
            timestamp=timestamp,
            hold_step_idx=step,
            hold_index=hold_index,
            phase=0,
            phase_name="move",
            amplitude_m=amplitude_m,
            hold_one_hot=hold_one_hot,
            direction_one_hot=direction_one_hot,
            excitation_direction=excitation_direction,
            snap=snap,
            sample_label=sample_label,
        )

        # Debug: replicate wrench computation from compute process for visibility
        if step == 0 or step % 50 == 0 or converge_count == CONVERGE_FRAMES - 1:
            pos_err, aa_err = compute_pose_error(
                snap.ee_pos, snap.ee_quat, target_pos, target_quat,
            )
            wrench = compute_pose_task_wrench(
                snap.ee_pos, snap.ee_quat, snap.ee_linvel, snap.ee_angvel,
                target_pos, target_quat,
                gains['task_prop_gains'], gains['task_deriv_gains'],
            )
            if(prnt):
                print(f"    [step {step:3d}] orn_error (axis-angle, base frame): "
                    f"[{aa_err[0].item():.6f}, {aa_err[1].item():.6f}, {aa_err[2].item():.6f}]")
                print(f"    [step {step:3d}] wrench [Fx,Fy,Fz,Tx,Ty,Tz] (base frame, pre-Lambda): "
                    f"[{wrench[0].item():.4f}, {wrench[1].item():.4f}, {wrench[2].item():.4f}, "
                    f"{wrench[3].item():.4f}, {wrench[4].item():.4f}, {wrench[5].item():.4f}]")
                print(f"    [step {step:3d}] frame: geometric Jacobian base frame "
                    f"(q_err = q_target * q_current^-1 -> axis-angle)")

        pos_delta = torch.norm(snap.ee_pos - prev_pos).item()
        prev_pos = snap.ee_pos.clone()

        if pos_delta < CONVERGE_THRESHOLD:
            converge_count += 1
        else:
            converge_count = 0

        if converge_count >= CONVERGE_FRAMES:
            break
    if manage_control:
        robot.end_control()

    # Position results
    achieved_pos = snap.ee_pos.clone()
    pos_error = (achieved_pos - target_pos).tolist()
    pos_error_norm = torch.norm(achieved_pos - target_pos).item()

    # Orientation results
    achieved_quat = snap.ee_quat.clone()
    target_rpy_deg = _quat_to_rpy_deg(target_quat)
    start_rpy_deg = _quat_to_rpy_deg(start_quat)
    achieved_rpy_deg = _quat_to_rpy_deg(achieved_quat)
    orn_error_deg = [(achieved_rpy_deg[i] - target_rpy_deg[i] + 180.0) % 360.0 - 180.0 for i in range(3)]

    steps_used = step + 1
    converged = converge_count >= CONVERGE_FRAMES

    if(prnt):
        print(f"  [{label}]")
        print(f"    Start Pos:      [{start_pos[0].item():.5f}, {start_pos[1].item():.5f}, {start_pos[2].item():.5f}]")
        print(f"    Target Pos:     [{target_pos[0].item():.5f}, {target_pos[1].item():.5f}, {target_pos[2].item():.5f}]")
        print(f"    Achieved Pos:   [{achieved_pos[0].item():.5f}, {achieved_pos[1].item():.5f}, {achieved_pos[2].item():.5f}]")
        print(f"    Pos Error:      [{pos_error[0]:.5f}, {pos_error[1]:.5f}, {pos_error[2]:.5f}] (norm={pos_error_norm*1000:.2f}mm)")
        print(f"    Orn Target  (RPY deg): [{target_rpy_deg[0]:.2f}, {target_rpy_deg[1]:.2f}, {target_rpy_deg[2]:.2f}]")
        print(f"    Orn Start   (RPY deg): [{start_rpy_deg[0]:.2f}, {start_rpy_deg[1]:.2f}, {start_rpy_deg[2]:.2f}]")
        print(f"    Orn Achieved(RPY deg): [{achieved_rpy_deg[0]:.2f}, {achieved_rpy_deg[1]:.2f}, {achieved_rpy_deg[2]:.2f}]")
        print(f"    Orn Error   (RPY deg): [{orn_error_deg[0]:.2f}, {orn_error_deg[1]:.2f}, {orn_error_deg[2]:.2f}]")
        print(f"    Steps:    {steps_used} ({'converged' if converged else 'MAX STEPS'})")

    return {
        'label': label,
        'start_pos': start_pos,
        'target_pos': target_pos,
        'achieved_pos': achieved_pos,
        'pos_error': pos_error,
        'pos_error_norm': pos_error_norm,
        'target_rpy_deg': target_rpy_deg,
        'start_rpy_deg': start_rpy_deg,
        'achieved_rpy_deg': achieved_rpy_deg,
        'orn_error_deg': orn_error_deg,
        'steps': steps_used,
        'converged': converged,
    }

import pandas as pd
import matplotlib.pyplot as plt

def hold_and_record(
    robot: FrankaInterface,
    gains,
    target_pos,
    target_quat,
    default_dof_pos,
    duration_sec,
    device="cpu",
    *,
    record_rows=None,
    hold_index=-1,
    hold_number=-1,
    n_holds=1,
    direction_idx=0,
    n_directions=1,
    excitation_direction=None,
    amplitude_m=0.0,
    phase: int = 1,
    phase_name: str = "hold",
    sample_label: str = "hold",
):
    """Hold a pose and optionally append complete system-ID robot rows.

    The ndarray return value is retained for the F/T calibration caller.
    """
    targets = build_position_targets(gains, target_pos, target_quat, default_dof_pos, device)
    steps = int(duration_sec * robot._control_rate_hz)
    ft_history = []
    excitation_direction = np.asarray(
        excitation_direction if excitation_direction is not None else np.zeros(3),
        dtype=np.float32,
    ).reshape(3)
    hold_one_hot = np.zeros(int(n_holds), dtype=np.float32)
    if 0 <= int(hold_number) < int(n_holds):
        hold_one_hot[int(hold_number)] = 1.0
    direction_one_hot = np.zeros(int(n_directions), dtype=np.float32)
    if 0 <= int(direction_idx) < int(n_directions):
        direction_one_hot[int(direction_idx)] = 1.0
    
    for hold_step_idx in range(steps):
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        timestamp = time.time()
        robot.check_safety(snap)
        robot.set_control_targets(targets)
        ft = snap.force_torque.cpu().numpy()
        ft_history.append(ft)
        _append_robot_sample(
            record_rows,
            timestamp=timestamp,
            hold_step_idx=hold_step_idx,
            hold_index=hold_index,
            phase=phase,
            phase_name=phase_name,
            sample_label=sample_label,
            amplitude_m=amplitude_m,
            hold_one_hot=hold_one_hot,
            direction_one_hot=direction_one_hot,
            excitation_direction=excitation_direction,
            snap=snap,
        )
    
    # print(ft_history)

        
    return np.array(ft_history)


def save_robot_hold_parquet(rows, filename, metadata):
    """Save raw robot-side hold rows and rich collection metadata."""
    table = pa.Table.from_pylist(rows)
    file_metadata = dict(metadata)
    file_metadata.setdefault("schema_name", "real_static_sysid_robot_raw")
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


def _read_parquet_metadata(path: Path) -> dict:
    payload = (pq.read_schema(path).metadata or {}).get(b"dataset_metadata")
    return json.loads(payload.decode("utf-8")) if payload else {}


def _validate_baseline_compatibility(current: dict, baseline: dict, baseline_path: Path) -> None:
    comparisons = (
        ("theta_rad", 1e-9),
        ("phi_rad", 1e-9),
        ("distance_m", 1e-9),
        ("n_holds", 0.0),
        ("pull_start_pose_name", None),
    )
    mismatches = []
    for key, tolerance in comparisons:
        if key not in current or key not in baseline:
            mismatches.append(f"{key}=missing")
            continue
        if tolerance is None:
            matches = current[key] == baseline[key]
        elif tolerance == 0.0:
            matches = int(current[key]) == int(baseline[key])
        else:
            matches = abs(float(current[key]) - float(baseline[key])) <= tolerance
        if not matches:
            mismatches.append(f"{key}: collect={current[key]!r}, baseline={baseline[key]!r}")

    current_pose = np.asarray(current.get("robot_start_pose_4x4", []), dtype=np.float64)
    baseline_pose = np.asarray(baseline.get("robot_start_pose_4x4", []), dtype=np.float64)
    if current_pose.shape != (4, 4) or baseline_pose.shape != (4, 4) or not np.allclose(
        current_pose, baseline_pose, atol=1e-7, rtol=0.0
    ):
        mismatches.append("robot_start_pose_4x4 differs")
    if mismatches:
        raise ValueError(
            f"Baseline {baseline_path} is incompatible with this collect run: "
            + "; ".join(mismatches)
        )


def apply_dynamic_baseline(robot_rows: list[dict], baseline_path: Path) -> dict:
    """Subtract an unloaded baseline profile within each corresponding hold."""
    baseline_table = pq.read_table(baseline_path)
    baseline_rows = baseline_table.to_pylist()
    if not baseline_rows:
        raise ValueError(f"Baseline file has no rows: {baseline_path}")

    baseline_by_hold = {}
    for row in baseline_rows:
        hold_index = int(row["hold_index"])
        if hold_index >= 0:
            baseline_by_hold.setdefault(hold_index, []).append(row)
    current_hold_indices = sorted({int(row["hold_index"]) for row in robot_rows if int(row["hold_index"]) >= 0})
    if current_hold_indices != sorted(baseline_by_hold):
        raise ValueError(
            f"Baseline hold indices {sorted(baseline_by_hold)} do not match "
            f"collect hold indices {current_hold_indices}"
        )

    for hold_index in current_hold_indices:
        current_hold = [row for row in robot_rows if int(row["hold_index"]) == hold_index]
        baseline_hold = sorted(
            baseline_by_hold[hold_index], key=lambda row: int(row.get("hold_step_idx", 0))
        )
        baseline_ft = np.asarray(
            [row.get("ft_wrist_raw", row["ft_wrist"]) for row in baseline_hold],
            dtype=np.float64,
        )
        baseline_progress = np.linspace(0.0, 1.0, len(baseline_ft))
        current_progress = np.linspace(0.0, 1.0, len(current_hold))
        interpolated = np.column_stack([
            np.interp(current_progress, baseline_progress, baseline_ft[:, component])
            for component in range(6)
        ])
        for row, dynamic_bias in zip(current_hold, interpolated):
            raw = np.asarray(row.get("ft_wrist_raw", row["ft_wrist"]), dtype=np.float64)
            row["ft_wrist_raw"] = raw.astype(np.float32)
            row["ft_wrist_baseline"] = dynamic_bias.astype(np.float32)
            row["ft_wrist"] = (raw - dynamic_bias).astype(np.float32)

    return {
        "method": "per-hold normalized-time linear interpolation",
        "source_path": str(baseline_path.resolve()),
        "source_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "source_row_count": len(baseline_rows),
        "corrected_field": "ft_wrist",
        "raw_field": "ft_wrist_raw",
        "bias_field": "ft_wrist_baseline",
    }

def plot_and_save_data(raw_ft_data, label="pull", window_size=5, baseline=False, plot=True, metadata=""):
    """Saves raw/smooth CSVs and plots the Fx, Fy, Fz forces."""
    # Create DataFrame
    cols = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    df_raw = pd.DataFrame(raw_ft_data, columns=cols)
   
    # Save to CSV
    name = f"{label}.csv"
    df_raw.to_csv(name, index=False)
    with open(name, "a") as f:
        f.write(f"# {metadata}")
    
    
    # Plot forces
    if(plot):
        plt.figure(figsize=(10, 5))
        for axis, color in zip(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"], ['r', 'g', 'b', 'yellow', 'teal', 'purple']):
            plt.plot(df_raw[axis], color=color, alpha=1.0, label=f"Raw {axis}")
            
        plt.title(f"Force/Torque Profile: {label}")
        plt.xlabel("Policy Steps (15Hz)")
        plt.ylabel("Force (N) / Torque (Nm)")
        plt.legend()
        plt.grid(True)
        plt.show()

def hold_position(
    robot: FrankaInterface,
    gains: dict,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    default_dof_pos: torch.Tensor,
    duration_sec: float,
    device: str = "cpu",
):
    """Actively hold a Cartesian pose while maintaining the 15Hz safety/timing loop."""
    targets = build_position_targets(gains, target_pos, target_quat, default_dof_pos, device)
    
    # 15Hz * duration = number of steps
    steps = int(duration_sec * robot._control_rate_hz)
    
    for _ in range(steps):
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        robot.check_safety(snap)
        robot.set_control_targets(targets)

def update_gains(gains, new_prop_gains, device):

    gains["task_prop_gains"] = torch.tensor(new_prop_gains, device=device, dtype=torch.float32)
    derivs = [0, 0, 0, 0, 0, 0]
    for i in range(len(new_prop_gains)):
        derivs[i] =  1.75 * math.sqrt(new_prop_gains[i]) # 1.75 was working best
    gains["task_deriv_gains"] = torch.tensor(derivs, device=device, dtype=torch.float32)
    return gains

def pull_test(theta, phi, robot: FrankaInterface, pull_start_pose_4x4, default_dof_pos, gains, home_pose_4x4, gc, device: str = "cpu", baseline: bool = False, debug: bool = False, to_plot: bool = False, distance: float = 0.05, stops: int = 5, args=None, config_snapshot=None, ee_config=None, ft_calibration_enabled: bool = False):
    collection_start_timestamp = time.time()
    episode_id = str(uuid4())
    run_args = dict(args or {})
    base_label = f"pull_theta{theta:.2f}_phi{phi:.2f}"
    label = f"{base_label}_{'baseline' if baseline else 'raw'}"
    time.sleep(2.0) # let it settle
    robot.reset_to_start_pose(pull_start_pose_4x4)
    snap = robot.get_state_snapshot()
    robot.start_torque_mode()
    robot_rows = []
    rest_reference_timestamp = time.time()
    # Short initial rest geometry sample: the apple is untouched and the arm is
    # at the pull start pose. This is the frame-0 / datapoint-0 geometry anchor.
    hold_and_record(
        robot,
        gains,
        snap.ee_pos.clone(),
        snap.ee_quat.clone(),
        default_dof_pos,
        duration_sec=0.5,
        device=device,
        record_rows=robot_rows,
        hold_index=-2,
        hold_number=0,
        n_holds=stops,
        direction_idx=int(run_args.get("direction_index", 0)),
        n_directions=int(run_args.get("num_directions", 1)),
        excitation_direction=np.zeros(3, dtype=np.float32),
        amplitude_m=0.0,
        phase=0,
        phase_name="rest_geometry",
        sample_label="rest_geometry",
    )
    gc.send_request(True)
    time.sleep(3.0)
    # Post-grab initial hold: the apple is grasped but the arm has not started
    # the pull yet. This should be the first “held” sample of the episode.
    snap = robot.get_state_snapshot()
    hold_and_record(
        robot,
        gains,
        snap.ee_pos.clone(),
        snap.ee_quat.clone(),
        default_dof_pos,
        duration_sec=1.0,
        device=device,
        record_rows=robot_rows,
        hold_index=-1,
        hold_number=0,
        n_holds=stops,
        direction_idx=int(run_args.get("direction_index", 0)),
        n_directions=int(run_args.get("num_directions", 1)),
        excitation_direction=np.zeros(3, dtype=np.float32),
        amplitude_m=0.0,
        phase=1,
        phase_name="post_grab_hold",
        sample_label="post_grab_hold",
    )
    
    # distance = .05
    # stops = 5
    steps = stops
    if(distance/steps > 0.01):
        steps *= 2
    print(f"steps is... {steps}")

    
    # if debug:
    #     print("Settling torque controller...")
    # time.sleep(1.0) 
    
    
    snap = robot.get_state_snapshot()
    target = snap.ee_pos.clone()
    #theta = math.pi/2 #'roll' pi/3 to 2pi/3
    #phi = math.pi/4 # 'pitch'
    dx = distance * math.sin(theta) * math.cos(phi)
    dy = distance * math.sin(theta) * math.sin(phi)
    dz = distance * math.cos(theta)
    displacement = np.array([-dx, -dy, -dz], dtype=np.float64)
    displacement_norm = float(np.linalg.norm(displacement))
    # if displacement_norm < 1e-12:
    #     raise ValueError("pull distance and direction must define a non-zero displacement")
    excitation_direction = (displacement / displacement_norm).astype(np.float32)

    pull_data = []

    for i in range(steps):
        if(debug):
            print(f"Starting {i+1} of {steps}...")
        
        segment_idx = len(pull_data)
        target[0] -= (dx/steps)
        target[1] -= (dy/steps)
        target[2] -= (dz/steps)
        apple_quat = snap.ee_quat.clone()
        #gains[""]
        run_move(
            robot,
            gains,
            target,
            apple_quat,
            default_dof_pos,
            f"closer #{i}",
            prnt=debug,
            manage_control=False,
            record_rows=robot_rows,
            hold_index=segment_idx,
            hold_number=segment_idx,
            n_holds=stops,
            direction_idx=int(run_args.get("direction_index", 0)),
            n_directions=int(run_args.get("num_directions", 1)),
            excitation_direction=excitation_direction,
            amplitude_m=distance * float(segment_idx + 1) / float(stops),
            sample_label="moving",
        )
        
        if((i+1) % (steps/stops) == 0):
            s = 1
            if(debug):
                print(f"Holding position for {s}s...")
            hold_idx = len(pull_data)
            data = hold_and_record(
                robot,
                gains,
                target,
                apple_quat,
                default_dof_pos,
                duration_sec=s,
                device=device,
                record_rows=robot_rows,
                hold_index=hold_idx,
                hold_number=hold_idx,
                n_holds=stops,
                direction_idx=int(run_args.get("direction_index", 0)),
                n_directions=int(run_args.get("num_directions", 1)),
                excitation_direction=excitation_direction,
                amplitude_m=distance * float(hold_idx + 1) / float(stops),
                phase=1,
                phase_name="hold",
                sample_label="hold",
            )
            pull_data.append(data)
            #hold_position(robot, gains, target, apple_quat, default_dof_pos, duration_sec=s, device=device)
        
    
        
   
    
    # 1. Zero out the PD error so the arm stops trying to pull
    if(debug):
        print("Relaxing tension before release...")
    snap = robot.get_state_snapshot()
    hold_position(
        robot,
        gains,
        snap.ee_pos,
        snap.ee_quat,
        default_dof_pos,
        duration_sec=1.0,
        device=device,
    )

    gc.send_request(False)
    time.sleep(1) # wait for gripper to open

    # Safely drop out of torque mode
    robot.end_control()

    # Restore the robot before Parquet writing or optional post-run compilation.
    time.sleep(2)
    robot.reset_to_start_pose(home_pose_4x4)

    # Assemble and persist the completed static-hold episode.
    full_pull_data = np.concatenate(pull_data, axis=0)
    #plot_and_save_data(full_pull_data, label=label, plot=to_plot, metadata=run_args)

    robot_output = run_args.get("robot_output") or f"{label}_robot.parquet"
    hold_ranges = []
    for hold_idx in range(stops):
        hold_timestamps = [
            row["timestamp"] for row in robot_rows if row["hold_index"] == hold_idx
        ]
        if hold_timestamps:
            hold_ranges.append({
                "hold_index": hold_idx,
                "start_timestamp": min(hold_timestamps),
                "end_timestamp": max(hold_timestamps),
                "n_robot_frames": len(hold_timestamps),
            })
    robot_metadata = {
        "episode_id": episode_id,
        "collection_start_timestamp": collection_start_timestamp,
        "collection_end_timestamp": time.time(),
        "rest_reference_timestamp": rest_reference_timestamp,
        "collection_mode": "baseline" if baseline else "collect",
        "excitation_type": "quasi_static",
        "control_hz": float(robot._control_rate_hz),
        "theta_rad": float(theta),
        "phi_rad": float(phi),
        "pull_direction": excitation_direction.tolist(),
        "distance_m": float(distance),
        "n_holds": int(stops),
        "hold_duration_s": 1.0,
        "hold_ranges": hold_ranges,
        "direction_index": int(run_args.get("direction_index", 0)),
        "num_directions": int(run_args.get("num_directions", 1)),
        "action_semantics": "legacy 6D control placeholder; row phase indicates move vs hold",
        "sample_labels": [
            "rest_geometry",
            "post_grab_hold",
            "moving",
            "hold",
        ],
        "phase_encoding": {"moving": 0, "hold": 1},
        "ft_wrist_frame": "force only in EE frame; torque and all other robot-side kinematics in Franka base frame",
        "ft_wrist_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        "ft_wrist_sign": "environment-on-robot; pro_robot_interface rotates base to body and negates",
        "joint_torque_fields": {
            "order": [f"joint_{i}" for i in range(1, 8)],
            "order_direction": "base-to-end-effector",
            "unit": "N*m",
            "tau_J_d": "commanded/desired link-side joint torques without gravity",
        },
        "joint_position_fields": {
            "order": [f"joint_{i}" for i in range(1, 8)],
            "order_direction": "base-to-end-effector",
            "unit": "rad",
        },
        "ft_calibration": {
            "enabled": bool(ft_calibration_enabled),
            "note": "ft_bias is a measurement-time calibration offset; ee_config describes the physical/tool model.",
        },
        "ee_config": ee_config,
        "tcp_velocity_order": ["vx", "vy", "vz", "wx", "wy", "wz"],
        "tcp_pose_order": ["x", "y", "z", "qx", "qy", "qz", "qw"],
        "position_unit": "m",
        "linear_velocity_unit": "m/s",
        "angular_velocity_unit": "rad/s",
        "force_unit": "N",
        "torque_unit": "N*m",
        "robot_start_pose_4x4": np.asarray(pull_start_pose_4x4).tolist(),
        "pull_start_pose_name": str(run_args.get("pull_start_pose_name", "unspecified")),
        "home_pose_4x4": np.asarray(home_pose_4x4).tolist(),
        "controller_gains": {
            key: value.detach().cpu().tolist() if torch.is_tensor(value) else value
            for key, value in gains.items()
        },
        "config_source": {
            "path": run_args.get("config_path", "real_robot_exps/config.yaml"),
            "sha256": hashlib.sha256(json.dumps(config_snapshot, sort_keys=True, default=str).encode("utf-8")).hexdigest() if config_snapshot is not None else None,
        },
        "run_arguments": {
            "mode": run_args.get("mode"),
            "theta": run_args.get("theta"),
            "phi": run_args.get("phi"),
            "distance": run_args.get("distance"),
            "stops": run_args.get("stops"),
            "direction_index": run_args.get("direction_index"),
            "num_directions": run_args.get("num_directions"),
            "pull_start_pose_name": run_args.get("pull_start_pose_name"),
        },
        "raw_robot_row_count": len(robot_rows),
    }
    if baseline:
        robot_metadata["dynamic_baseline"] = {
            "role": "unloaded_baseline_source",
            "applied": False,
            "profile_field": "ft_wrist_raw",
            "note": "Use this file with a matching collect run; no correction is applied to baseline rows.",
        }
    elif USE_DYNAMIC_BASELINE_CORRECTION:
        baseline_path = Path(f"{base_label}_baseline_robot.parquet")
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Dynamic baseline correction is enabled, but {baseline_path} does not exist. "
                "Run the same trajectory with --mode baseline first, or set "
                "USE_DYNAMIC_BASELINE_CORRECTION = False."
            )
        baseline_metadata = _read_parquet_metadata(baseline_path)
        _validate_baseline_compatibility(robot_metadata, baseline_metadata, baseline_path)
        correction_metadata = apply_dynamic_baseline(robot_rows, baseline_path)
        correction_metadata.update({"role": "corrected_collect_run", "applied": True})
        robot_metadata["dynamic_baseline"] = correction_metadata
    else:
        robot_metadata["dynamic_baseline"] = {
            "role": "uncorrected_collect_run",
            "applied": False,
            "reason": "USE_DYNAMIC_BASELINE_CORRECTION is False",
        }
    saved_robot_path = save_robot_hold_parquet(robot_rows, robot_output, robot_metadata)
    print(f"Wrote robot hold data to {saved_robot_path}")

    return {"episode_id": episode_id, "robot_path": saved_robot_path}



def main():


    from real_robot_exps.gripper_test import GripperClient
    gc = GripperClient()
    



    parser = argparse.ArgumentParser(description="Integrated static apple-pull system-ID collection")
    parser.add_argument("--config", type=str, default="real_robot_exps/config.yaml", help="Real robot config path")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--override", action="append", default=[], help="Override config values")
    parser.add_argument("--mode", type=str, default="collect", choices=["collect", "baseline"], help="collect/baseline")
    parser.add_argument("--plot", default=None, action="store_true", help="True/False[default]")
    parser.add_argument("--debug", default="none", help="none/all/...")
    parser.add_argument("--kp", type=int, default=80, help="kp from 20-120 (kd is auto calculated)")
    parser.add_argument("--distance", type=float, default=0.05, help="pull distance in meters (0.01 to 0.075)")
    parser.add_argument("--stops", type=int, default=5, help="number of stops to record data during pull")
    parser.add_argument("--theta", type=float, default=2.36, help="angle determining height of pull (z-direction) in radians")
    parser.add_argument("--phi", type=float, default=1.57, help="angle determining left/right of pull (circle on xy plane) in radians")
    parser.add_argument("--direction-index", type=int, default=0, help="Zero-based direction index for one-hot encoding")
    parser.add_argument("--num-directions", type=int, default=1, help="Width of the direction one-hot vector")
    parser.add_argument("--robot-output", default=None, help="Raw robot Parquet output path")
    parser.add_argument("--tracking", default=None, help="Existing raw camera Parquet to compile after the robot run")
    parser.add_argument("--camera-frames", type=int, default=5, help="Camera frames per hold when compiling")
    parser.add_argument("--max-camera-delta", type=float, default=1.0, help="Maximum camera/robot timestamp difference when compiling")
    parser.add_argument("--unified-output", default=None, help="Compiled unified Parquet output path")
    args = parser.parse_args()

    if args.num_directions < 1:
        parser.error("--num-directions must be >= 1")
    if not 0 <= args.direction_index < args.num_directions:
        parser.error("--direction-index must be in [0, --num-directions)")

    device = args.device
    mode = args.mode # collect or baseline
    to_plot = args.plot is not None
    debug = args.debug
    kp = args.kp
    distance = args.distance
    stops = args.stops
    theta = args.theta
    phi = args.phi
    is_baseline = (mode == "baseline")

    if (mode != "collect") and (mode != "baseline"):
        print("Invalid mode command. Should be 'collect' or 'baseline'")
        sys.exit()

    if(debug != "none"):
        print("=" * 80)
        print("CONTROLLER VERIFICATION TEST")
        print("=" * 80)

    # 1. Load config
    if(debug != "none"):
        print(f"\nLoading config: {args.config}")

    with open(args.config, 'r') as f:
        real_config = yaml.safe_load(f)

    if args.override:
        for override in args.override:
            if '=' not in override:
                raise ValueError(f"Override must be 'key=value', got: {override}")
            key_path, value_str = override.split('=', 1)
            keys = key_path.split('.')
            parent = real_config
            for k in keys[:-1]:
                parent = parent[k]
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    if value_str.lower() == 'true':
                        value = True
                    elif value_str.lower() == 'false':
                        value = False
                    else:
                        value = value_str
            parent[keys[-1]] = value
            print(f"  Override: {key_path} = {value}")

    # 2. Load gains from config
    if debug != "none":
        print("\nLoading controller gains...")
    gains = load_gains_from_config(real_config, device)

   
    import pylibfranka as plf

    robot_cfg = real_config['robot']
    diag_robot = plf.Robot(robot_cfg['ip'])

    # Set NE_T_EE and EE_T_K exactly as FrankaInterface does
    NE_T_EE_cfg = robot_cfg.get('NE_T_EE', [
        0.7071, -0.7071, 0.0, 0.0,
        0.7071,  0.7071, 0.0, 0.0,
        0.0,     0.0,    1.0, 0.0,
        0.0,     0.0,    0.1034, 1.0,
    ])
    EE_T_K_cfg = robot_cfg.get('EE_T_K', [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
    
    diag_robot.set_EE(NE_T_EE_cfg)
    diag_robot.set_K(EE_T_K_cfg)


    diag_state = diag_robot.read_once()

    T = np.array(diag_state.O_T_EE)
    R = np.array([
        [T[0], T[4], T[8]],
        [T[1], T[5], T[9]],
        [T[2], T[6], T[10]],
    ])
    pos = T[12:15]

   
    ee_config = {
        "F_T_EE": np.asarray(diag_state.F_T_EE, dtype=np.float64).tolist(),
        "EE_T_K": np.asarray(diag_state.EE_T_K, dtype=np.float64).tolist(),
        "m_ee": float(getattr(diag_state, "m_ee", 0.0)),
        "F_x_Cee": np.asarray(getattr(diag_state, "F_x_Cee", [0.0, 0.0, 0.0]), dtype=np.float64).tolist(),
        "I_ee": np.asarray(getattr(diag_state, "I_ee", [0.0] * 9), dtype=np.float64).tolist(),
        "m_load": float(getattr(diag_state, "m_load", 0.0)),
        "F_x_Cload": np.asarray(getattr(diag_state, "F_x_Cload", [0.0, 0.0, 0.0]), dtype=np.float64).tolist(),
        "I_load": np.asarray(getattr(diag_state, "I_load", [0.0] * 9), dtype=np.float64).tolist(),
        "source": "pylibfranka RobotState",
        "note": "Recorded from the live robot state, not from config.yaml.",
    }

    diag_robot.stop()


    # 3. Initialize robot
    if(debug != "none"):
        print("\nInitializing robot interface...")
    robot = FrankaInterface(real_config, device=device)

 

    home_pose_4x4 = make_ee_target_pose_from_matrix(pos, R)

    # arbitrarily chosen 'home'
    home_rot = np.array([[-1, 0, 0.0], [0.0, 0.0, 1.0], [0, 1, 0]])
    home_pos = np.array([0.0, 0.85, 0.42])
    home_pose_4x4 = make_ee_target_pose_from_matrix(home_pos, home_rot)

    apple_rot = np.array([
                 [ -0.994, -.110, 0.00],
                 [0, 0, 1.000],
                 [-0.11,  .991,  0 ]
                 ])
    # print(R)
    # print(apple_rot)
    apple_pose_4x4 = make_ee_target_pose_from_matrix(np.array([0, .9262, .41]), apple_rot)
    close_roll_rad = math.radians(CLOSE_PULL_ROLL_FORWARD_DEG)
    close_roll_local_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(close_roll_rad), -math.sin(close_roll_rad)],
        [0.0, math.sin(close_roll_rad), math.cos(close_roll_rad)],
    ], dtype=np.float64)
    # Post-multiplication rolls about the local EE X axis. For apple_rot, a
    # positive angle tips the close-pose tool axis downward in base Z.
    close_rot = apple_rot @ close_roll_local_x
    close_pose_4x4 = make_ee_target_pose_from_matrix(
        CLOSE_PULL_START_POSITION_M,
        close_rot,
    )
    pull_start_pose_4x4 = (
        close_pose_4x4 if USE_CLOSE_PULL_START_POSE else apple_pose_4x4
    )
    pull_start_pose_name = "close_pose_4x4" if USE_CLOSE_PULL_START_POSE else "apple_pose_4x4"
    print(
        f"Pull start selection: {pull_start_pose_name} at "
        f"{pull_start_pose_4x4[:3, 3].tolist()} m"
    )

    # 6. Move to home and wait for user
    print("\nMoving to home position...")
    robot.reset_to_start_pose(home_pose_4x4)
    snap = robot.get_state_snapshot()
    home_actual = snap.ee_pos.clone()
    home_quat = snap.ee_quat.clone()
    default_dof_pos = snap.joint_pos.clone()
    home_rpy_deg = _quat_to_rpy_deg(home_quat)
    # print(f"  Home Pos: [{home_actual[0].item():.5f}, {home_actual[1].item():.5f}, {home_actual[2].item():.5f}]")
    # print(f"  Home Orn (RPY deg): [{home_rpy_deg[0]:.2f}, {home_rpy_deg[1]:.2f}, {home_rpy_deg[2]:.2f}]")   

    unified_result = None
    try:
        input(f"Press Enter to begin apple pull {mode} run...")

        gains = update_gains(gains, [kp, kp, kp, 30, 30, 30], device)
        angles = [(theta, phi)]
        run_arguments = dict(vars(args))
        run_arguments.update({
            "use_close_pull_start_pose": bool(USE_CLOSE_PULL_START_POSE),
            "pull_start_pose_name": pull_start_pose_name,
            "close_pull_start_position_m": CLOSE_PULL_START_POSITION_M.tolist(),
            "close_pull_roll_forward_deg": float(CLOSE_PULL_ROLL_FORWARD_DEG),
            "apple_pose_4x4": apple_pose_4x4.tolist(),
            "close_pose_4x4": close_pose_4x4.tolist(),
            "camera_collection": "separate process; compile after robot collection",
        })
        for theta_value, phi_value in angles:
            unified_result = pull_test(
                theta_value,
                phi_value,
                robot,
                pull_start_pose_4x4,
                default_dof_pos,
                gains,
                home_pose_4x4,
                gc,
                device=device,
                baseline=is_baseline,
                to_plot=to_plot,
                debug=(debug != "none"),
                distance=distance,
                stops=stops,
                args=run_arguments,
                config_snapshot=real_config,
                ee_config=ee_config,
                ft_calibration_enabled=bool(
                    real_config.get("robot", {}).get("ft_calibration_duration_sec", 0)
                ),
            )
    finally:
        robot.shutdown()
        gc.terminate()

    if unified_result is not None and args.tracking and not is_baseline:
        try:
            from real_robot_exps.compile_static_sysid import compile_static_episode
            from real_robot_exps.viz_static_sysid import _load_plot_data, plot_static_sysid
            import matplotlib.pyplot as plt

            unified_output = args.unified_output or f"pull_theta{theta:.2f}_phi{phi:.2f}_unified.parquet"
            unified_path = compile_static_episode(
                unified_result["robot_path"],
                args.tracking,
                unified_output,
                camera_frame_count=args.camera_frames,
                max_camera_delta_s=args.max_camera_delta,
                command_argv=sys.argv,
            )
            print(f"Wrote unified system-ID data to {unified_path}")
            viz_data = _load_plot_data(unified_path)
            fig = plot_static_sysid(viz_data, title=f"Unified system-ID viewer: {unified_path.name}")
            try:
                plt.show()
            except BaseException as exc:
                fallback_png = unified_path.with_suffix(".png")
                fig.savefig(fallback_png, dpi=200)
                print(f"Matplotlib GUI unavailable ({exc}); saved visualization to {fallback_png}")
        except BaseException as exc:
            print(f"Could not compile/open unified parquet visualizer: {exc}")

if __name__ == "__main__":
    main()
