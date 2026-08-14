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
import subprocess
import sys
import tempfile
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
# These preserve the original 15 Hz application timing as durations.  The
# configured control_rate_hz may be 1000 Hz, so frame-count constants would
# otherwise make convergence happen in milliseconds.
CONVERGE_DURATION_SEC = 10.0 / 15.0
MAX_MOVE_DURATION_SEC = 500.0 / 15.0
MOVE_DISTANCE = 0.02     # 5cm
DYNAMIC_PULL_APPROACH_CLEARANCE_M = 0.02
DYNAMIC_PULL_LOCAL_Z_TWIST_DEG = -18.5

MANUAL_SETUP_START_POSE_NAME = "manual_setup_current_tcp_pose"

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


def _pose_4x4_from_pos_quat(pos: torch.Tensor, quat: torch.Tensor) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _quat_wxyz_to_rotmat(quat)
    pose[:3, 3] = pos.detach().cpu().numpy().astype(np.float64, copy=False)
    return pose


def _pose_4x4_with_translation(template_pose_4x4: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.asarray(template_pose_4x4, dtype=np.float64).copy()
    pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return pose


def _pose_4x4_translated_along_direction(
    origin_pose_4x4: np.ndarray,
    direction: np.ndarray,
    distance_m: float,
) -> np.ndarray:
    """Translate a pose from its current position along a unit direction."""
    origin_pose_4x4 = np.asarray(origin_pose_4x4, dtype=np.float64).reshape(4, 4)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    pose = origin_pose_4x4.copy()
    pose[:3, 3] = pose[:3, 3] + direction * float(distance_m)
    return pose


def _rotmat_wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / max(float(np.linalg.norm(quat)), 1e-12)


def _pose_4x4_to_quat_tensor(pose_4x4: np.ndarray, *, device: str = "cpu") -> torch.Tensor:
    quat_wxyz = _rotmat_wxyz_from_matrix(np.asarray(pose_4x4, dtype=np.float64)[:3, :3])
    return torch.as_tensor(quat_wxyz, device=device, dtype=torch.float32)


def _rotation_about_local_z(angle_rad: float) -> np.ndarray:
    """Return a 3x3 rotation that twists about the local +Z axis."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _look_at_rotation_toward_apple(
    position: np.ndarray,
    apple_center: np.ndarray,
    fallback_rotation: np.ndarray,
    *,
    local_z_twist_deg: float = DYNAMIC_PULL_LOCAL_Z_TWIST_DEG,
) -> np.ndarray:
    """Return the fixed apple-facing reference rotation for dynamic lineup.

    The dynamic lineup path used to point the tool's local +Z axis at the pull
    direction. We now keep the apple-facing orientation from the reference
    pose so the arm stays face-on to the apple regardless of the requested
    pull vector.
    """
    _ = position, apple_center, local_z_twist_deg
    return np.asarray(fallback_rotation, dtype=np.float64).reshape(3, 3)


def _pull_direction_vector(theta: float, phi: float) -> np.ndarray:
    """Return the current pull direction convention used by the trajectory."""
    dx = math.sin(theta) * math.cos(phi)
    dy = math.sin(theta) * math.sin(phi)
    dz = math.cos(theta)
    return np.array([-dx, -dy, -dz], dtype=np.float64)


def _snapshot_geometry(snap, *, target_pose_4x4: np.ndarray | None = None) -> dict:
    geometry = {
        "tcp_pos": _flat_float32(snap.ee_pos.cpu().numpy()).tolist(),
        "tcp_pose_4x4": _flat_float32(_tcp_pose_4x4_from_snapshot(snap)).tolist(),
    }
    if target_pose_4x4 is not None:
        geometry["target_pose_4x4"] = _flat_float32(target_pose_4x4).tolist()
    return geometry


def _capture_camera_snapshot(*, request_path: str | None = None, output_path: str | None = None) -> dict:
    """Capture a snapshot through the running detector, or directly if absent."""
    if request_path and output_path:
        request = Path(request_path)
        output = Path(output_path)
        output.unlink(missing_ok=True)
        request.touch()
        deadline = time.time() + 15.0
        while time.time() < deadline:
            if output.exists():
                with output.open("r", encoding="utf-8") as f:
                    return json.load(f)
            time.sleep(0.05)
        raise RuntimeError(f"Timed out waiting for running detector snapshot: {output}")

    # Direct invocation fallback when apple_pullto_static is not launched by
    # runner.py with a detector process.
    with tempfile.NamedTemporaryFile(
        prefix="post_grasp_camera_snapshot_",
        suffix=".json",
        delete=False,
    ) as tmp:
        output_path = Path(tmp.name)
    command = [
        sys.executable,
        "-m",
        "real_robot_exps.camera_snapshot",
        "--output",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
        with output_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        if output_path.exists():
            output_path.unlink()


def _select_pre_grasp_snapshot(pre_grasp_geometry: dict) -> tuple[dict, str | None]:
    """Pick the best pre-grasp snapshot for dynamic pull staging.

    Settled / under-gravity snapshots are preferred because they reflect the
    relaxed apple pose we want to line the robot up against. Older cached
    metadata may only have a lengthened snapshot, so we keep that as a legacy
    fallback instead of hard-failing.
    """
    pre = dict(pre_grasp_geometry or {})
    for key in ("settled_snapshot", "under_gravity_snapshot", "lengthened_snapshot", "snapshot"):
        snapshot = dict(pre.get(key, {}) or {})
        if snapshot:
            return snapshot, key
    return {}, None


def _load_dynamic_pull_start_pose(
    run_metadata: dict,
    fallback_pose_4x4: np.ndarray,
    *,
    theta: float | None = None,
    phi: float | None = None,
    approach_clearance_m: float = DYNAMIC_PULL_APPROACH_CLEARANCE_M,
) -> tuple[np.ndarray, str, float | None, np.ndarray]:
    pre = dict(run_metadata.get("pre_grasp_geometry", {}) or {})
    snapshot, snapshot_source = _select_pre_grasp_snapshot(pre)
    if not snapshot:
        fallback_pose = np.asarray(fallback_pose_4x4, dtype=np.float64)
        return fallback_pose, "apple_pose_4x4", None, fallback_pose

    apple_pos_flat = snapshot.get("apple_pos")
    if apple_pos_flat is None:
        fallback_pose = np.asarray(fallback_pose_4x4, dtype=np.float64)
        return fallback_pose, "apple_pose_4x4", None, fallback_pose

    parts = dict(pre.get("parts", {}) or {})
    apple_radius_m = parts.get("apple", {}).get("radius_m")
    if apple_radius_m is None:
        fallback_pose = np.asarray(fallback_pose_4x4, dtype=np.float64)
        return fallback_pose, "apple_pose_4x4", None, fallback_pose
    apple_radius_m = float(apple_radius_m)

    apple_center = np.asarray(apple_pos_flat, dtype=np.float64).reshape(3)
    surface_pos = apple_center + np.array([0.0, -apple_radius_m, 0.0], dtype=np.float64)
    staged_pos = surface_pos + np.array([0.0, -float(approach_clearance_m), 0.0], dtype=np.float64)
    staged_rot = np.asarray(fallback_pose_4x4[:3, :3], dtype=np.float64)
    surface_rot = np.asarray(fallback_pose_4x4[:3, :3], dtype=np.float64)
    pose = _pose_4x4_with_translation(np.eye(4, dtype=np.float64), staged_pos)
    pose[:3, :3] = staged_rot
    surface_pose = _pose_4x4_with_translation(np.eye(4, dtype=np.float64), surface_pos)
    surface_pose[:3, :3] = surface_rot
    source_name = "settled_snapshot" if snapshot_source in {"settled_snapshot", "under_gravity_snapshot"} else "lengthened_snapshot"
    return pose, f"{source_name}_apple_surface_plus_2cm_pull_direction_offset", apple_radius_m, surface_pose


def _load_baseline_front_of_apple_pose(
    run_metadata: dict,
    fallback_pose_4x4: np.ndarray,
) -> tuple[np.ndarray, str, float | None, np.ndarray]:
    """Return a fixed-orientation pose translated to the live apple position."""
    pre = dict(run_metadata.get("pre_grasp_geometry", {}) or {})
    snapshot, snapshot_source = _select_pre_grasp_snapshot(pre)
    apple_pos_flat = snapshot.get("apple_pos")
    if apple_pos_flat is None:
        fallback_pose = np.asarray(fallback_pose_4x4, dtype=np.float64)
        return fallback_pose, "apple_pose_4x4", None, fallback_pose

    parts = dict(pre.get("parts", {}) or {})
    apple_radius_m = parts.get("apple", {}).get("radius_m")
    if apple_radius_m is None:
        fallback_pose = np.asarray(fallback_pose_4x4, dtype=np.float64)
        return fallback_pose, "apple_pose_4x4", None, fallback_pose

    apple_center = np.asarray(apple_pos_flat, dtype=np.float64).reshape(3)
    offset = np.array([0.0, -float(apple_radius_m), 0.0], dtype=np.float64)
    pose = _pose_4x4_with_translation(fallback_pose_4x4, apple_center + offset)
    source_name = "settled_snapshot" if snapshot_source in {"settled_snapshot", "under_gravity_snapshot"} else "lengthened_snapshot"
    return pose, f"{source_name}_front_of_apple_pose", float(apple_radius_m), pose


def _metadata_entry(metadata: dict) -> dict:
    row = {
        "row_kind": "metadata",
        "metadata_json": json.dumps(metadata, sort_keys=True, default=str),
    }
    return row


def _flat_float32(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _format_pos_m(value) -> str:
    vec = np.asarray(value, dtype=np.float64).reshape(3)
    return np.array2string(vec, precision=3, suppress_small=True)


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
    target_pose_4x4: np.ndarray,
    task_prop_gains: np.ndarray | torch.Tensor,
    task_deriv_gains: np.ndarray | torch.Tensor,
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
        "target_pose_4x4": _flat_float32(target_pose_4x4),
        "task_prop_gains": _flat_float32(task_prop_gains),
        "task_deriv_gains": _flat_float32(task_deriv_gains),
        "ft_wrist": _flat_float32(snap.force_torque.cpu().numpy()),
        "ft_wrist_raw": _flat_float32(snap.force_torque.cpu().numpy()),
        "tau_J_d": _flat_float32(snap.tau_J_d.cpu().numpy()),
        "joint_pos": _flat_float32(snap.joint_pos.cpu().numpy()),
        "joint_vel": _flat_float32(snap.joint_vel.cpu().numpy()),
        "tcp_velocity": _flat_float32(np.concatenate([
            snap.ee_linvel.cpu().numpy(),
            snap.ee_angvel.cpu().numpy(),
        ])),
        "action_wrench_ee": _flat_float32(
            np.zeros(6, dtype=np.float32) if action is None else np.asarray(action, dtype=np.float32).reshape(6)
        ),
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
    phase_name: str = "move",
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

    converge_start = None
    max_steps = max(1, int(math.ceil(MAX_MOVE_DURATION_SEC * robot._control_rate_hz)))
    for step in range(max_steps):
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        timestamp = time.time()
        robot.check_safety(snap)
        action_wrench = compute_pose_task_wrench(
            snap.ee_pos, snap.ee_quat, snap.ee_linvel, snap.ee_angvel,
            target_pos, target_quat,
            targets.task_prop_gains, targets.task_deriv_gains,
        )
        
        robot.set_control_targets(targets)
        _append_robot_sample(
            record_rows,
            timestamp=timestamp,
            hold_step_idx=step,
            hold_index=hold_index,
            phase=0,
            phase_name=phase_name,
            amplitude_m=amplitude_m,
            target_pose_4x4=_pose_4x4_from_pos_quat(target_pos, target_quat),
            task_prop_gains=targets.task_prop_gains,
            task_deriv_gains=targets.task_deriv_gains,
            hold_one_hot=hold_one_hot,
            direction_one_hot=direction_one_hot,
            excitation_direction=excitation_direction,
            snap=snap,
            sample_label=sample_label,
            action=action_wrench.detach().cpu().numpy(),
        )

        # Debug: replicate wrench computation from compute process for visibility
        if step == 0 or step % max(1, int(robot._control_rate_hz / 10.0)) == 0:
            pos_err, aa_err = compute_pose_error(
                snap.ee_pos, snap.ee_quat, target_pos, target_quat,
            )
            if(prnt):
                print(f"    [step {step:3d}] orn_error (axis-angle, base frame): "
                    f"[{aa_err[0].item():.6f}, {aa_err[1].item():.6f}, {aa_err[2].item():.6f}]")
                print(f"    [step {step:3d}] wrench [Fx,Fy,Fz,Tx,Ty,Tz] (base frame, pre-Lambda): "
                    f"[{action_wrench[0].item():.4f}, {action_wrench[1].item():.4f}, {action_wrench[2].item():.4f}, "
                    f"{action_wrench[3].item():.4f}, {action_wrench[4].item():.4f}, {action_wrench[5].item():.4f}]")
                print(f"    [step {step:3d}] frame: geometric Jacobian base frame "
                    f"(q_err = q_target * q_current^-1 -> axis-angle)")

        pos_delta = torch.norm(snap.ee_pos - prev_pos).item()
        prev_pos = snap.ee_pos.clone()

        if pos_delta < CONVERGE_THRESHOLD:
            converge_count += 1
            if converge_start is None:
                converge_start = time.monotonic()
        else:
            converge_count = 0
            converge_start = None

        if (
            converge_start is not None
            and time.monotonic() - converge_start >= CONVERGE_DURATION_SEC
        ):
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
    converged = (
        converge_start is not None
        and time.monotonic() - converge_start >= CONVERGE_DURATION_SEC
    )

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

        action_wrench = compute_pose_task_wrench(
            snap.ee_pos, snap.ee_quat, snap.ee_linvel, snap.ee_angvel,
            target_pos, target_quat,
            targets.task_prop_gains, targets.task_deriv_gains,
        )

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
            target_pose_4x4=_pose_4x4_from_pos_quat(target_pos, target_quat),
            task_prop_gains=targets.task_prop_gains,
            task_deriv_gains=targets.task_deriv_gains,
            hold_one_hot=hold_one_hot,
            direction_one_hot=direction_one_hot,
            excitation_direction=excitation_direction,
            snap=snap,
            action=action_wrench.detach().cpu().numpy(),
        )
    
    # print(ft_history)

        
    return np.array(ft_history)


def save_robot_hold_parquet(rows, filename, metadata):
    """Save raw robot-side hold rows and rich collection metadata."""
    data_rows = list(rows)
    if data_rows:
        data_table = pa.Table.from_pylist(data_rows)
        schema = pa.schema([
            pa.field("row_kind", pa.string()),
            pa.field("metadata_json", pa.string()),
            *data_table.schema,
        ])
        combined_rows = [_metadata_entry(metadata)] + [
            {"row_kind": None, "metadata_json": None, **row} for row in data_rows
        ]
    else:
        schema = pa.schema([
            pa.field("row_kind", pa.string()),
            pa.field("metadata_json", pa.string()),
        ])
        combined_rows = [_metadata_entry(metadata)]
    table = pa.Table.from_pylist(combined_rows, schema=schema)
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


def _extract_pre_grasp_apple_pos(pre_grasp_geometry: dict) -> np.ndarray | None:
    snapshot, _ = _select_pre_grasp_snapshot(pre_grasp_geometry)
    apple_pos = snapshot.get("apple_pos")
    if apple_pos is None:
        return None
    try:
        apple_pos_arr = np.asarray(apple_pos, dtype=np.float64).reshape(3)
    except Exception:
        return None
    if not np.all(np.isfinite(apple_pos_arr)):
        return None
    return apple_pos_arr


def _extract_baseline_kp(metadata: dict) -> float | None:
    dump = dict(metadata.get("dump", {}) or {})
    robot_info = dict(dump.get("robot_info", {}) or {})
    kp = robot_info.get("kp")
    if kp is None:
        return None
    try:
        kp_value = float(kp)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(kp_value):
        return None
    return kp_value


def _default_baseline_path(base_label: str, kp_value: float | None) -> Path:
    kp_suffix = f"_kp{float(kp_value):.0f}" if kp_value is not None else ""
    return Path(f"{base_label}{kp_suffix}_baseline_robot.parquet")


def _effective_manual_setup(mode: str, requested_manual_setup: bool) -> bool:
    return bool(requested_manual_setup) and mode != "baseline"


def _validate_baseline_compatibility(current: dict, baseline: dict, baseline_path: Path) -> None:
    mismatches = []

    for key, tolerance in (("theta_rad", 1e-9), ("phi_rad", 1e-9)):
        if key not in current or key not in baseline:
            mismatches.append(f"{key}=missing")
            continue
        if abs(float(current[key]) - float(baseline[key])) > tolerance:
            mismatches.append(f"{key}: collect={current[key]!r}, baseline={baseline[key]!r}")

    current_kp = _extract_baseline_kp(current)
    baseline_kp = _extract_baseline_kp(baseline)
    if current_kp is None or baseline_kp is None:
        mismatches.append("dump.robot_info.kp=missing")
    elif abs(current_kp - baseline_kp) > 1e-6:
        mismatches.append(f"kp: collect={current_kp!r}, baseline={baseline_kp!r}")

    current_pre = dict(current.get("pre_grasp_geometry", {}) or {})
    baseline_pre = dict(baseline.get("pre_grasp_geometry", {}) or {})
    current_apple_pos = _extract_pre_grasp_apple_pos(current_pre)
    baseline_apple_pos = _extract_pre_grasp_apple_pos(baseline_pre)
    if current_apple_pos is not None and baseline_apple_pos is not None:
        apple_delta = np.linalg.norm(current_apple_pos - baseline_apple_pos)
        if apple_delta > 0.10:
            mismatches.append(f"apple_pos differs by {apple_delta:.4f} m")
    if mismatches:
        raise ValueError(
            f"Baseline {baseline_path} is incompatible with this collect run: "
            + "; ".join(mismatches)
        )


def apply_dynamic_baseline(robot_rows: list[dict], baseline_path: Path) -> dict:
    """Subtract an unloaded baseline profile within each corresponding hold."""
    baseline_table = pq.read_table(baseline_path)
    baseline_rows_all = baseline_table.to_pylist()
    baseline_rows = [
        row for row in baseline_rows_all
        if str(row.get("row_kind", "data")) != "metadata"
    ]
    if not baseline_rows:
        raise ValueError(f"Baseline file has no rows: {baseline_path}")

    baseline_by_hold = {}
    for row in baseline_rows:
        if "hold_index" not in row:
            continue
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
    """Actively hold a Cartesian pose for the requested duration."""
    targets = build_position_targets(gains, target_pos, target_quat, default_dof_pos, device)
    
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

def pull_test(theta, phi, robot: FrankaInterface, pull_start_pose_4x4, pull_surface_pose_4x4, default_dof_pos, gains, home_pose_4x4, gc, device: str = "cpu", baseline: bool = False, debug: bool = False, to_plot: bool = False, distance: float = 0.05, stops: int = 5, args=None, config_snapshot=None, ee_config=None, ft_calibration_enabled: bool = False, pre_grasp_geometry=None, robot_info=None, kp_value=None, metadata_overrides=None):
    collection_start_timestamp = time.time()
    episode_id = str(uuid4())
    run_args = dict(args or {})
    base_label = f"pull_theta{theta:.2f}_phi{phi:.2f}"
    label = f"{base_label}_{'baseline' if baseline else 'raw'}"
    approach_geometry = {}
    post_grasp_geometry = {}
    only_metadata = bool(run_args.get("only_metadata", False))
    manual_setup_enabled = bool(run_args.get("manual_setup", False))
    approach_clearance_m = float(run_args.get("approach_clearance_m", DYNAMIC_PULL_APPROACH_CLEARANCE_M))
    approach_offset_m = float(run_args.get("approach_offset_m", approach_clearance_m))
    excitation_direction = np.zeros(3, dtype=np.float32)
    time.sleep(2.0)  # let the robot settle before recording the pull-start TCP snapshot
    
    snap = robot.get_state_snapshot()
    if manual_setup_enabled:
        # In manual mode we trust the current live pose instead of commanding a
        # staged reset that would override the user's setup.
        pull_start_pose_4x4 = _tcp_pose_4x4_from_snapshot(snap)
        pull_surface_pose_4x4 = pull_start_pose_4x4
    else:
        _reset_to_pose_if_needed(
            robot,
            pull_start_pose_4x4,
            label="Pull-start reset" if not baseline else "Baseline alignment",
        )
        snap = robot.get_state_snapshot()
    pull_start_snapshot = _snapshot_geometry(
        snap,
        target_pose_4x4=_pose_4x4_from_pos_quat(snap.ee_pos, snap.ee_quat),
    )
    pull_start_snapshot["setup_mode"] = "manual" if manual_setup_enabled else "dynamic"
    pull_start_snapshot["manual_setup_enabled"] = manual_setup_enabled
    pull_start_snapshot["pull_start_pose_name"] = str(run_args.get("pull_start_pose_name", "unspecified"))
    # `snapshot` is the compatibility field for the camera's lengthened state.
    # The manual TCP setup snapshot must never replace it.
    pre_geometry = dict(pre_grasp_geometry or {})
    lengthened_snapshot = dict(
        pre_geometry.get("snapshot", {})
        or pre_geometry.get("lengthened_snapshot", {})
        or pre_geometry.get("settled_snapshot", {})
        or {}
    )
    if bool(run_args.get("debug_pre_grasp", False)):
        settled = dict(pre_grasp_geometry or {})
        settled_snapshot = dict(
            settled.get("settled_snapshot", {})
            or settled.get("under_gravity_snapshot", {})
            or settled.get("lengthened_snapshot", {})
            or settled.get("snapshot", {})
            or {}
        )
        apple_pos = settled_snapshot.get("apple_pos")
        pull_start_target_pos = np.asarray(pull_start_pose_4x4[:3, 3], dtype=np.float64)
        tcp_pos = np.asarray(pull_start_snapshot["tcp_pos"], dtype=np.float64)
        print("\n[settled pre-grasp debug]")
        print(f"  pull_start_pose_name: {pull_start_snapshot['pull_start_pose_name']}")
        print(f"  pull_start_target_base_m: {_format_pos_m(pull_start_target_pos)}")
        print(f"  tcp_pos_base_m: {_format_pos_m(tcp_pos)}")
        if apple_pos is not None:
            apple_pos = np.asarray(apple_pos, dtype=np.float64)
            print(f"  apple_pos_base_m: {_format_pos_m(apple_pos)}")
            print(f"  tcp_minus_apple_base_m: {_format_pos_m(tcp_pos - apple_pos)}")
            print(f"  staged_pull_start_base_m: {_format_pos_m(pull_start_target_pos)}")
        else:
            print("  apple_pos_base_m: <missing>")
    robot_rows = []
    rest_reference_timestamp = time.time()
    # distance = .05
    # stops = 5
    steps = stops
    if(distance/steps > 0.01):
        steps *= 2
    if not only_metadata:
        print(f"steps is... {steps}")

    
    # if debug:
    #     print("Settling torque controller...")
    # time.sleep(1.0) 
    
    
    pull_data = []
    snap = robot.get_state_snapshot()
    pull_direction = _pull_direction_vector(theta, phi)
    pull_direction_norm = float(np.linalg.norm(pull_direction))
    if pull_direction_norm < 1e-12:
        raise ValueError("Pull direction is ill-defined for the provided theta/phi")
    pull_direction = pull_direction / pull_direction_norm
    excitation_direction = pull_direction.astype(np.float32)

    surface_target = torch.as_tensor(
        np.asarray(pull_surface_pose_4x4[:3, 3], dtype=np.float64),
        device=snap.ee_pos.device,
        dtype=snap.ee_pos.dtype,
    )
    surface_quat = _pose_4x4_to_quat_tensor(pull_surface_pose_4x4, device=snap.ee_pos.device)

    if debug:
        surface_target_pos = np.asarray(pull_surface_pose_4x4[:3, 3], dtype=np.float64)
        print("\n[apple alignment]")
        print(f"  approach_clearance_m: {approach_clearance_m:.3f}")
        print(f"  approach_offset_m: {approach_offset_m:.3f}")
        print(f"  target_base_m: {_format_pos_m(pull_start_pose_4x4[:3, 3])}")
        print(f"  pull_direction_base_m: {_format_pos_m(pull_direction)}")

    # Both collection modes use the same approach and grasp sequence. Metadata
    # mode only skips the force/pull portion after this point.
    if manual_setup_enabled:
        print(
            "Manual setup mode: keeping the live robot pose and using it as the "
            "apple pose without an additional alignment step."
        )
    elif baseline:
        print(
            "Baseline mode: moving straight to the front-of-apple pose with the "
            "normal orientation."
        )
    else:
        print(
            "Collect mode: moving straight to the apple pose using the live "
            "apple translation."
        )
    if not manual_setup_enabled:
        # The initial alignment is the only reset we want before force/pull
        # data collection. Keep it to a single apple-facing move.
        snap = robot.get_state_snapshot()
    position_error = float(torch.linalg.vector_norm(snap.ee_pos - surface_target).item())
    _, orientation_error = compute_pose_error(
        snap.ee_pos,
        snap.ee_quat,
        surface_target,
        surface_quat,
    )
    orientation_error_deg = float(torch.linalg.vector_norm(orientation_error).item() * 180.0 / math.pi)
    approach_geometry = _snapshot_geometry(
        snap,
        target_pose_4x4=pull_surface_pose_4x4,
    )
    approach_geometry["pose_name"] = str(run_args.get("pull_surface_pose_name", "unspecified"))
    approach_geometry["target_reached"] = bool(position_error <= CONVERGE_THRESHOLD)
    approach_geometry["target_position_error_m"] = position_error
    approach_geometry["target_orientation_error_deg"] = orientation_error_deg
    alignment_label = "Baseline" if baseline else "Pull"
    print(
        f"{alignment_label} apple alignment complete: "
        f"reached={approach_geometry['target_reached']} "
        f"position_error={approach_geometry['target_position_error_m'] * 1000.0:.2f} mm"
    )

    rest_reference_timestamp = time.time()
    gc.send_request(True)
    # This is the only close command. Capture post-grasp state after closure,
    # while retaining the pre-close approach snapshot above.
    time.sleep(4)
    snap = robot.get_state_snapshot()
    post_grasp_pose_4x4 = _pose_4x4_from_pos_quat(snap.ee_pos, snap.ee_quat)
    if not baseline:
        robot_post_grasp_geometry = _snapshot_geometry(
            snap,
            target_pose_4x4=post_grasp_pose_4x4,
        )
        post_grasp_geometry = dict(robot_post_grasp_geometry)
        post_grasp_camera_snapshot = _capture_camera_snapshot(
            request_path=run_args.get("post_grasp_camera_request"),
            output_path=run_args.get("post_grasp_camera_output"),
        )
        post_grasp_geometry["snapshot"] = post_grasp_camera_snapshot
        post_grasp_geometry["robot_snapshot"] = robot_post_grasp_geometry
        post_grasp_geometry["camera_snapshot_source"] = "post_grasp_camera_capture"
        post_grasp_geometry["setup_mode"] = "manual" if manual_setup_enabled else "dynamic"
        post_grasp_geometry["manual_setup_enabled"] = manual_setup_enabled
        post_grasp_geometry["pull_start_pose_name"] = str(run_args.get("pull_start_pose_name", "unspecified"))
        post_grasp_geometry["pull_surface_pose_name"] = str(run_args.get("pull_surface_pose_name", "unspecified"))
        post_grasp_geometry["pull_surface_pose_4x4"] = np.asarray(pull_surface_pose_4x4).tolist()
        post_grasp_geometry["pull_origin_pose_4x4"] = np.asarray(post_grasp_pose_4x4).tolist()
        post_grasp_geometry["post_grasp_pose_name"] = "post_grasp_surface_pose"

    if only_metadata:
        direction_idx = int(run_args.get("direction_index", 0))
        n_directions = int(run_args.get("num_directions", 1))
        hold_one_hot = np.zeros(max(1, stops), dtype=np.float32)
        hold_one_hot[0] = 1.0
        direction_one_hot = np.zeros(max(1, n_directions), dtype=np.float32)
        if 0 <= direction_idx < direction_one_hot.size:
            direction_one_hot[direction_idx] = 1.0
        _append_robot_sample(
            robot_rows,
            timestamp=time.time(),
            hold_step_idx=0,
            hold_index=0,
            phase=1,
            phase_name="metadata_only",
            sample_label="metadata_only_post_grasp",
            amplitude_m=0.0,
            target_pose_4x4=_pose_4x4_from_pos_quat(snap.ee_pos, snap.ee_quat),
            task_prop_gains=gains["task_prop_gains"],
            task_deriv_gains=gains["task_deriv_gains"],
            hold_one_hot=hold_one_hot,
            direction_one_hot=direction_one_hot,
            excitation_direction=excitation_direction,
            snap=snap,
            action=np.zeros(6, dtype=np.float32),
        )
    else:
        robot.start_torque_mode()

    if not only_metadata:
        pull_origin_pose_4x4 = post_grasp_pose_4x4
        pull_origin_quat = _pose_4x4_to_quat_tensor(pull_origin_pose_4x4, device=snap.ee_pos.device)
        for i in range(steps):
            if(debug):
                print(f"Starting {i+1} of {steps}...")
            
            segment_idx = len(pull_data)
            step_target = torch.as_tensor(
                _pose_4x4_translated_along_direction(
                    pull_origin_pose_4x4,
                    pull_direction,
                    distance * float(segment_idx + 1) / float(stops),
                )[:3, 3],
                device=snap.ee_pos.device,
                dtype=snap.ee_pos.dtype,
            )
            run_move(
                robot,
                gains,
                step_target,
                pull_origin_quat,
                default_dof_pos,
                f"pull #{i}",
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
                phase_name="pull",
                sample_label="pull",
            )
            
            if((i+1) % (steps/stops) == 0):
                s = 1
                if(debug):
                    print(f"Holding position for {s}s...")
                hold_idx = len(pull_data)
                data = hold_and_record(
                    robot,
                    gains,
                    step_target,
                    pull_origin_quat,
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
        
    
        
   
    
    # 1. Zero out the PD error so the arm stops trying to pull
    if not only_metadata:
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
    if not manual_setup_enabled:
        robot.reset_to_start_pose(home_pose_4x4)

    # Assemble and persist the completed static-hold episode.
    full_pull_data = np.concatenate(pull_data, axis=0) if pull_data else np.zeros((0, 6), dtype=np.float32)
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
    dump_blob = {
        "episode_id": episode_id,
        "collection_start_timestamp": collection_start_timestamp,
        "collection_end_timestamp": time.time(),
        "rest_reference_timestamp": rest_reference_timestamp,
        "collection_mode": "metadata_only" if only_metadata else ("baseline" if baseline else "collect"),
        "excitation_type": "quasi_static",
        "control_hz": float(robot._control_rate_hz),
        "theta_rad": float(theta),
        "phi_rad": float(phi),
        "pull_direction": excitation_direction.tolist(),
        "distance_m": float(0.0 if only_metadata else distance),
        "approach_offset_m": float(run_args.get("approach_offset_m", DYNAMIC_PULL_APPROACH_CLEARANCE_M)),
        "approach_clearance_m": float(run_args.get("approach_clearance_m", DYNAMIC_PULL_APPROACH_CLEARANCE_M)),
        "n_holds": int(1 if only_metadata else stops),
        "hold_duration_s": 0.0 if only_metadata else 1.0,
        "hold_ranges": hold_ranges,
        "direction_index": int(run_args.get("direction_index", 0)),
        "num_directions": int(run_args.get("num_directions", 1)),
        "action_semantics": "per-frame pose-control wrench [Fx, Fy, Fz, Tx, Ty, Tz] computed from the current pose error and velocity",
        "sample_labels": [
            "approach",
            "pull",
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
        "pull_surface_pose_4x4": np.asarray(pull_surface_pose_4x4).tolist(),
        "pull_surface_pose_name": str(run_args.get("pull_surface_pose_name", "unspecified")),
        "approach_offset_m": float(run_args.get("approach_offset_m", DYNAMIC_PULL_APPROACH_CLEARANCE_M)),
        "approach_clearance_m": float(run_args.get("approach_clearance_m", DYNAMIC_PULL_APPROACH_CLEARANCE_M)),
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
            "only_metadata": bool(only_metadata),
            "theta": run_args.get("theta"),
            "phi": run_args.get("phi"),
            "distance": run_args.get("distance"),
            "approach_offset_m": run_args.get("approach_offset_m"),
            "approach_clearance_m": run_args.get("approach_clearance_m"),
            "stops": run_args.get("stops"),
            "direction_index": run_args.get("direction_index"),
            "num_directions": run_args.get("num_directions"),
            "pull_start_pose_name": run_args.get("pull_start_pose_name"),
            "pull_surface_pose_name": run_args.get("pull_surface_pose_name"),
        },
        "raw_robot_row_count": len(robot_rows),
    }
    robot_metadata = {
        "dump": {
            **dump_blob,
            "robot_info": {
                "kp": float(kp_value) if kp_value is not None else float(np.asarray(gains["task_prop_gains"]).reshape(-1)[0].item()),
            },
        },
        "pre_grasp_geometry": {
            **(pre_grasp_geometry or {}),
            "snapshot": dict(
                pre_geometry.get("snapshot", {})
                or pre_geometry.get("lengthened_snapshot", {})
                or pre_geometry.get("settled_snapshot", {})
                or {}
            ),
            "pull_start_pose_name": str(run_args.get("pull_start_pose_name", "unspecified")),
            "pull_surface_pose_name": str(run_args.get("pull_surface_pose_name", "unspecified")),
            "robot_snapshot": pull_start_snapshot,
        },
        "approach_geometry": approach_geometry,
        "post_grasp_geometry": post_grasp_geometry,
        "theta_rad": float(theta),
        "phi_rad": float(phi),
        "distance_m": float(distance),
        "approach_offset_m": float(run_args.get("approach_offset_m", DYNAMIC_PULL_APPROACH_CLEARANCE_M)),
        "approach_clearance_m": float(run_args.get("approach_clearance_m", DYNAMIC_PULL_APPROACH_CLEARANCE_M)),
        "n_holds": int(stops),
        "pull_start_pose_name": str(run_args.get("pull_start_pose_name", "unspecified")),
        "pull_surface_pose_name": str(run_args.get("pull_surface_pose_name", "unspecified")),
        "robot_start_pose_4x4": np.asarray(pull_start_pose_4x4).tolist(),
        "pull_surface_pose_4x4": np.asarray(pull_surface_pose_4x4).tolist(),
    }
    if metadata_overrides:
        robot_metadata["dump"].setdefault("runner_metadata", metadata_overrides)
    if only_metadata:
        robot_metadata["dynamic_baseline"] = {
            "role": "metadata_only_run",
            "applied": False,
            "reason": "only_metadata mode skips force/torque baseline handling",
        }
    elif baseline:
        robot_metadata["dynamic_baseline"] = {
            "role": "unloaded_baseline_source",
            "applied": False,
            "profile_field": "ft_wrist_raw",
            "note": "Use this file with a matching collect run; no correction is applied to baseline rows.",
        }
    elif USE_DYNAMIC_BASELINE_CORRECTION:
        baseline_path = Path(run_args.get("baseline_path")) if run_args.get("baseline_path") else _default_baseline_path(base_label, kp_value)
        if not baseline_path.exists():
            print(
                f"Warning: dynamic baseline correction is enabled, but {baseline_path} does not exist. "
                "Writing the collect run with uncorrected ft_wrist data."
            )
            robot_metadata["dynamic_baseline"] = {
                "role": "uncorrected_collect_run",
                "applied": False,
                "reason": "baseline file missing",
                "baseline_path": str(baseline_path.resolve()),
            }
        else:
            try:
                baseline_metadata = _read_parquet_metadata(baseline_path)
                _validate_baseline_compatibility(robot_metadata, baseline_metadata, baseline_path)
                correction_metadata = apply_dynamic_baseline(robot_rows, baseline_path)
                correction_metadata.update({"role": "corrected_collect_run", "applied": True})
                robot_metadata["dynamic_baseline"] = correction_metadata
            except ValueError as exc:
                print(
                    f"Warning: {exc}. "
                    "Writing the collect run with uncorrected ft_wrist data."
                )
                robot_metadata["dynamic_baseline"] = {
                    "role": "uncorrected_collect_run",
                    "applied": False,
                    "reason": "baseline incompatible with collect run",
                    "error": str(exc),
                    "baseline_path": str(baseline_path.resolve()),
                }
    else:
        robot_metadata["dynamic_baseline"] = {
            "role": "uncorrected_collect_run",
            "applied": False,
            "reason": "USE_DYNAMIC_BASELINE_CORRECTION is False",
        }
    saved_robot_path = save_robot_hold_parquet(robot_rows, robot_output, robot_metadata)
    print(f"Wrote robot hold data to {saved_robot_path}")

    return {
        "episode_id": episode_id,
        "robot_path": saved_robot_path,
    }


def _prompt_or_continue(prompt: str, skip: bool) -> None:
    if skip:
        print(prompt)
        return
    input(prompt)


def _reset_to_pose_if_needed(robot, target_pose_4x4: np.ndarray, *, label: str, threshold_m: float = 0.02) -> bool:
    """Skip a reset if the TCP is already close to the requested pose."""
    target_pose = np.asarray(target_pose_4x4, dtype=np.float64)
    if target_pose.shape != (4, 4):
        raise ValueError(f"Expected [4, 4] target pose for {label}, got shape {target_pose.shape}")

    try:
        current_snap = robot.get_state_snapshot()
    except Exception:
        current_snap = None

    if current_snap is not None:
        current_pos = np.asarray(current_snap.ee_pos.detach().cpu().numpy(), dtype=np.float64).reshape(3)
        target_pos = np.asarray(target_pose[:3, 3], dtype=np.float64).reshape(3)
        translation_delta = float(np.linalg.norm(current_pos - target_pos))
        if translation_delta <= float(threshold_m):
            print(
                f"{label}: already within {translation_delta * 1000.0:.1f} mm of target, skipping reset."
            )
            return False

    print(f"{label}: resetting to target pose.")
    robot.reset_to_start_pose(target_pose)
    return True



def main():
    parser = argparse.ArgumentParser(description="Integrated static apple-pull system-ID collection")
    parser.add_argument("--config", type=str, default="real_robot_exps/config.yaml", help="Real robot config path")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--override", action="append", default=[], help="Override config values")
    parser.add_argument("--mode", type=str, default="collect", choices=["collect", "baseline"], help="collect/baseline")
    parser.add_argument("--plot", default=None, action="store_true", help="True/False[default]")
    parser.add_argument("--debug", default="none", help="none/all/...")
    parser.add_argument("--kp", type=float, default=100, help="kp from 20-120 (kd is auto calculated)")
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
    parser.add_argument("--baseline-path", default=None, help="Explicit dynamic baseline Parquet path to use in collect mode")
    parser.add_argument("--unified-output", default=None, help="Compiled unified Parquet output path")
    parser.add_argument("--run-metadata-file", default=None, help="Optional JSON file containing structure/direction metadata to embed in the run output")
    parser.add_argument("--only-metadata", action=argparse.BooleanOptionalAction, default=False, help="Capture pre/post-grasp reconstruction metadata only; skip baseline correction and pull trajectory")
    parser.add_argument("--manual-setup", action=argparse.BooleanOptionalAction, default=False, help="Pause without torque mode so the arm can be manually positioned on the apple surface before the pull")
    parser.add_argument("--debug-pre-grasp", action=argparse.BooleanOptionalAction, default=False, help="Print lengthened pre-grasp apple and TCP positions during the run")
    parser.add_argument("--mock-gripper", action=argparse.BooleanOptionalAction, default=False, help="Use a no-op gripper client and never connect to the real gripper")
    parser.add_argument("--skip-enter", action=argparse.BooleanOptionalAction, default=False, help="Skip the run-start Enter prompt")
    parser.add_argument("--post-grasp-camera-request", default=None, help="Request path used to ask a running detector for the post-grasp snapshot")
    parser.add_argument("--post-grasp-camera-output", default=None, help="Output path for the running detector post-grasp snapshot")
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
    manual_setup_enabled = _effective_manual_setup(mode, bool(args.manual_setup))

    if (mode != "collect") and (mode != "baseline"):
        print("Invalid mode command. Should be 'collect' or 'baseline'")
        sys.exit()

    from real_robot_exps.gripper_test import GripperClient
    gc = GripperClient(mock=bool(args.mock_gripper))

    run_metadata = {}
    if args.run_metadata_file:
        with open(args.run_metadata_file, "r", encoding="utf-8") as f:
            run_metadata = json.load(f)

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

    # arbitrarily chosen 'home'
    home_rot = np.array([[-1, 0, 0.0], [0.0, 0.0, 1.0], [0, 1, 0]])
    home_pos = np.array([0.00, 0.61, 0.41])
    home_pose_4x4 = make_ee_target_pose_from_matrix(home_pos, home_rot)

    apple_rot = np.array([
                 [ -0.994, -.110, 0.00],
                 [0, 0, 1.000],
                 [-0.11,  .991,  0 ]
                 ])
    # print(R)
    # print(apple_rot)
    apple_pose_4x4 = make_ee_target_pose_from_matrix(np.array([0, .87, .41]), apple_rot)
    dynamic_pull_apple_radius_m = None
    dynamic_pull_stage_pose_4x4 = apple_pose_4x4
    dynamic_pull_surface_pose_4x4 = apple_pose_4x4
    dynamic_pull_stage_pose_name = "apple_pose_4x4"
    dynamic_pull_surface_pose_name = "apple_pose_4x4"
    manual_setup_snapshot = None
    pull_start_pose_4x4 = dynamic_pull_surface_pose_4x4
    pull_surface_pose_4x4 = dynamic_pull_surface_pose_4x4
    pull_start_pose_name = dynamic_pull_surface_pose_name
    pull_surface_pose_name = dynamic_pull_surface_pose_name
    # Keep the metadata field defined for both manual and dynamic setup flows.
    approach_offset_m = float(DYNAMIC_PULL_APPROACH_CLEARANCE_M)
    approach_clearance_m = float(DYNAMIC_PULL_APPROACH_CLEARANCE_M)
    if is_baseline:
        print("\nBaseline mode selected.")
        print("Using the live apple position, the apple radius, and the normal orientation.")
        approach_offset_m = 0.0
        approach_clearance_m = 0.0
        dynamic_pull_stage_pose_4x4, dynamic_pull_stage_pose_name, dynamic_pull_apple_radius_m, dynamic_pull_surface_pose_4x4 = _load_baseline_front_of_apple_pose(
            run_metadata,
            apple_pose_4x4,
        )
        dynamic_pull_surface_pose_name = dynamic_pull_stage_pose_name
        pull_start_pose_4x4 = dynamic_pull_stage_pose_4x4
        pull_surface_pose_4x4 = dynamic_pull_surface_pose_4x4
        pull_start_pose_name = dynamic_pull_stage_pose_name
        pull_surface_pose_name = dynamic_pull_surface_pose_name
    elif manual_setup_enabled:
        print("\nManual setup mode selected.")
        print("The robot will stay where you place it and torque mode will remain off.")
        print("Move the arm to the desired manual pull-start pose and make the apple/structure visible.")
        input("Press Enter once the manual setup pose is ready: ")
        robot.refresh_state_snapshot()
        manual_snap = robot.get_state_snapshot()
        pull_start_pose_4x4 = _tcp_pose_4x4_from_snapshot(manual_snap)
        pull_surface_pose_4x4 = pull_start_pose_4x4
        pull_start_pose_name = MANUAL_SETUP_START_POSE_NAME
        pull_surface_pose_name = MANUAL_SETUP_START_POSE_NAME
        dynamic_pull_apple_radius_m = None
        manual_setup_snapshot = _snapshot_geometry(
            manual_snap,
            target_pose_4x4=pull_start_pose_4x4,
        )
        manual_setup_snapshot["setup_mode"] = "manual"
        manual_setup_snapshot["manual_setup_enabled"] = True
        manual_setup_snapshot["pull_start_pose_name"] = pull_start_pose_name
        run_metadata.setdefault("pre_grasp_geometry", {})
        run_metadata["pre_grasp_geometry"]["manual_setup_snapshot"] = manual_setup_snapshot
        run_metadata["pre_grasp_geometry"]["manual_setup_enabled"] = True
        run_metadata["pre_grasp_geometry"]["setup_mode"] = "manual"
        default_dof_pos = manual_snap.joint_pos.clone()
        print(
            f"Manual pull start selection: {pull_start_pose_name} at "
            f"{_format_pos_m(pull_start_pose_4x4[:3, 3])} m"
        )
    else:
        dynamic_pull_stage_pose_4x4, dynamic_pull_stage_pose_name, dynamic_pull_apple_radius_m, dynamic_pull_surface_pose_4x4 = _load_dynamic_pull_start_pose(
            run_metadata,
            apple_pose_4x4,
            theta=theta,
            phi=phi,
            approach_clearance_m=DYNAMIC_PULL_APPROACH_CLEARANCE_M,
        )
        approach_offset_m = (
            float(dynamic_pull_apple_radius_m + DYNAMIC_PULL_APPROACH_CLEARANCE_M)
            if dynamic_pull_apple_radius_m is not None
            else float(DYNAMIC_PULL_APPROACH_CLEARANCE_M)
        )
        dynamic_pull_surface_pose_name = (
            "apple_pose_4x4"
            if dynamic_pull_stage_pose_name == "apple_pose_4x4"
            else dynamic_pull_stage_pose_name.removesuffix("_plus_2cm_pull_direction_offset")
        )
        pull_start_pose_4x4 = dynamic_pull_stage_pose_4x4
        pull_surface_pose_4x4 = dynamic_pull_surface_pose_4x4
        pull_start_pose_name = dynamic_pull_stage_pose_name
        pull_surface_pose_name = dynamic_pull_surface_pose_name
        if dynamic_pull_stage_pose_name == "apple_pose_4x4":
            print(
                "WARNING: dynamic apple snapshot/radius was unavailable, so the "
                "old hardcoded apple pose fallback is being used."
            )
        print(
            f"Pull start selection: {pull_start_pose_name} at "
            f"{_format_pos_m(pull_start_pose_4x4[:3, 3])} m"
        )
        if pull_surface_pose_4x4 is not None:
            print(
                f"Pull surface selection: {pull_surface_pose_name} at "
                f"{_format_pos_m(pull_surface_pose_4x4[:3, 3])} m"
            )
        if dynamic_pull_apple_radius_m is not None:
            print(f"Dynamic apple radius from structure metadata: {dynamic_pull_apple_radius_m:.5f} m")
        print(
            "Direct staged-front mode: the first arm move goes to the front of "
            f"the apple at {pull_start_pose_name} ({_format_pos_m(pull_start_pose_4x4[:3, 3])} m)"
        )

    # 6. Move to home and wait for user
    if not manual_setup_enabled:
        print("\nMoving to home position...")
        robot.reset_to_start_pose(home_pose_4x4)
        snap = robot.get_state_snapshot()
        home_actual = snap.ee_pos.clone()
        home_quat = snap.ee_quat.clone()
        default_dof_pos = snap.joint_pos.clone()
        home_rpy_deg = _quat_to_rpy_deg(home_quat)
    else:
        print("\nManual setup will use the current robot pose directly; skipping home move.")
        snap = robot.get_state_snapshot()
        home_actual = snap.ee_pos.clone()
        home_quat = snap.ee_quat.clone()
        home_rpy_deg = _quat_to_rpy_deg(home_quat)
    # print(f"  Home Pos: [{home_actual[0].item():.5f}, {home_actual[1].item():.5f}, {home_actual[2].item():.5f}]")
    # print(f"  Home Orn (RPY deg): [{home_rpy_deg[0]:.2f}, {home_rpy_deg[1]:.2f}, {home_rpy_deg[2]:.2f}]")   

    unified_result = None
    try:
        print(args.skip_enter)
        _prompt_or_continue(f"Press Enter to begin apple pull {mode} run...", bool(args.skip_enter))

        gains = update_gains(gains, [kp, kp, kp, 30, 30, 30], device)
        angles = [(theta, phi)]
        run_arguments = dict(vars(args))
        run_arguments.update({
            "manual_setup": bool(manual_setup_enabled),
            "pull_start_pose_name": pull_start_pose_name,
            "pull_surface_pose_name": pull_surface_pose_name,
            "dynamic_apple_radius_m": None if dynamic_pull_apple_radius_m is None else float(dynamic_pull_apple_radius_m),
            "approach_offset_m": float(approach_offset_m),
            "approach_clearance_m": float(approach_clearance_m),
            "dynamic_pull_local_z_twist_deg": float(DYNAMIC_PULL_LOCAL_Z_TWIST_DEG),
            "apple_pose_4x4": apple_pose_4x4.tolist(),
            "dynamic_pull_pose_4x4": dynamic_pull_stage_pose_4x4.tolist(),
            "dynamic_pull_surface_pose_4x4": dynamic_pull_surface_pose_4x4.tolist(),
            "manual_setup_snapshot": manual_setup_snapshot,
            "camera_collection": "separate process; compile after robot collection",
        })
        for theta_value, phi_value in angles:
            unified_result = pull_test(
                theta_value,
                phi_value,
                robot,
                pull_start_pose_4x4,
                pull_surface_pose_4x4,
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
                pre_grasp_geometry=run_metadata.get("pre_grasp_geometry"),
                robot_info=run_metadata.get("robot_info"),
                kp_value=kp,
                metadata_overrides=run_metadata,
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
