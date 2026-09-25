"""Field pull series: one manual grasp, every direction, one torque session.

Run by ``field_session`` after the operator has hand-guided the gripper onto the
apple and the gripper has closed. The pose the arm is in when this starts is
the start pose for every direction. For each direction it:

1. settles at the start pose (structure stops swinging),
2. records ``post_grasp_geometry`` (robot state + a detector snapshot),
3. checks the grasp has not slipped (apple-to-TCP offset vs the first direction),
4. pulls in ``stops`` steps along the direction, holding after each step,
5. returns to the start pose (not recorded) and writes ``dXX_robot.parquet``.

The gripper stays closed until the last direction is done. Any error or Ctrl-C
opens the gripper, writes the rows of the interrupted direction with
``aborted: true`` and records progress in the status file, so the session can
resume with the remaining directions after a new grasp.

Usage (normally invoked by field_session):
    python -m real_robot_exps.field_pull --plan A003/pulls/pull_plan.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml

from real_robot_exps.apple_pullto_static import (
    _pose_4x4_from_pos_quat,
    _pose_4x4_translated_along_direction,
    _pull_direction_vector,
    _snapshot_geometry,
    _tcp_pose_4x4_from_snapshot,
    hold_and_record,
    hold_position,
    load_gains_from_config,
    run_move,
    save_robot_hold_parquet,
    update_gains,
)
from real_robot_exps.field_config import apply_overrides
from real_robot_exps.snapshot_geometry import SnapshotError, request_snapshot


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _read_ee_config(config: dict) -> dict:
    """End-effector/load parameters from the live robot (as apple_pullto_static records them)."""
    robot_cfg = config["robot"]
    if robot_cfg.get("use_mock", False):
        return {"source": "mock robot"}
    import pylibfranka as plf

    robot = plf.Robot(robot_cfg["ip"])
    try:
        robot.set_EE(robot_cfg["NE_T_EE"])
        robot.set_K(robot_cfg["EE_T_K"])
        state = robot.read_once()
        return {
            "F_T_EE": np.asarray(state.F_T_EE, dtype=np.float64).tolist(),
            "EE_T_K": np.asarray(state.EE_T_K, dtype=np.float64).tolist(),
            "m_ee": float(getattr(state, "m_ee", 0.0)),
            "F_x_Cee": np.asarray(getattr(state, "F_x_Cee", [0.0] * 3), dtype=np.float64).tolist(),
            "I_ee": np.asarray(getattr(state, "I_ee", [0.0] * 9), dtype=np.float64).tolist(),
            "m_load": float(getattr(state, "m_load", 0.0)),
            "F_x_Cload": np.asarray(getattr(state, "F_x_Cload", [0.0] * 3), dtype=np.float64).tolist(),
            "I_load": np.asarray(getattr(state, "I_load", [0.0] * 9), dtype=np.float64).tolist(),
            "source": "pylibfranka RobotState",
        }
    finally:
        robot.stop()


def _apple_offset_from_tcp(camera_snapshot: dict | None, tcp_pos: np.ndarray) -> np.ndarray | None:
    if not camera_snapshot or "apple_pos" not in camera_snapshot:
        return None
    return np.asarray(camera_snapshot["apple_pos"], dtype=np.float64) - np.asarray(tcp_pos, dtype=np.float64)


class PullSeries:
    """State for one apple's pull series; see the module docstring."""

    def __init__(self, plan: dict, *, robot=None, gripper=None, input_fn=input):
        self.plan = plan
        self.input_fn = input_fn
        self.status_path = Path(plan["status_path"])
        self.status = {
            "completed_directions": [],
            "aborted": False,
            "error": None,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.config = self._load_config()
        self.robot = robot
        self.gripper = gripper
        self.gains = None
        self.first_apple_offset = None

    # -- setup -----------------------------------------------------------------
    def _load_config(self) -> dict:
        with Path(self.plan["config_path"]).open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        return apply_overrides(config, list(self.plan.get("overrides", [])))

    def _connect(self) -> None:
        if self.gripper is None:
            from real_robot_exps.gripper_test import GripperClient

            self.gripper = GripperClient(mock=bool(self.plan.get("mock_gripper", False)), timeout_s=30.0)
        self.ee_config = _read_ee_config(self.config)
        if self.robot is None:
            from real_robot_exps.pro_robot_interface import FrankaInterface

            self.robot = FrankaInterface(self.config, device="cpu")
        kp = float(self.plan["kp"])
        gains = load_gains_from_config(self.config, "cpu")
        self.gains = update_gains(gains, [kp, kp, kp, 30, 30, 30], "cpu")

    # -- helpers ---------------------------------------------------------------
    def _snapshot(self, label: str, output: Path) -> dict:
        snapshot_dir = self.plan.get("snapshot_dir")
        if not snapshot_dir:
            return {"error": "no detector (snapshot_dir not set)"}
        try:
            return request_snapshot(
                snapshot_dir,
                label,
                output,
                frames=int(self.plan.get("snapshot_frames", 5)),
                timeout_s=float(self.plan.get("snapshot_timeout_s", 10.0)),
            )
        except SnapshotError as exc:
            # Never abort while holding the apple just because a tag was hidden.
            print(f"[field_pull] WARNING: {exc}")
            return {"error": str(exc)}

    def _save_status(self) -> None:
        _write_json(self.status_path, self.status)

    # -- main flow -------------------------------------------------------------
    def run(self) -> dict:
        self._save_status()
        rows: list[dict] = []
        direction = None
        try:
            self._connect()
            self.robot.refresh_state_snapshot()
            start_snap = self.robot.get_state_snapshot()
            self.start_pos = start_snap.ee_pos.clone()
            self.start_quat = start_snap.ee_quat.clone()
            self.start_pose_4x4 = _tcp_pose_4x4_from_snapshot(start_snap)
            self.start_joint_pos = start_snap.joint_pos.clone()
            self.status["start_pose_4x4"] = self.start_pose_4x4.tolist()
            self.status["start_joint_pos"] = self.start_joint_pos.tolist()
            self._check_expected_start()
            print(f"[field_pull] start pose (TCP) = {np.round(self.start_pose_4x4[:3, 3], 4).tolist()} m")
            self._report_calibration_check()

            self.robot.start_torque_mode()
            directions = list(self.plan["directions"])
            for position, direction in enumerate(directions):
                rows = []
                self._run_direction(direction, position, rows)
                rows = []
                self.status["completed_directions"].append(int(direction["index"]))
                self._save_status()
                if position + 1 < len(directions) and self.plan.get("confirm_each", False):
                    answer = self.input_fn(
                        f"Direction d{int(direction['index']):02d} done; apple still held. "
                        "Enter = next direction, 's' = stop the series: "
                    ).strip().lower()
                    if answer == "s":
                        self.status["stopped_by_operator"] = True
                        break
            self._finish()
        except BaseException as exc:
            self.status["aborted"] = True
            self.status["error"] = f"{type(exc).__name__}: {exc}"
            self._abort(rows, direction, exc)
            self._save_status()
            raise
        self.status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self._save_status()
        return self.status

    def _report_calibration_check(self) -> None:
        """TCP (robot) vs apple tag (camera): a gross calibration/frame error shows up here."""
        pre = (self.plan.get("run_metadata") or {}).get("pre_grasp_geometry") or {}
        apple_pos = (pre.get("under_gravity_snapshot") or {}).get("apple_pos")
        if apple_pos is None:
            return
        distance = float(np.linalg.norm(np.asarray(apple_pos, dtype=np.float64) - self.start_pose_4x4[:3, 3]))
        self.status["tcp_to_apple_tag_m"] = distance
        verdict = "OK" if distance < 0.08 else "CHECK THE CALIBRATION"
        print(f"[field_pull] gripper TCP to apple tag (under-gravity snapshot): {distance * 1000:.0f} mm -> {verdict}")

    def _check_expected_start(self) -> None:
        expected = self.plan.get("expected_start_pose_4x4")
        if expected is None:
            return
        delta = float(np.linalg.norm(np.asarray(expected, dtype=np.float64)[:3, 3] - self.start_pose_4x4[:3, 3]))
        tolerance = float(self.plan.get("start_tolerance_m", 0.005))
        if delta > tolerance:
            raise RuntimeError(
                f"Arm is {delta * 1000:.1f} mm from the expected start pose (> {tolerance * 1000:.0f} mm)"
            )

    def _hold_start(self, duration_s: float) -> None:
        hold_position(
            self.robot, self.gains, self.start_pos, self.start_quat, self.start_joint_pos,
            duration_sec=float(duration_s),
        )

    def _run_direction(self, direction: dict, position: int, rows: list[dict]) -> None:
        index = int(direction["index"])
        theta = float(direction["theta"])
        phi = float(direction["phi"])
        label = f"d{index:02d}"
        print(f"\n[field_pull] === {label}: theta={theta:.3f} phi={phi:.3f} ({position + 1}/{len(self.plan['directions'])}) ===")
        collection_start = time.time()

        # 1. settle at the start pose with the apple held
        settle_s = float(self.plan.get("settle_sec", 5.0))
        print(f"[field_pull] settling {settle_s:.1f} s at the start pose")
        self._hold_start(settle_s)
        rest_reference_timestamp = time.time()

        # 2. post-grasp geometry: robot state + camera snapshot
        snap = self.robot.get_state_snapshot()
        robot_geometry = _snapshot_geometry(snap, target_pose_4x4=self.start_pose_4x4)
        robot_geometry["joint_pos"] = snap.joint_pos.tolist()
        snapshot_dir_out = Path(self.plan["snapshot_output_dir"])
        camera_snapshot = self._snapshot(f"post_grasp_{label}", snapshot_dir_out / f"post_grasp_{label}.json")

        # 3. slip check against the first direction
        slip = {"checked": False}
        offset = _apple_offset_from_tcp(
            None if "error" in camera_snapshot else camera_snapshot, snap.ee_pos.cpu().numpy()
        )
        if offset is not None:
            if self.first_apple_offset is None:
                self.first_apple_offset = offset
            drift = float(np.linalg.norm(offset - self.first_apple_offset))
            threshold = float(self.plan.get("slip_threshold_m", 0.01))
            slip = {"checked": True, "drift_m": drift, "threshold_m": threshold, "apple_minus_tcp_m": offset.tolist()}
            print(f"[field_pull] apple-to-TCP drift vs first direction: {drift * 1000:.1f} mm")
            if drift > threshold:
                answer = self.input_fn(
                    f"Apple moved {drift * 1000:.1f} mm relative to the gripper (> {threshold * 1000:.0f} mm). "
                    "Enter = continue anyway, 's' = stop the series: "
                ).strip().lower()
                slip["operator_continued"] = answer != "s"
                if answer == "s":
                    raise RuntimeError(f"Stopped by operator: grasp slipped {drift * 1000:.1f} mm")
        post_grasp_geometry = {
            **robot_geometry,
            "robot_snapshot": robot_geometry,
            "camera_snapshot": camera_snapshot,
            "pull_origin_pose_4x4": self.start_pose_4x4.tolist(),
            "direction_index": index,
            "series_position": position,
            "settle_sec": settle_s,
            "slip_check": slip,
            "grasp": "single manual grasp held across the series",
        }
        if "error" in camera_snapshot:
            post_grasp_geometry["camera_error"] = camera_snapshot["error"]

        # 4. pull + holds (recorded)
        pull_direction = _pull_direction_vector(theta, phi)
        pull_direction = pull_direction / float(np.linalg.norm(pull_direction))
        excitation = pull_direction.astype(np.float32)
        stops = int(self.plan["stops"])
        distance = float(self.plan["distance_m"])
        hold_s = float(self.plan.get("hold_duration_s", 1.0))
        n_directions = int(self.plan["num_directions"])
        for hold_idx in range(stops):
            amplitude = distance * float(hold_idx + 1) / float(stops)
            step_target = torch.as_tensor(
                _pose_4x4_translated_along_direction(self.start_pose_4x4, pull_direction, amplitude)[:3, 3],
                dtype=self.start_pos.dtype,
            )
            common = dict(
                record_rows=rows,
                hold_index=hold_idx,
                hold_number=hold_idx,
                n_holds=stops,
                direction_idx=index,
                n_directions=n_directions,
                excitation_direction=excitation,
                amplitude_m=amplitude,
            )
            run_move(
                self.robot, self.gains, step_target, self.start_quat, self.start_joint_pos,
                f"{label} pull #{hold_idx}", prnt=False, manage_control=False,
                phase_name="pull", sample_label="pull", **common,
            )
            hold_and_record(
                self.robot, self.gains, step_target, self.start_quat, self.start_joint_pos,
                duration_sec=hold_s, phase=1, phase_name="hold", sample_label="hold", **common,
            )
        collection_end = time.time()

        # 5. back to the start pose (not recorded) and save
        run_move(
            self.robot, self.gains, self.start_pos, self.start_quat, self.start_joint_pos,
            f"{label} return", prnt=False, manage_control=False,
        )
        self._hold_start(1.0)
        self._save_direction(
            direction, rows,
            collection_start=collection_start,
            collection_end=collection_end,
            rest_reference_timestamp=rest_reference_timestamp,
            post_grasp_geometry=post_grasp_geometry,
            pull_direction=pull_direction,
        )

    def _metadata(self, direction: dict, rows: list[dict], **info) -> dict:
        plan = self.plan
        index = int(direction["index"])
        stops = int(plan["stops"])
        run_metadata = dict(plan.get("run_metadata") or {})
        hold_ranges = []
        for hold_idx in range(stops):
            stamps = [row["timestamp"] for row in rows if row["hold_index"] == hold_idx]
            if stamps:
                hold_ranges.append({
                    "hold_index": hold_idx,
                    "start_timestamp": min(stamps),
                    "end_timestamp": max(stamps),
                    "n_robot_frames": len(stamps),
                })
        episode_id = str(uuid4())
        dump = {
            "episode_id": episode_id,
            "collection_start_timestamp": info["collection_start"],
            "collection_end_timestamp": info["collection_end"],
            "rest_reference_timestamp": info["rest_reference_timestamp"],
            "collection_mode": "field_series",
            "excitation_type": "quasi_static",
            "control_hz": float(self.robot._control_rate_hz),
            "theta_rad": float(direction["theta"]),
            "phi_rad": float(direction["phi"]),
            "pull_direction": np.asarray(info["pull_direction"]).tolist(),
            "distance_m": float(plan["distance_m"]),
            "n_holds": stops,
            "hold_duration_s": float(plan.get("hold_duration_s", 1.0)),
            "hold_ranges": hold_ranges,
            "direction_index": index,
            "num_directions": int(plan["num_directions"]),
            "action_semantics": "per-frame pose-control wrench [Fx, Fy, Fz, Tx, Ty, Tz] computed from the current pose error and velocity",
            "phase_encoding": {"moving": 0, "hold": 1},
            "ft_wrist_frame": "K_F_ext_hat_K (stiffness = EE frame), EMA-filtered, negated to environment-on-robot",
            "ft_wrist_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            "joint_torque_fields": {"order": [f"joint_{i}" for i in range(1, 8)], "unit": "N*m",
                                    "tau_J_d": "commanded/desired link-side joint torques without gravity"},
            "ee_config": self.ee_config,
            "controller_gains": {
                key: value.detach().cpu().tolist() if torch.is_tensor(value) else value
                for key, value in self.gains.items()
            },
            "robot_info": {"kp": float(plan["kp"])},
            "config_source": {
                "path": str(plan["config_path"]),
                "overrides": list(plan.get("overrides", [])),
                "sha256": hashlib.sha256(
                    json.dumps(self.config, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest(),
            },
            "raw_robot_row_count": len(rows),
            "aborted": bool(info.get("aborted", False)),
        }
        if info.get("error"):
            dump["error"] = info["error"]
        return {
            # Top-level keys read by compile_static_sysid / collect_joint_velocity_baseline.
            "episode_id": episode_id,
            "rest_reference_timestamp": info["rest_reference_timestamp"],
            "theta_rad": float(direction["theta"]),
            "phi_rad": float(direction["phi"]),
            "direction_index": index,
            "direction_name": str(direction.get("name", "")),
            "distance_m": float(plan["distance_m"]),
            "n_holds": stops,
            "robot_start_pose_4x4": self.start_pose_4x4.tolist(),
            "robot_start_joint_pos": self.start_joint_pos.tolist(),
            "apple_id": run_metadata.get("apple_id"),
            "session": run_metadata.get("session"),
            "structure": run_metadata.get("apple_id"),
            "dump": dump,
            "pre_grasp_geometry": run_metadata.get("pre_grasp_geometry", {}),
            "post_grasp_geometry": info.get("post_grasp_geometry", {}),
            "dynamic_baseline": {
                "role": "baseline_pending",
                "applied": False,
                "application_stage": "compile_static_episode",
                "reason": "baseline is replayed after the apple is released",
            },
            "field_session": run_metadata,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "command_argv": list(sys.argv),
        }

    def _save_direction(self, direction: dict, rows: list[dict], **info) -> Path:
        output = Path(direction["output"])
        if info.get("aborted"):
            # Keep interrupted rows for inspection, but never under the name a
            # resumed run (or baseline/compile) would pick up.
            stem = output.name[: -len(".parquet")]
            number = len(list(output.parent.glob(f"{stem}.aborted-*.parquet")))
            output = output.with_name(f"{stem}.aborted-{number:02d}.parquet")
        metadata = self._metadata(direction, rows, **info)
        path = save_robot_hold_parquet(rows, output, metadata)
        print(f"[field_pull] wrote {path} ({len(rows)} rows)")
        return path

    def _finish(self) -> None:
        print("\n[field_pull] series done: holding, then opening the gripper")
        self._hold_start(1.0)
        self.gripper.send_request(False)
        time.sleep(1.0)
        self.robot.end_control()
        self.robot.shutdown()
        self.gripper.terminate()

    def _abort(self, rows: list[dict], direction: dict | None, exc: BaseException) -> None:
        print(f"\n[field_pull] ABORT: {type(exc).__name__}: {exc}")
        robot, gripper = self.robot, self.gripper
        if robot is not None:
            try:
                snap = robot.get_state_snapshot()
                hold_position(robot, self.gains, snap.ee_pos, snap.ee_quat, snap.joint_pos, duration_sec=0.5)
            except BaseException as hold_exc:  # keep going: the gripper must open
                print(f"[field_pull] hold before release failed: {hold_exc}")
        if gripper is not None:
            try:
                gripper.send_request(False)
                print("[field_pull] gripper opened")
            except BaseException as grip_exc:
                print(f"[field_pull] COULD NOT OPEN THE GRIPPER: {grip_exc}")
        if rows and direction is not None:
            try:
                self._save_direction(
                    direction, rows,
                    collection_start=rows[0]["timestamp"],
                    collection_end=rows[-1]["timestamp"],
                    rest_reference_timestamp=rows[0]["timestamp"],
                    post_grasp_geometry={},
                    pull_direction=_pull_direction_vector(float(direction["theta"]), float(direction["phi"])),
                    aborted=True,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.status["partial_direction"] = int(direction["index"])
            except BaseException as save_exc:
                print(f"[field_pull] could not save partial rows: {save_exc}")
        if robot is not None:
            for step in (robot.end_control, robot.shutdown):
                try:
                    step()
                except BaseException as stop_exc:
                    print(f"[field_pull] {step.__name__} failed: {stop_exc}")
        if gripper is not None:
            try:
                gripper.terminate()
            except BaseException:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True, help="pull_plan.json written by field_session")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    PullSeries(plan).run()


if __name__ == "__main__":
    main()
