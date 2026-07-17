"""Visualize unified static system-ID Parquet data against time.

This viewer is designed for the unified episode Parquet written by
``real_robot_exps.compile_static_sysid``. It plots the robot-side signals,
camera-derived geometry, and experiment state on a shared time axis.

Usage:
    python -m real_robot_exps.viz_static_sysid \
        --input pull_unified.parquet \
        --save fig.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


TRACKED_NAMES = ("Branch", "Spur", "Apple")
WOODY_PART_NAMES = ("Branch", "Spur", "Apple")


def _read_dataset_metadata(path: Path) -> dict[str, Any]:
    raw = pq.read_schema(path).metadata or {}
    payload = raw.get(b"dataset_metadata")
    if payload is None:
        return {}
    return json.loads(payload.decode("utf-8"))


def _as_array(values: list[list[float]] | list[float], ndim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if ndim == 1:
        return arr
    return arr.reshape(arr.shape[0], -1)


def _vector_columns(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def _decode_one_hot(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0], dtype=int)
    return np.argmax(matrix, axis=1)


@dataclass
class PlotData:
    timestamps: np.ndarray
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]
    has_camera: bool


def _load_plot_data(path: Path) -> PlotData:
    table = pq.read_table(path)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"No rows found in {path}")
    timestamps = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
    metadata = _read_dataset_metadata(path)
    required_camera_fields = {
        "apple_pos",
        "woody_part_start_pos",
        "woody_part_end_pos",
        "woody_bending_angles",
        "camera_timestamp",
        "robot_camera_timestamp_offset_s",
        "camera_frame_count",
        "camera_data_valid",
    }
    has_camera = required_camera_fields.issubset(rows[0].keys())
    return PlotData(timestamps=timestamps, rows=rows, metadata=metadata, has_camera=has_camera)


def _episode_id_from_metadata(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    candidates = [
        metadata.get("episode_id"),
        metadata.get("source_metadata", {}).get("robot", {}).get("episode_id"),
        metadata.get("source_metadata", {}).get("robot", {}).get("command_arguments", {}).get("episode_id"),
        rows[0].get("episode_id") if rows else None,
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return ""


def _plot_vector_panel(ax, t, values, labels, title):
    for idx, label in enumerate(labels):
        ax.plot(t, values[:, idx], label=label, linewidth=1.4)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=min(3, len(labels)), fontsize=8)


def _hold_boundaries(rows: list[dict[str, Any]]) -> list[float]:
    boundaries: list[float] = []
    last = None
    for row in rows:
        hold = int(row["hold_index"])
        if last is None:
            last = hold
            continue
        if hold != last:
            boundaries.append(float(row["timestamp"]))
            last = hold
    return boundaries


def _delta_cm(values: np.ndarray) -> np.ndarray:
    return (values - values[0]) * 100.0


def plot_static_sysid(
    data: PlotData,
    *,
    title: str | None = None,
):
    rows = data.rows
    t = data.timestamps

    ft = _vector_columns(rows, "ft_wrist")
    torque_fields = [
        ("tau_J", "Measured joint torque $\\tau_J$"),
        ("tau_ext_hat_filtered", "Estimated external joint torque $\\hat{\\tau}_{ext}$ (filtered)"),
        ("tau_J_d", "Desired joint torque $\\tau_{J,d}$ (without gravity)"),
    ]
    available_torques = [
        (field, title, _vector_columns(rows, field))
        for field, title in torque_fields if field in rows[0]
    ]
    # Allow older files to remain inspectable; new collection files use the
    # explicit libfranka field names above.
    if not available_torques and "joint_torques" in rows[0]:
        available_torques = [
            ("joint_torques", "Measured joint torque (legacy tau_J field)",
             _vector_columns(rows, "joint_torques"))
        ]
    vel = _vector_columns(rows, "tcp_velocity")
    action = _vector_columns(rows, "action")
    tcp_pos = _vector_columns(rows, "tcp_pos")
    has_camera = data.has_camera
    if has_camera:
        apple_pos = _vector_columns(rows, "apple_pos")
        start_pos = _vector_columns(rows, "woody_part_start_pos").reshape(len(rows), 3, 3)
        end_pos = _vector_columns(rows, "woody_part_end_pos").reshape(len(rows), 3, 3)
        bend = _vector_columns(rows, "woody_bending_angles")
        camera_offset = np.asarray([row["robot_camera_timestamp_offset_s"] for row in rows], dtype=np.float64)
        camera_count = np.asarray([row["camera_frame_count"] for row in rows], dtype=np.float64)
        camera_valid = np.asarray([row["camera_data_valid"] for row in rows], dtype=bool)
    else:
        apple_pos = start_pos = end_pos = bend = None
        camera_offset = camera_count = camera_valid = None
    hold_number = np.asarray([row["hold_number"] for row in rows], dtype=np.float64)
    direction = np.asarray([row["direction"] for row in rows], dtype=np.float64)
    phase = np.asarray([row["phase"] for row in rows], dtype=np.float64)
    amplitude = np.asarray([row["amplitude_m"] for row in rows], dtype=np.float64)
    hold_index = np.asarray([row["hold_index"] for row in rows], dtype=int)
    hold_step_idx = np.asarray([row["hold_step_idx"] for row in rows], dtype=int)
    episode_id = _episode_id_from_metadata(data.metadata, rows)

    torque_panel_count = len(available_torques)
    n_panels = (8 if has_camera else 6) + torque_panel_count
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 4 * n_panels), sharex=True, constrained_layout=True)

    _plot_vector_panel(axes[0], t, ft, ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"], "Wrist wrench")
    for panel_idx, (_, panel_title, values) in enumerate(available_torques, start=1):
        _plot_vector_panel(
            axes[panel_idx], t, values,
            [f"joint {i}" for i in range(1, 8)], panel_title,
        )
        axes[panel_idx].set_ylabel("N m")
    offset = torque_panel_count
    _plot_vector_panel(axes[1 + offset], t, vel, ["vx", "vy", "vz", "wx", "wy", "wz"], "TCP velocity")
    _plot_vector_panel(axes[2 + offset], t, action, ["ax", "ay", "az", "awx", "awy", "awz"], "Recorded action")

    if has_camera:
        tcp_pos_cm = tcp_pos * 100.0
        apple_pos_cm = apple_pos * 100.0
        start_pos_cm = start_pos * 100.0
        end_pos_cm = end_pos * 100.0
        tcp_pos_delta_cm = _delta_cm(tcp_pos)
        apple_pos_delta_cm = _delta_cm(apple_pos)
        start_pos_delta_cm = _delta_cm(start_pos.reshape(len(rows), -1)).reshape(len(rows), 3, 3)
        end_pos_delta_cm = _delta_cm(end_pos.reshape(len(rows), -1)).reshape(len(rows), 3, 3)

        axes[3 + offset].plot(t, tcp_pos_cm[:, 0], label="tcp x")
        axes[3 + offset].plot(t, tcp_pos_cm[:, 1], label="tcp y")
        axes[3 + offset].plot(t, tcp_pos_cm[:, 2], label="tcp z")
        axes[3 + offset].plot(t, apple_pos_cm[:, 0], "--", label="apple x")
        axes[3 + offset].plot(t, apple_pos_cm[:, 1], "--", label="apple y")
        axes[3 + offset].plot(t, apple_pos_cm[:, 2], "--", label="apple z")
        axes[3 + offset].set_title("Absolute positions")
        axes[3 + offset].set_ylabel("cm")
        axes[3 + offset].grid(True, alpha=0.25)
        axes[3 + offset].legend(loc="upper right", ncol=3, fontsize=8)

        axes[4 + offset].plot(t, tcp_pos_delta_cm[:, 0], label="tcp x")
        axes[4 + offset].plot(t, tcp_pos_delta_cm[:, 1], label="tcp y")
        axes[4 + offset].plot(t, tcp_pos_delta_cm[:, 2], label="tcp z")
        axes[4 + offset].plot(t, apple_pos_delta_cm[:, 0], "--", label="apple x")
        axes[4 + offset].plot(t, apple_pos_delta_cm[:, 1], "--", label="apple y")
        axes[4 + offset].plot(t, apple_pos_delta_cm[:, 2], "--", label="apple z")
        axes[4 + offset].set_title("Position deltas from first sample")
        axes[4 + offset].set_ylabel("delta cm")
        axes[4 + offset].grid(True, alpha=0.25)
        axes[4 + offset].legend(loc="upper right", ncol=3, fontsize=8)

        axes[5 + offset].plot(t, start_pos_cm[:, 0, 0], label="Branch start x")
        axes[5 + offset].plot(t, start_pos_cm[:, 0, 1], label="Branch start y")
        axes[5 + offset].plot(t, start_pos_cm[:, 0, 2], label="Branch start z")
        axes[5 + offset].plot(t, end_pos_cm[:, 0, 0], "--", label="Branch end x")
        axes[5 + offset].plot(t, end_pos_cm[:, 0, 1], "--", label="Branch end y")
        axes[5 + offset].plot(t, end_pos_cm[:, 0, 2], "--", label="Branch end z")
        axes[5 + offset].set_title("Branch endpoint absolute positions")
        axes[5 + offset].set_ylabel("cm")
        axes[5 + offset].grid(True, alpha=0.25)
        axes[5 + offset].legend(loc="upper right", ncol=3, fontsize=8)

        axes[6 + offset].plot(t, start_pos_delta_cm[:, 0, 0], label="Branch start delta x")
        axes[6 + offset].plot(t, start_pos_delta_cm[:, 0, 1], label="Branch start delta y")
        axes[6 + offset].plot(t, start_pos_delta_cm[:, 0, 2], label="Branch start delta z")
        axes[6 + offset].plot(t, end_pos_delta_cm[:, 0, 0], "--", label="Branch end delta x")
        axes[6 + offset].plot(t, end_pos_delta_cm[:, 0, 1], "--", label="Branch end delta y")
        axes[6 + offset].plot(t, end_pos_delta_cm[:, 0, 2], "--", label="Branch end delta z")
        axes[6 + offset].set_title("Branch endpoint deltas from first sample")
        axes[6 + offset].set_ylabel("delta cm")
        axes[6 + offset].grid(True, alpha=0.25)
        axes[6 + offset].legend(loc="upper right", ncol=3, fontsize=8)

        axes[7 + offset].plot(t, bend[:, 0], label="Branch")
        axes[7 + offset].plot(t, bend[:, 1], label="Spur")
        axes[7 + offset].plot(t, bend[:, 2], label="Apple")
        axes[7 + offset].set_title("Woody bending angles")
        axes[7 + offset].set_ylabel("rad")
        axes[7 + offset].grid(True, alpha=0.25)
        axes[7 + offset].legend(loc="upper right")
    else:
        axes[3 + offset].plot(t, tcp_pos[:, 0] * 100.0, label="tcp x")
        axes[3 + offset].plot(t, tcp_pos[:, 1] * 100.0, label="tcp y")
        axes[3 + offset].plot(t, tcp_pos[:, 2] * 100.0, label="tcp z")
        axes[3 + offset].set_title("TCP position")
        axes[3 + offset].set_ylabel("cm")
        axes[3 + offset].grid(True, alpha=0.25)
        axes[3 + offset].legend(loc="upper right", ncol=3, fontsize=8)

        axes[4 + offset].plot(t, _delta_cm(tcp_pos)[:, 0], label="tcp x")
        axes[4 + offset].plot(t, _delta_cm(tcp_pos)[:, 1], label="tcp y")
        axes[4 + offset].plot(t, _delta_cm(tcp_pos)[:, 2], label="tcp z")
        axes[4 + offset].set_title("TCP position deltas from first sample")
        axes[4 + offset].set_ylabel("delta cm")
        axes[4 + offset].grid(True, alpha=0.25)
        axes[4 + offset].legend(loc="upper right", ncol=3, fontsize=8)

        axes[5 + offset].plot(t, hold_index, label="hold_index")
        axes[5 + offset].plot(t, hold_step_idx, label="hold_step_idx", alpha=0.8)
        axes[5 + offset].plot(t, phase, label="phase", alpha=0.8)
        axes[5 + offset].plot(t, amplitude, label="amplitude_m", alpha=0.8)
        axes[5 + offset].plot(t, direction.argmax(axis=1), label="direction idx", alpha=0.8)
        axes[5 + offset].plot(t, hold_number.argmax(axis=1), label="hold idx one-hot", alpha=0.8)
        axes[5 + offset].set_title("Experiment state")
        axes[5 + offset].grid(True, alpha=0.25)
        axes[5 + offset].legend(loc="upper right", ncol=3, fontsize=8)

    boundaries = _hold_boundaries(rows)
    for ax in axes:
        for boundary in boundaries:
            ax.axvline(boundary, color="k", alpha=0.12, linewidth=1)
        if has_camera and len(t):
            ax.scatter(t[~camera_valid], np.zeros(np.count_nonzero(~camera_valid)), s=12, color="red", alpha=0.3)

    axes[-1].set_xlabel("timestamp (Unix seconds)")

    meta_lines = [
        f"episode_id: {episode_id or 'n/a'}",
        f"rows: {len(rows)}",
        f"unique holds: {len(np.unique(hold_index))}",
        f"camera present: {has_camera}",
        (
            f"camera valid rows: {int(np.count_nonzero(camera_valid))}/{len(camera_valid)}"
            if has_camera
            else "camera valid rows: n/a"
        ),
        "position units: centimeters for plotting; deltas are relative to first sample",
    ]
    top = "\n".join(meta_lines)
    fig.suptitle(title or "Unified static system-ID viewer", fontsize=14)
    fig.text(0.01, 0.995, top, ha="left", va="top", fontsize=9, family="monospace")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Unified Parquet to visualize")
    parser.add_argument("--save", type=Path, default=None, help="Save figure instead of showing it")
    parser.add_argument("--title", default=None, help="Custom plot title")
    args = parser.parse_args()

    data = _load_plot_data(args.input)
    fig = plot_static_sysid(data, title=args.title)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=200)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
