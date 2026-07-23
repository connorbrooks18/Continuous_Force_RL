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


def _phase_spans(rows: list[dict[str, Any]]) -> list[tuple[float, float, int]]:
    if not rows:
        return []
    spans: list[tuple[float, float, int]] = []
    start = float(rows[0]["timestamp"])
    last_phase = int(rows[0].get("phase", 0))
    prev_t = start
    for row in rows[1:]:
        t = float(row["timestamp"])
        phase = int(row.get("phase", 0))
        if phase != last_phase:
            spans.append((start, prev_t, last_phase))
            start = t
            last_phase = phase
        prev_t = t
    spans.append((start, float(rows[-1]["timestamp"]), last_phase))
    return spans


def _shade_phase_background(ax, rows: list[dict[str, Any]]) -> None:
    for start, end, phase in _phase_spans(rows):
        if end <= start:
            end = start + 1e-9
        if phase == 0:
            color = "#f2f2f2"  # light gray for moving
        else:
            color = "#fce4ec"  # light pink for holding
        ax.axvspan(start, end, color=color, alpha=0.35, zorder=0)


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
    tcp_pos = _vector_columns(rows, "tcp_pos")
    has_camera = data.has_camera
    if has_camera:
        robot_camera_offset = np.asarray([row["robot_camera_timestamp_offset_s"] for row in rows], dtype=np.float64)
    else:
        robot_camera_offset = np.zeros(len(rows), dtype=np.float64)
    if has_camera:
        apple_pos = _vector_columns(rows, "apple_pos")
        start_pos = _vector_columns(rows, "woody_part_start_pos").reshape(len(rows), 3, 3)
        end_pos = _vector_columns(rows, "woody_part_end_pos").reshape(len(rows), 3, 3)
        bend = _vector_columns(rows, "woody_bending_angles")
        camera_valid = np.asarray([row["camera_data_valid"] for row in rows], dtype=bool)
    else:
        apple_pos = start_pos = end_pos = bend = None
    hold_index = np.asarray([row["hold_index"] for row in rows], dtype=int)
    episode_id = _episode_id_from_metadata(data.metadata, rows)

    n_panels = 8 if has_camera else 5
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 4 * n_panels), sharex=True, constrained_layout=True)
    for ax in axes:
        _shade_phase_background(ax, rows)

    _plot_vector_panel(axes[0], t, ft, ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"], "Wrist wrench")
    force_magnitude = np.linalg.norm(ft[:, :3], axis=1)
    torque_magnitude = np.linalg.norm(ft[:, 3:], axis=1)
    axes[1].plot(t, force_magnitude, label=r"$\|F\|$ [N]", linewidth=1.5)
    axes[1].plot(t, torque_magnitude, label=r"$\|T\|$ [N m]", linewidth=1.5)
    axes[1].set_title("Wrist wrench magnitudes")
    axes[1].set_ylabel("magnitude")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, robot_camera_offset, color="tab:gray", linewidth=1.5, label="robot-camera timestamp offset")
    axes[2].axhline(0.0, color="k", linewidth=1, alpha=0.2)
    if robot_camera_offset.size:
        worst_idx = int(np.argmax(np.abs(robot_camera_offset)))
        axes[2].scatter([t[worst_idx]], [robot_camera_offset[worst_idx]], color="tab:red", s=35, zorder=3,
                        label=f"max |offset| {abs(robot_camera_offset[worst_idx]):.3f}s")
    axes[2].set_title("Robot-camera timestamp offset")
    axes[2].set_ylabel("seconds")
    if not has_camera:
        axes[2].text(
            0.02,
            0.90,
            "No camera timing fields in this file",
            transform=axes[2].transAxes,
            fontsize=9,
            color="tab:red",
            va="top",
        )
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right", fontsize=8)

    if has_camera:
        tcp_pos_cm = tcp_pos * 100.0
        apple_pos_cm = apple_pos * 100.0
        start_pos_cm = start_pos * 100.0
        end_pos_cm = end_pos * 100.0
        tcp_pos_delta_cm = _delta_cm(tcp_pos)
        apple_pos_delta_cm = _delta_cm(apple_pos)
        apple_tcp_delta_cm = (apple_pos - tcp_pos) * 100.0
        tcp_travel_cm = np.linalg.norm(tcp_pos - tcp_pos[0], axis=1) * 100.0
        apple_travel_cm = np.linalg.norm(apple_pos - apple_pos[0], axis=1) * 100.0
        apple_tcp_dist_cm = np.linalg.norm(apple_pos - tcp_pos, axis=1) * 100.0
        start_pos_delta_cm = _delta_cm(start_pos.reshape(len(rows), -1)).reshape(len(rows), 3, 3)
        end_pos_delta_cm = _delta_cm(end_pos.reshape(len(rows), -1)).reshape(len(rows), 3, 3)

        axes[3].plot(t, tcp_pos_cm[:, 0], label="tcp x")
        axes[3].plot(t, tcp_pos_cm[:, 1], label="tcp y")
        axes[3].plot(t, tcp_pos_cm[:, 2], label="tcp z")
        axes[3].plot(t, apple_pos_cm[:, 0], "--", label="apple x")
        axes[3].plot(t, apple_pos_cm[:, 1], "--", label="apple y")
        axes[3].plot(t, apple_pos_cm[:, 2], "--", label="apple z")
        axes[3].set_title("Absolute positions")
        axes[3].set_ylabel("cm")
        axes[3].grid(True, alpha=0.25)
        axes[3].legend(loc="upper right", ncol=3, fontsize=8)

        axes[4].plot(t, tcp_pos_delta_cm[:, 0], label="tcp x")
        axes[4].plot(t, tcp_pos_delta_cm[:, 1], label="tcp y")
        axes[4].plot(t, tcp_pos_delta_cm[:, 2], label="tcp z")
        axes[4].plot(t, apple_pos_delta_cm[:, 0], "--", label="apple x")
        axes[4].plot(t, apple_pos_delta_cm[:, 1], "--", label="apple y")
        axes[4].plot(t, apple_pos_delta_cm[:, 2], "--", label="apple z")
        axes[4].set_title("Position deltas from first sample")
        axes[4].set_ylabel("delta cm")
        axes[4].grid(True, alpha=0.25)
        axes[4].legend(loc="upper right", ncol=3, fontsize=8)

        axes[5].plot(t, apple_tcp_delta_cm[:, 0], label="apple-tcp x")
        axes[5].plot(t, apple_tcp_delta_cm[:, 1], label="apple-tcp y")
        axes[5].plot(t, apple_tcp_delta_cm[:, 2], label="apple-tcp z")
        axes[5].plot(t, apple_tcp_dist_cm, color="tab:purple", linewidth=1.8, label="apple-tcp distance")
        axes[5].plot(t, apple_travel_cm, "--", color="tab:orange", linewidth=1.6, label="apple travel from first sample")
        axes[5].plot(t, tcp_travel_cm, "--", color="tab:blue", linewidth=1.6, label="tcp travel from first sample")
        axes[5].set_title("Distances and travel")
        axes[5].set_ylabel("cm")
        axes[5].grid(True, alpha=0.25)
        axes[5].legend(loc="upper right", ncol=2, fontsize=8)

        axes[6].plot(t, start_pos_cm[:, 0, 0], label="Branch start x")
        axes[6].plot(t, start_pos_cm[:, 0, 1], label="Branch start y")
        axes[6].plot(t, start_pos_cm[:, 0, 2], label="Branch start z")
        axes[6].plot(t, end_pos_cm[:, 0, 0], "--", label="Branch end x")
        axes[6].plot(t, end_pos_cm[:, 0, 1], "--", label="Branch end y")
        axes[6].plot(t, end_pos_cm[:, 0, 2], "--", label="Branch end z")
        axes[6].set_title("Branch endpoint absolute positions")
        axes[6].set_ylabel("cm")
        axes[6].grid(True, alpha=0.25)
        axes[6].legend(loc="upper right", ncol=3, fontsize=8)

        axes[7].plot(t, start_pos_delta_cm[:, 0, 0], label="Branch start delta x")
        axes[7].plot(t, start_pos_delta_cm[:, 0, 1], label="Branch start delta y")
        axes[7].plot(t, start_pos_delta_cm[:, 0, 2], label="Branch start delta z")
        axes[7].plot(t, end_pos_delta_cm[:, 0, 0], "--", label="Branch end delta x")
        axes[7].plot(t, end_pos_delta_cm[:, 0, 1], "--", label="Branch end delta y")
        axes[7].plot(t, end_pos_delta_cm[:, 0, 2], "--", label="Branch end delta z")
        axes[7].plot(t, bend[:, 0], label="Branch bend", alpha=0.8)
        axes[7].plot(t, bend[:, 1], label="Spur bend", alpha=0.8)
        axes[7].plot(t, bend[:, 2], label="Apple bend", alpha=0.8)
        axes[7].set_title("Branch endpoint deltas and woody bending")
        axes[7].set_ylabel("delta cm / rad")
        axes[7].grid(True, alpha=0.25)
        axes[7].legend(loc="upper right", ncol=3, fontsize=8)
    else:
        axes[3].plot(t, tcp_pos[:, 0] * 100.0, label="tcp x")
        axes[3].plot(t, tcp_pos[:, 1] * 100.0, label="tcp y")
        axes[3].plot(t, tcp_pos[:, 2] * 100.0, label="tcp z")
        axes[3].set_title("TCP position")
        axes[3].set_ylabel("cm")
        axes[3].grid(True, alpha=0.25)
        axes[3].legend(loc="upper right", ncol=3, fontsize=8)

        axes[4].plot(t, _delta_cm(tcp_pos)[:, 0], label="tcp x")
        axes[4].plot(t, _delta_cm(tcp_pos)[:, 1], label="tcp y")
        axes[4].plot(t, _delta_cm(tcp_pos)[:, 2], label="tcp z")
        axes[4].set_title("TCP position deltas from first sample")
        axes[4].set_ylabel("delta cm")
        axes[4].grid(True, alpha=0.25)
        axes[4].legend(loc="upper right", ncol=3, fontsize=8)

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
        "plot focus: wrench, positions, deltas, and camera-derived geometry",
    ]
    top = "\n".join(meta_lines)
    fig.suptitle(title or "Unified static system-ID viewer", fontsize=14)
    fig.text(0.01, 0.995, top, ha="left", va="top", fontsize=9, family="monospace")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Unified Parquet to visualize")
    parser.add_argument("--save", type=Path, default=None, help="Save figure instead of showing it")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open an interactive Matplotlib window (use --no-show to disable)",
    )
    parser.add_argument("--title", default=None, help="Custom plot title")
    args = parser.parse_args()

    data = _load_plot_data(args.input)
    fig = plot_static_sysid(data, title=args.title)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=200)
        print(f"Saved figure to {args.save}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
