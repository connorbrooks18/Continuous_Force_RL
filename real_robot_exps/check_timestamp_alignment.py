"""Check timestamp alignment inside a unified static system-ID Parquet file.

This inspects the timestamps already stored in the unified episode file:
- robot sample timestamp
- selected camera timestamp
- robot-camera offset
- camera window bounds
- per-hold timing consistency

Example:
    python -m real_robot_exps.check_timestamp_alignment --input pull_unified.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def _read_dataset_metadata(path: Path) -> dict[str, Any]:
    raw = pq.read_schema(path).metadata or {}
    payload = raw.get(b"dataset_metadata")
    if payload is None:
        return {}
    return json.loads(payload.decode("utf-8"))


def _as_float_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _list_len(values: Any) -> int:
    if values is None:
        return 0
    try:
        return len(values)
    except TypeError:
        return 0


def _summarize(name: str, values: np.ndarray, unit: str = "s") -> None:
    if values.size == 0:
        print(f"{name}: no samples")
        return
    print(
        f"{name}: mean={values.mean():+.6f}{unit}, std={values.std():.6f}{unit}, "
        f"median={np.median(values):+.6f}{unit}, min/max={values.min():+.6f}{unit}/{values.max():+.6f}{unit}"
    )


def check_alignment(path: Path) -> None:
    table = pq.read_table(path)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"No rows found in {path}")

    metadata = _read_dataset_metadata(path)
    print(f"file: {path}")
    print(f"episode_id: {metadata.get('episode_id', rows[0].get('episode_id', ''))}")
    print(f"rows: {len(rows)}")
    print(f"schema_name: {metadata.get('schema_name', 'n/a')}")
    print()

    robot_ts = _as_float_array(rows, "timestamp")
    _summarize("robot timestamp spacing", np.diff(robot_ts))

    if "camera_timestamp" not in rows[0]:
        print("\nNo camera fields present in this Parquet.")
        return

    camera_ts = _as_float_array(rows, "camera_timestamp")
    offset = _as_float_array(rows, "robot_camera_timestamp_offset_s")
    window_start = _as_float_array(rows, "camera_window_start_timestamp")
    window_end = _as_float_array(rows, "camera_window_end_timestamp")
    camera_valid = np.asarray([bool(row.get("camera_data_valid", False)) for row in rows], dtype=bool)

    print("camera validity:")
    print(f"  valid rows: {int(camera_valid.sum())}/{len(camera_valid)}")
    print(f"  invalid rows: {int((~camera_valid).sum())}/{len(camera_valid)}")
    print()

    _summarize("robot-camera offset", offset)
    _summarize("robot minus camera timestamp", robot_ts - camera_ts)
    _summarize("camera window duration", window_end - window_start)
    _summarize("robot timestamp minus window start", robot_ts - window_start)
    _summarize("window end minus robot timestamp", window_end - robot_ts)

    selected_offsets: list[float] = []
    for row in rows:
        row_ts = float(row["timestamp"])
        selected = row.get("camera_selected_timestamps", [])
        if not selected:
            continue
        selected_offsets.extend(float(x) - row_ts for x in selected)
    if selected_offsets:
        _summarize("selected camera timestamp minus row robot timestamp", np.asarray(selected_offsets, dtype=np.float64))
    else:
        print("selected camera timestamps: none")

    hold_indices = sorted({int(row["hold_index"]) for row in rows if "hold_index" in row})
    print()
    print(f"holds observed: {hold_indices}")
    for hold_idx in hold_indices:
        hold_rows = [row for row in rows if int(row.get("hold_index", -1)) == hold_idx]
        if not hold_rows:
            continue
        hold_robot_ts = _as_float_array(hold_rows, "timestamp")
        hold_camera_ts = _as_float_array(hold_rows, "camera_timestamp") if "camera_timestamp" in hold_rows[0] else np.array([])
        hold_offset = _as_float_array(hold_rows, "robot_camera_timestamp_offset_s") if "robot_camera_timestamp_offset_s" in hold_rows[0] else np.array([])
        hold_selected = [float(x) for row in hold_rows for x in row.get("camera_selected_timestamps", [])]
        print(f"hold {hold_idx}:")
        _summarize("  robot span", np.array([hold_robot_ts.max() - hold_robot_ts.min()]))
        if hold_offset.size:
            _summarize("  robot-camera offset", hold_offset)
        if hold_camera_ts.size:
            _summarize("  robot minus camera timestamp", hold_robot_ts - hold_camera_ts)
        if hold_selected:
            hold_selected_offsets = []
            for row in hold_rows:
                row_ts = float(row["timestamp"])
                selected = row.get("camera_selected_timestamps", [])
                hold_selected_offsets.extend(float(x) - row_ts for x in selected)
            _summarize(
                "  selected camera timestamp minus row robot timestamp",
                np.asarray(hold_selected_offsets, dtype=np.float64),
            )

    print()
    if np.abs(offset).size:
        large = np.where(np.abs(offset) > 0.1)[0]
        if large.size:
            print(f"warning: {large.size} rows have |robot_camera_timestamp_offset_s| > 0.1 s")
        else:
            print("no large robot-camera offset outliers over 0.1 s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Unified Parquet file to inspect")
    args = parser.parse_args()
    check_alignment(args.input)


if __name__ == "__main__":
    main()
