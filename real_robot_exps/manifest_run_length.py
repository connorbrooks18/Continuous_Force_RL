"""Compute run duration from a collection manifest.

This script reads a manifest written by ``real_robot_exps.runner`` and
estimates the overall wall-clock duration of the collection by taking the
first timestamp from the first run and the last timestamp from the last run.
It prints that span in minutes.

Usage:
    python -m real_robot_exps.manifest_run_length --manifest manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(manifest_path: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _read_timestamps(path: Path) -> list[float]:
    table = pq.read_table(path, columns=["timestamp"])
    rows = table.to_pylist()
    timestamps = [float(row["timestamp"]) for row in rows if row.get("timestamp") is not None]
    if not timestamps:
        raise ValueError(f"No timestamp values found in {path}")
    return timestamps


def _pick_run_file(manifest_path: Path, run: dict[str, Any]) -> Path:
    files = run.get("files", {})
    for key in ("unified", "robot"):
        candidate = _resolve_path(manifest_path, files.get(key))
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a usable Parquet file for run {run.get('run_id', '<unknown>')}"
    )


def _run_timestamp_bounds(manifest_path: Path, run: dict[str, Any]) -> tuple[float, float]:
    parquet_path = _pick_run_file(manifest_path, run)
    timestamps = _read_timestamps(parquet_path)
    return min(timestamps), max(timestamps)


def compute_manifest_duration(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("Manifest field 'runs' must be a list")

    first_run = runs[0]
    last_run = runs[-1]
    if not isinstance(first_run, dict) or not isinstance(last_run, dict):
        raise ValueError("Each run entry in the manifest must be a dictionary")

    first_timestamp, _ = _run_timestamp_bounds(manifest_path, first_run)
    _, last_timestamp = _run_timestamp_bounds(manifest_path, last_run)
    total_seconds = last_timestamp - first_timestamp

    return {
        "manifest": str(manifest_path),
        "n_runs": len(runs),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60.0,
        "first_run_id": first_run.get("run_id", ""),
        "last_run_id": last_run.get("run_id", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Manifest JSON file")
    args = parser.parse_args()

    summary = compute_manifest_duration(args.manifest)
    print(f"manifest: {summary['manifest']}")
    print(f"runs: {summary['n_runs']}")
    print(f"first run: {summary['first_run_id']} @ {summary['first_timestamp']:.6f}")
    print(f"last run: {summary['last_run_id']} @ {summary['last_timestamp']:.6f}")
    print(f"total: {summary['total_minutes']:.2f} min")


if __name__ == "__main__":
    main()
