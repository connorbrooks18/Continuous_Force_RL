"""Structured multi-run orchestrator for apple pull data collection.

This runner selects a structure by index, loads a direction list, and launches
one pull run per (structure, direction) pair. It writes:

- indexed raw robot files: s00-d00_robot.parquet
- optional raw tracking files: s00-d00_tracking.parquet
- unified compiled files: s00-d00.parquet
- a manifest.json describing what was collected

The robot execution itself still lives in ``apple_pullto_static.py``. This file
exists to keep collection clean, repeatable, and easy to scale.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


PART_ORDER = ("primary", "spur", "stem", "apple")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)




def _normalize_direction(entry: Any) -> dict[str, float | str]:
    if isinstance(entry, dict):
        if "theta" not in entry or "phi" not in entry:
            raise ValueError(f"Direction entry must contain theta and phi: {entry}")
        return {
            "theta": float(entry["theta"]),
            "phi": float(entry["phi"]),
            "name": str(entry.get("name", "")),
        }
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return {"theta": float(entry[0]), "phi": float(entry[1]), "name": ""}
    raise ValueError(f"Unsupported direction entry: {entry!r}")


def _normalized_pre_grasp_geometry(structure: dict[str, Any]) -> dict[str, Any]:
    parts = structure.get("parts", {})
    out: dict[str, Any] = {}
    for idx, part_name in enumerate(PART_ORDER):
        part = dict(parts.get(part_name, {}))
        part["connection_rpy_deg"] = [0.0, 0.0, 0.0] if idx == 0 else [
            float(x) for x in part.get("connection_rpy_deg", [0.0, 0.0, 0.0])
        ]
        if "mass_kg" in part and "density_kg_m3" not in part:
            part["density_kg_m3"] = part.pop("mass_kg")
        part["connection_source"] = "catalog" if idx == 0 else "catalog_or_lengthened_state_placeholder"
        out[part_name] = part
    return out


def _build_run_metadata(
    *,
    structure_index: int,
    structure: dict[str, Any],
    direction_index: int,
    direction: dict[str, Any],
    pre_grasp_geometry: dict[str, Any],
    kp: float,
) -> dict[str, Any]:
    return {
        "structure_index": structure_index,
        "structure_name": structure.get("name", f"structure_{structure_index:02d}"),
        "direction_index": direction_index,
        "direction_name": direction.get("name", ""),
        "theta": direction["theta"],
        "phi": direction["phi"],
        "robot_info": {"kp": float(kp)},
        "pre_grasp_geometry": pre_grasp_geometry,
        "post_grasp_geometry": {},
        "dump": {
            "structure_catalog_entry": structure,
            "direction_entry": direction,
            "note": "structure index selects the manual geometry package; angles are only derived from the lengthened-state check prompt",
        },
    }


def _run_one(
    *,
    structure_index: int,
    structure: dict[str, Any],
    direction_index: int,
    direction: dict[str, Any],
    num_directions: int,
    args,
    pre_grasp_geometry: dict[str, Any],
    manifest_runs: list[dict[str, Any]],
) -> None:
    run_id = f"s{structure_index:02d}-d{direction_index:02d}"
    base_label = f"pull_theta{direction['theta']:.2f}_phi{direction['phi']:.2f}"
    expected_baseline = args.output_dir / f"{base_label}_baseline_robot.parquet"
    robot_path = args.output_dir / f"{run_id}_robot.parquet"
    tracking_path = args.output_dir / f"{run_id}_tracking.parquet"
    unified_path = args.output_dir / f"{run_id}.parquet"
    metadata_path = args.output_dir / f"{run_id}_metadata.tmp.json"
    detector_proc = None

    for stale_path in (tracking_path, unified_path):
        if stale_path.exists():
            stale_path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()

    run_metadata = _build_run_metadata(
        structure_index=structure_index,
        structure=structure,
        direction_index=direction_index,
        direction=direction,
        pre_grasp_geometry=pre_grasp_geometry,
        kp=float(args.kp),
    )
    _write_json(metadata_path, run_metadata)

    if args.mode == "collect" and not expected_baseline.exists():
        print(
            f"[WARN] Baseline file not found: {expected_baseline}\n"
            "Remove the apple/contact load now, then press Enter to run the baseline pass."
        )
        input("Press Enter to start the baseline run...")
        baseline_cmd = [
            sys.executable,
            "-m",
            "real_robot_exps.apple_pullto_static",
            "--config",
            str(args.config),
            "--mode",
            "baseline",
            "--kp",
            str(args.kp),
            "--distance",
            str(args.distance),
            "--stops",
            str(args.stops),
            "--theta",
            str(direction["theta"]),
            "--phi",
            str(direction["phi"]),
            "--direction-index",
            str(direction_index),
            "--num-directions",
            str(num_directions),
            "--robot-output",
            str(expected_baseline),
            "--run-metadata-file",
            str(metadata_path),
        ]
        print("\n=== Running baseline ===")
        print(" ".join(baseline_cmd))
        subprocess.run(baseline_cmd, check=True)

    input(
        "Pull the apple so that the spur and stem are lengthened all the way, "
        "then press Enter to start the run..."
    )

    if args.start_detector:
        detector_cmd = [
            sys.executable,
            str(args.detector_script),
            "--output",
            str(tracking_path),
            "--headless",
        ]
        if args.detector_extra_args:
            detector_cmd.extend(args.detector_extra_args)
        print(" ".join(detector_cmd))
        detector_proc = subprocess.Popen(detector_cmd)

    cmd = [
        sys.executable,
        "-m",
        "real_robot_exps.apple_pullto_static",
        "--config",
        str(args.config),
        "--mode",
        args.mode,
        "--kp",
        str(args.kp),
        "--distance",
        str(args.distance),
        "--stops",
        str(args.stops),
        "--theta",
        str(direction["theta"]),
        "--phi",
        str(direction["phi"]),
        "--direction-index",
        str(direction_index),
        "--num-directions",
        str(num_directions),
        "--robot-output",
        str(robot_path),
        "--baseline-path",
        str(expected_baseline),
        "--run-metadata-file",
        str(metadata_path),
    ]

    print(f"\n=== Running {run_id} ===")
    print(" ".join(cmd))
    start = time.time()
    try:
        subprocess.run(cmd, check=True)
    finally:
        if detector_proc is not None:
            detector_proc.send_signal(signal.SIGINT)
            try:
                detector_proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                detector_proc.kill()
                detector_proc.wait(timeout=10.0)
    duration = time.time() - start
    if metadata_path.exists():
        metadata_path.unlink()

    skip_compile = False
    if args.expect_tracking:
        try:
            robot_rows = pq.read_table(robot_path, columns=["timestamp"]).to_pylist()
            robot_times = [float(row["timestamp"]) for row in robot_rows if "timestamp" in row and row["timestamp"] is not None]
            tracking_rows = pq.read_table(tracking_path, columns=["timestamp"]).to_pylist() if tracking_path.exists() else []
            tracking_times = [float(row["timestamp"]) for row in tracking_rows if "timestamp" in row and row["timestamp"] is not None]
            if robot_times and tracking_times:
                robot_min, robot_max = min(robot_times), max(robot_times)
                tracking_min, tracking_max = min(tracking_times), max(tracking_times)
                overlap = max(0.0, min(robot_max, tracking_max) - max(robot_min, tracking_min))
                if overlap <= 0.0:
                    print(
                        "[WARN] Robot and tracking timestamps do not overlap:\n"
                        f"       robot   {robot_min:.3f} -> {robot_max:.3f}\n"
                        f"       camera  {tracking_min:.3f} -> {tracking_max:.3f}\n"
                        "       Skipping unified compile for this run."
                    )
                    skip_compile = True
        except Exception as exc:
            print(f"[WARN] Could not check timestamp overlap before compile: {exc}")
        if (not skip_compile):
            deadline = time.time() + 300.0
            while time.time() < deadline and not tracking_path.exists():
                time.sleep(1.0)
            if tracking_path.exists():
                compile_cmd = [
                    sys.executable,
                    "-m",
                    "real_robot_exps.compile_static_sysid",
                    "--robot",
                    str(robot_path),
                    "--tracking",
                    str(tracking_path),
                    "--output",
                    str(unified_path),
                ]
                print(" ".join(compile_cmd))
                subprocess.run(compile_cmd, check=True)
            else:
                print(f"[WARN] Expected tracking file not found within timeout: {tracking_path}")
    run_record = {
        "run_id": run_id,
        "structure_index": structure_index,
        "structure_name": structure.get("name", f"structure_{structure_index:02d}"),
        "direction_index": direction_index,
        "direction": direction,
        "robot_parquet": str(robot_path),
        "tracking_parquet": str(tracking_path) if tracking_path.exists() else None,
        "unified_parquet": str(unified_path) if unified_path.exists() else None,
        "duration_sec": duration,
    }
    manifest_runs.append(run_record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexed runner for apple-pull system-ID collection")
    parser.add_argument("--structures", type=Path, default=Path("real_robot_exps/structures.json"))
    parser.add_argument("--directions", type=Path, default=Path("real_robot_exps/directions.json"))
    parser.add_argument("--structure-index", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("manifest.json"))
    parser.add_argument("--config", type=Path, default=Path("real_robot_exps/config.yaml"))
    parser.add_argument("--mode", choices=["collect", "baseline"], default="collect")
    parser.add_argument("--kp", type=float, default=80.0)
    parser.add_argument("--distance", type=float, default=0.04)
    parser.add_argument("--stops", type=int, default=4)
    parser.add_argument("--expect-tracking", action="store_true", help="Assume a tracking parquet will appear for each run")
    parser.add_argument("--start-detector", action="store_true", help="Launch at-tracking/Detecting.py before each robot run")
    parser.add_argument("--detector-script", type=Path, default=Path("at-tracking/Detecting.py"))
    parser.add_argument("--detector-extra-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    structures_payload = _load_json(args.structures)
    if isinstance(structures_payload, dict):
        structures = structures_payload.get("structures", [])
    else:
        structures = structures_payload
    if not structures:
        raise SystemExit(f"No structures found in {args.structures}")

    directions_payload = _load_json(args.directions)
    if isinstance(directions_payload, dict):
        directions_raw = directions_payload.get("directions", [])
    else:
        directions_raw = directions_payload
    directions = [_normalize_direction(entry) for entry in directions_raw]
    if not directions:
        raise SystemExit(f"No directions found in {args.directions}")

    if args.structure_index is None:
        print("\nAvailable structures:")
        for idx, structure in enumerate(structures):
            print(f"  {idx}: {structure.get('name', f'structure_{idx:02d}')}")
        selected = input("Structure index: ").strip()
        if not selected:
            raise SystemExit("A structure index is required.")
        structure_index = int(selected)
    else:
        structure_index = int(args.structure_index)

    if not 0 <= structure_index < len(structures):
        raise SystemExit(f"--structure-index must be in [0, {len(structures)})")

    structure = structures[structure_index]
    pre_grasp_geometry = _normalized_pre_grasp_geometry(structure)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_runs: list[dict[str, Any]] = []
    for direction_index, direction in enumerate(directions):
        _run_one(
            structure_index=structure_index,
            structure=structure,
            direction_index=direction_index,
            direction=direction,
            num_directions=len(directions),
            args=args,
            pre_grasp_geometry=pre_grasp_geometry,
            manifest_runs=manifest_runs,
        )

    manifest = {
        "structure_index": structure_index,
        "structure": structure,
        "directions_source": str(args.directions),
        "directions": directions,
        "pre_grasp_geometry": pre_grasp_geometry,
        "runs": manifest_runs,
    }
    _write_json(args.manifest, manifest)
    print(f"\nWrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
