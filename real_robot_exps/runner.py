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

from real_robot_exps.camera_snapshot import update_pre_grasp_geometry_with_snapshots
from real_robot_exps.metadata_cache import (
    load_pre_grasp_metadata_cache,
    write_pre_grasp_metadata_cache,
)


PART_ORDER = ("primary", "spur", "stem", "apple")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _unique_path(path: Path) -> Path:
    """Return a non-colliding path by appending -01, -02, ... before suffix."""
    if not path.exists():
        return path
    suffix = path.suffix
    stem = path.stem
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}-{idx:02d}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _run_output_path(path: Path) -> Path:
    return _unique_path(path)


def _resolve_metadata_cache_path(args) -> Path:
    cache_path = getattr(args, "metadata_cache", Path("metadata_cache.json"))
    if cache_path.is_absolute():
        return cache_path
    return args.output_dir / cache_path



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


def _normalized_pre_grasp_geometry(structure_index: int, structure: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "structure_index": int(structure_index),
        "structure_name": structure.get("name", f"structure_{int(structure_index):02d}"),
        "angles_source": structure.get("angles_source", ""),
        "geometry_source": structure.get("geometry_source", ""),
        "note": "Manual structure catalog plus a lengthened camera snapshot for angle/length estimation.",
        "parts": out,
        "snapshot": {},
        "settled_snapshot": {},
        "under_gravity_snapshot": {},
        "lengthened_snapshot": {},
    }


def _baseline_path_for_direction(args, structure_index: int, direction: dict[str, Any]) -> Path:
    return args.output_dir / (
        f"s{structure_index:02d}_pull_theta{direction['theta']:.2f}_phi{direction['phi']:.2f}_kp{args.kp:.0f}_baseline_robot.parquet"
    )


def _build_run_metadata(
    *,
    structure_index: int,
    structure: dict[str, Any],
    direction_index: int,
    direction: dict[str, Any],
    pre_grasp_geometry: dict[str, Any],
    kp: float,
    manual_setup: bool,
    debug_pre_grasp: bool,
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
        "manual_setup": bool(manual_setup),
        "debug_pre_grasp": bool(debug_pre_grasp),
        "dump": {
            "structure_catalog_entry": structure,
            "direction_entry": direction,
            "note": "structure index selects the manual geometry package; angles and segment lengths come from the lengthened-state capture prompt",
        },
    }


def _capture_snapshot_via_subprocess() -> dict[str, Any]:
    import tempfile

    with tempfile.NamedTemporaryFile(prefix="camera_snapshot_", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    cmd = [
        sys.executable,
        "-m",
        "real_robot_exps.camera_snapshot",
        "--output",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        return _load_json(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


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
    expected_baseline = _baseline_path_for_direction(args, structure_index, direction)
    robot_path = _run_output_path(args.output_dir / f"{run_id}_robot.parquet")
    tracking_path = _run_output_path(args.output_dir / f"{run_id}_tracking.parquet")
    unified_path = _run_output_path(args.output_dir / f"{run_id}.parquet")
    metadata_path = _run_output_path(args.output_dir / f"{run_id}_metadata.tmp.json")
    post_snapshot_request_path = _run_output_path(args.output_dir / f"{run_id}_post_grasp_camera.request")
    post_snapshot_output_path = _run_output_path(args.output_dir / f"{run_id}_post_grasp_camera.json")
    detector_proc = None
    baseline_path_for_collect = expected_baseline

    run_metadata = _build_run_metadata(
        structure_index=structure_index,
        structure=structure,
        direction_index=direction_index,
        direction=direction,
        pre_grasp_geometry=pre_grasp_geometry,
        kp=float(args.kp),
        manual_setup=bool(args.manual_setup),
        debug_pre_grasp=bool(args.debug_pre_grasp),
    )
    _write_json(metadata_path, run_metadata)
    baseline_cmd = None
    detector_cmd = None
    robot_cmd = None
    compile_cmd = None
    viz_cmd = None

    if args.start_detector:
        detector_cmd = [
            sys.executable,
            str(args.detector_script),
            "--output",
            str(tracking_path),
            "--headless",
            "--snapshot-request",
            str(post_snapshot_request_path),
            "--snapshot-output",
            str(post_snapshot_output_path),
        ]
        if args.detector_extra_args:
            detector_cmd.extend(args.detector_extra_args)
        print(" ".join(detector_cmd))
        detector_proc = subprocess.Popen(detector_cmd)

    robot_cmd = [
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
        str(baseline_path_for_collect),
        "--run-metadata-file",
        str(metadata_path),
    ]
    if args.only_metadata:
        robot_cmd.append("--only-metadata")
    if args.manual_setup:
        robot_cmd.append("--manual-setup")
    if args.debug_pre_grasp:
        robot_cmd.append("--debug-pre-grasp")
    if args.mock_gripper:
        robot_cmd.append("--mock-gripper")
    if args.start_detector:
        robot_cmd.extend([
            "--post-grasp-camera-request",
            str(post_snapshot_request_path),
            "--post-grasp-camera-output",
            str(post_snapshot_output_path),
        ])

    print(f"\n=== Running {run_id} ===")
    print(" ".join(robot_cmd))
    start = time.time()
    try:
        subprocess.run(robot_cmd, check=True)
    finally:
        if detector_proc is not None:
            detector_proc.send_signal(signal.SIGINT)
            try:
                detector_proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                detector_proc.kill()
                detector_proc.wait(timeout=10.0)
    post_snapshot_request_path.unlink(missing_ok=True)
    post_snapshot_output_path.unlink(missing_ok=True)
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
                latest_start = max(robot_min, tracking_min)
                earliest_end = min(robot_max, tracking_max)
                # Inclusive overlap is the right rule here. Metadata-only runs
                # can have exactly one robot timestamp that still lies inside
                # the camera interval even though the overlap length is zero.
                if latest_start > earliest_end:
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
                    "--camera-ema-alpha",
                    str(args.camera_ema_alpha),
                ]
                print(" ".join(compile_cmd))
                subprocess.run(compile_cmd, check=True)
                viz_png = unified_path.with_suffix(".png")
                viz_cmd = [
                    sys.executable,
                    "-m",
                    "real_robot_exps.viz_static_sysid",
                    "--input",
                    str(unified_path),
                    "--save",
                    str(viz_png),
                    "--no-show"
                ]
                print(" ".join(viz_cmd))
                try:
                    subprocess.run(viz_cmd, check=True)
                except subprocess.CalledProcessError as exc:
                    print(f"[WARN] Visualization failed for {unified_path}: {exc}")
            else:
                print(f"[WARN] Expected tracking file not found within timeout: {tracking_path}")
    run_record = {
        "run_id": run_id,
        "direction_index": direction_index,
        "commands": {
            "baseline": baseline_cmd,
            "detector": detector_cmd,
            "robot": robot_cmd,
            "compile": compile_cmd,
            "viz": viz_cmd,
        },
        "files": {
            "baseline": str(baseline_path_for_collect) if baseline_path_for_collect.exists() else None,
            "robot": str(robot_path),
            "tracking": str(tracking_path) if tracking_path.exists() else None,
            "unified": str(unified_path) if unified_path.exists() else None,
        },
    }
    manifest_runs.append(run_record)


def _run_missing_baselines(
    *,
    structure_index: int,
    structure: dict[str, Any],
    directions: list[dict[str, Any]],
    start_at: int,
    args,
    pre_grasp_geometry: dict[str, Any],
) -> None:
    missing = []
    for direction_index, direction in enumerate(directions[start_at:], start=start_at):
        baseline_path = _baseline_path_for_direction(args, structure_index, direction)
        if not baseline_path.exists():
            missing.append((direction_index, direction, baseline_path))
    if not missing:
        return

    print("Missing baseline files:")
    for _, direction, baseline_path in missing:
        print(f"  {baseline_path}")

    input(
        "Remove the apple/contact load once, then press Enter to run all missing baselines..."
    )

    for direction_index, direction, baseline_path in missing:
        metadata_path = _run_output_path(
            args.output_dir / f"s{structure_index:02d}-d{direction_index:02d}_metadata.tmp.json"
        )
        run_metadata = _build_run_metadata(
            structure_index=structure_index,
            structure=structure,
            direction_index=direction_index,
            direction=direction,
            pre_grasp_geometry=pre_grasp_geometry,
            kp=float(args.kp),
            debug_pre_grasp=bool(args.debug_pre_grasp),
            manual_setup=bool(args.manual_setup),
        )
        _write_json(metadata_path, run_metadata)
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
            str(len(directions)),
            "--robot-output",
            str(baseline_path),
            "--run-metadata-file",
            str(metadata_path),
        ]
        if args.debug_pre_grasp:
            baseline_cmd.append("--debug-pre-grasp")
        if args.mock_gripper:
            baseline_cmd.append("--mock-gripper")
        print("\n=== Running baseline ===")
        print(" ".join(baseline_cmd))
        subprocess.run(baseline_cmd, check=True)
        if metadata_path.exists():
            metadata_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexed runner for apple-pull system-ID collection")
    parser.add_argument("--structures", type=Path, default=Path("real_robot_exps/structures.json"))
    parser.add_argument("--directions", type=Path, default=Path("real_robot_exps/directions.json"))
    parser.add_argument("--structure-index", type=int, default=None)
    parser.add_argument(
        "--start-at",
        type=int,
        default=0,
        help="Zero-based direction index to start at; runs this direction and all later ones",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("manifest.json"))
    parser.add_argument(
        "--metadata-cache",
        type=Path,
        default=Path("metadata_cache.json"),
        help="Cache file for measured pre-grasp metadata; relative paths are resolved under --output-dir",
    )
    parser.add_argument("--config", type=Path, default=Path("real_robot_exps/config.yaml"))
    parser.add_argument("--mode", choices=["collect", "baseline"], default="collect")
    parser.add_argument("--kp", type=float, default=100.0)
    parser.add_argument("--distance", type=float, default=0.04)
    parser.add_argument("--stops", type=int, default=4)
    parser.add_argument(
        "--expect-tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile unified output when tracking is available (default: true)",
    )
    parser.add_argument(
        "--start-detector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch at-tracking/Detecting.py before each robot run (default: true)",
    )
    parser.add_argument("--detector-script", type=Path, default=Path("at-tracking/Detecting.py"))
    parser.add_argument("--detector-extra-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--camera-ema-alpha", type=float, default=1.0, help="EMA alpha for camera smoothing during compile; 1.0 disables smoothing")
    parser.add_argument(
        "--only-metadata",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect reconstruction metadata only; skip baseline generation and pull trajectory",
    )
    parser.add_argument(
        "--manual-setup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward manual pull-start setup to apple_pullto_static so the arm can be positioned by hand",
    )
    parser.add_argument(
        "--debug-pre-grasp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print pre-grasp apple and TCP positions during each run",
    )
    parser.add_argument(
        "--mock-gripper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward a no-op gripper client to apple_pullto_static so no gripper connection is attempted",
    )
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
    if not 0 <= args.start_at < len(directions):
        raise SystemExit(f"--start-at must be in [0, {len(directions)})")

    if args.structure_index is None:
        print("\nAvailable structures:")
        for idx, structure in enumerate(structures):
            print(f"  {idx}: {structure.get('name', f'structure_{idx:02d}')}")
        selected = input("Structure index [0]: ").strip()
        if not selected:
            structure_index = 0
        else:
            structure_index = int(selected)
    else:
        structure_index = int(args.structure_index)

    if not 0 <= structure_index < len(structures):
        raise SystemExit(f"--structure-index must be in [0, {len(structures)})")

    structure = structures[structure_index]
    pre_grasp_geometry = _normalized_pre_grasp_geometry(structure_index, structure)
    metadata_cache_path = _resolve_metadata_cache_path(args)
    cached_pre_grasp_geometry = None
    try:
        cached_pre_grasp_geometry = load_pre_grasp_metadata_cache(
            metadata_cache_path,
            structure_index=structure_index,
            structure=structure,
        )
    except ValueError as exc:
        print(f"[WARN] Ignoring invalid metadata cache {metadata_cache_path}: {exc}")
    if cached_pre_grasp_geometry is not None:
        pre_grasp_geometry = cached_pre_grasp_geometry
        print(f"Using cached pre-grasp metadata from {metadata_cache_path}")

    if args.mode == "collect" and cached_pre_grasp_geometry is None:
        input("Let the apple/structure settle naturally under gravity, then press Enter to capture that snapshot...")
        under_gravity_snapshot = _capture_snapshot_via_subprocess()
        input(
            "Now lengthen the apple/structure so the connection angles and segment lengths are visible, "
            "then press Enter to capture the lengthened snapshot..."
        )
        lengthened_snapshot = _capture_snapshot_via_subprocess()
        pre_grasp_geometry = update_pre_grasp_geometry_with_snapshots(
            pre_grasp_geometry,
            lengthened_snapshot=lengthened_snapshot,
            settled_snapshot=under_gravity_snapshot,
            under_gravity_snapshot=under_gravity_snapshot,
        )
        write_pre_grasp_metadata_cache(
            metadata_cache_path,
            structure_index=structure_index,
            structure=structure,
            pre_grasp_geometry=pre_grasp_geometry,
        )
        print(f"Wrote cached pre-grasp metadata to {metadata_cache_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "collect" and not args.only_metadata:
        _run_missing_baselines(
            structure_index=structure_index,
            structure=structure,
            directions=directions,
            start_at=args.start_at,
            args=args,
            pre_grasp_geometry=pre_grasp_geometry,
        )

    manifest_runs: list[dict[str, Any]] = []
    for direction_index, direction in enumerate(directions[args.start_at:], start=args.start_at):
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
        "structure_name": structure.get("name", f"structure_{structure_index:02d}"),
        "start_at": int(args.start_at),
        "structures_source": str(args.structures),
        "directions_source": str(args.directions),
        "runs": manifest_runs,
    }
    manifest_path = _run_output_path(args.manifest)
    _write_json(manifest_path, manifest)
    print(f"\nWrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
