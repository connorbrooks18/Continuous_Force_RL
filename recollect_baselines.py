"""Recollect and rebuild the saved episodes for structures s02 through s10.

For every ``*_robot.parquet`` in each structure directory, this script:

1. renames the existing baseline, unified Parquet, and visualization PNG to
   ``*_old`` names;
2. replays the robot run to collect a fresh baseline;
3. recompiles the unified Parquet using the existing tracking data; and
4. renders the visualization again.

The baseline filename is read from each directory's ``manifest.json`` because
it contains the direction-specific baseline name used by the original run.

Run from the repository root:

    python recollect_baselines.py

Use ``--dry-run`` to inspect all planned renames and commands before moving
anything or starting robot motion.

To resume from a checkpoint, for example structure ``s06`` direction ``d03``:

    python recollect_baselines.py --start-at-structure 6 --start-at-direction 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_STRUCTURE_DIRS = tuple(f"s{index:02d}" for index in range(2, 11))


@dataclass(frozen=True)
class Episode:
    structure_dir: Path
    robot: Path
    tracking: Path
    baseline: Path
    unified: Path
    visualization: Path


def _old_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_old{path.suffix}")


def _load_manifest(directory: Path) -> dict[str, dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing manifest: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {manifest_path}: {exc}") from exc

    records = payload.get("runs", [])
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        files = record.get("files", {})
        robot_name = Path(str(files.get("robot", ""))).name
        if robot_name:
            result[robot_name] = files
    return result


def _baseline_from_robot_metadata(robot: Path) -> Path:
    metadata = pq.read_schema(robot).metadata or {}
    payload = metadata.get(b"dataset_metadata")
    if payload is None:
        raise ValueError(f"No dataset metadata in {robot}; cannot determine baseline name")
    try:
        values = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid dataset metadata in {robot}") from exc
    try:
        structure = robot.parent.name
        theta = float(values["theta_rad"])
        phi = float(values["phi_rad"])
        kp = float(values["dump"]["robot_info"]["kp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing theta/phi/kp metadata in {robot}") from exc
    return robot.parent / (
        f"{structure}_pull_theta{theta:.2f}_phi{phi:.2f}_kp{kp:.0f}_baseline_robot.parquet"
    )


def discover_episodes(
    root: Path,
    structure_dirs: tuple[str, ...],
    *,
    start_at_structure: int,
    start_at_direction: int,
) -> list[Episode]:
    episodes: list[Episode] = []
    for directory_name in structure_dirs:
        structure_index = int(directory_name[1:])
        if structure_index < start_at_structure:
            continue
        directory = root / directory_name
        if not directory.is_dir():
            raise ValueError(f"Missing structure directory: {directory}")
        manifest_files = _load_manifest(directory)
        robot_files = sorted(
            path for path in directory.glob("*_robot.parquet")
            if not path.name.endswith("_baseline_robot.parquet")
        )
        if not robot_files:
            raise ValueError(f"No robot Parquet files found in {directory}")

        for robot in robot_files:
            direction_text = robot.stem.removesuffix("_robot").rsplit("-d", 1)[-1]
            direction_index = int(direction_text)
            if structure_index == start_at_structure and direction_index < start_at_direction:
                continue
            files = manifest_files.get(robot.name)
            baseline = (
                directory / Path(str(files["baseline"])).name
                if files is not None and files.get("baseline")
                else _baseline_from_robot_metadata(robot)
            )
            unified = robot.with_name(robot.name.removesuffix("_robot.parquet") + ".parquet")
            tracking = robot.with_name(robot.name.removesuffix("_robot.parquet") + "_tracking.parquet")
            visualization = unified.with_suffix(".png")
            episodes.append(Episode(directory, robot, tracking, baseline, unified, visualization))
    return episodes


def _validate(episodes: list[Episode], config: Path) -> None:
    if not config.is_file():
        raise ValueError(f"Config file does not exist: {config}")
    missing: list[Path] = []
    for episode in episodes:
        for path in (episode.robot, episode.tracking, episode.baseline, episode.unified, episode.visualization):
            if not path.is_file():
                missing.append(path)
        for path in (episode.baseline, episode.unified, episode.visualization):
            archived = _old_path(path)
            if archived.exists():
                raise ValueError(
                    f"Archive target already exists: {archived}; remove it or move it before retrying"
                )
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise ValueError(f"Required input/artifact files are missing:\n{formatted}")


def _print_episode(episode: Episode) -> None:
    print(f"\n[{episode.structure_dir.name}/{episode.robot.stem.removesuffix('_robot')}]\n"
          f"  archive: {episode.baseline.name} -> {_old_path(episode.baseline).name}\n"
          f"  archive: {episode.unified.name} -> {_old_path(episode.unified).name}\n"
          f"  archive: {episode.visualization.name} -> {_old_path(episode.visualization).name}")


def _run(command: list[str], root: Path) -> None:
    print("  $", " ".join(command))
    subprocess.run(command, cwd=root, check=True)


def process_episode(episode: Episode, root: Path, config: Path, camera_ema_alpha: float, dry_run: bool) -> None:
    _print_episode(episode)
    commands = [
        [sys.executable, "-m", "real_robot_exps.collect_joint_velocity_baseline",
         "--actual-robot", str(episode.robot), "--output", str(episode.baseline), "--config", str(config)],
        [sys.executable, "-m", "real_robot_exps.compile_static_sysid",
         "--robot", str(episode.robot), "--tracking", str(episode.tracking),
         "--output", str(episode.unified), "--camera-ema-alpha", str(camera_ema_alpha),
         "--baseline", str(episode.baseline)],
        [sys.executable, "-m", "real_robot_exps.viz_static_sysid",
         "--input", str(episode.unified), "--save", str(episode.visualization), "--no-show"],
    ]
    if dry_run:
        for command in commands:
            print("  $", " ".join(command))
        return

    for path in (episode.baseline, episode.unified, episode.visualization):
        shutil.move(str(path), str(_old_path(path)))
    for command in commands:
        _run(command, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current directory)")
    parser.add_argument("--config", type=Path, default=Path("real_robot_exps/config.yaml"))
    parser.add_argument("--camera-ema-alpha", type=float, default=1.0)
    parser.add_argument(
        "--start-at-structure",
        type=int,
        default=2,
        help="First structure number to process, inclusive (default: 2)",
    )
    parser.add_argument(
        "--start-at-direction",
        type=int,
        default=0,
        help="First direction number within the starting structure, inclusive (default: 0)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show renames and commands without changing files")
    args = parser.parse_args()

    if not 2 <= args.start_at_structure <= 10:
        parser.error("--start-at-structure must be between 2 and 10")
    if not 0 <= args.start_at_direction <= 7:
        parser.error("--start-at-direction must be between 0 and 7")

    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    episodes = discover_episodes(
        root,
        DEFAULT_STRUCTURE_DIRS,
        start_at_structure=args.start_at_structure,
        start_at_direction=args.start_at_direction,
    )
    _validate(episodes, config)
    print(f"Found {len(episodes)} episodes across {len(DEFAULT_STRUCTURE_DIRS)} structure directories.")
    if args.dry_run:
        print("Dry run: no files will be moved and no commands will run.")
    for episode in episodes:
        process_episode(episode, root, config, args.camera_ema_alpha, args.dry_run)


if __name__ == "__main__":
    main()
