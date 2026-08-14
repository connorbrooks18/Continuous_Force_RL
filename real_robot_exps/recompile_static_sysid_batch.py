"""Recompile all static system-ID episodes for one structure.

Examples:
    # Safe: write refreshed files under recompiled/s11/.
    python -m real_robot_exps.recompile_static_sysid_batch --structure s11

    # Replace the compiled files in the current directory.
    python -m real_robot_exps.recompile_static_sysid_batch \
        --structure s11 --output-dir . --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from real_robot_exps.compile_static_sysid import compile_static_episode


ROBOT_RE = re.compile(r"^(?P<episode>[^/]+)_robot\.parquet$")
BASELINE_RE = re.compile(
    r"^(?P<structure>s\d+)_pull_theta(?P<theta>[-+0-9.]+)_phi(?P<phi>[-+0-9.]+)_"
    r"kp[^_]+_baseline_robot\.parquet$"
)


def _metadata(path: Path) -> dict:
    raw = pq.read_schema(path).metadata or {}
    payload = raw.get(b"dataset_metadata")
    if payload is None:
        return {}
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _episode_key(metadata: dict) -> tuple[str, float, float] | None:
    structure = metadata.get("structure")
    theta = metadata.get("theta_rad")
    phi = metadata.get("phi_rad")
    if structure is None or theta is None or phi is None:
        return None
    try:
        return str(structure), round(float(theta), 2), round(float(phi), 2)
    except (TypeError, ValueError):
        return None


def _baseline_index(baseline_dir: Path, structure: str) -> dict[tuple[str, float, float], Path]:
    index: dict[tuple[str, float, float], Path] = {}
    for path in sorted(baseline_dir.glob(f"{structure}_pull*_baseline_robot.parquet")):
        key = _episode_key(_metadata(path))
        if key is None:
            match = BASELINE_RE.match(path.name)
            if match:
                key = (
                    match.group("structure"),
                    round(float(match.group("theta")), 2),
                    round(float(match.group("phi")), 2),
                )
        if key is not None:
            index[key] = path
    return index


def recompile_structure(
    structure: str,
    *,
    input_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    overwrite: bool,
    require_baseline: bool,
    camera_ema_alpha: float,
) -> list[Path]:
    robot_paths = sorted(input_dir.glob(f"{structure}-d*_robot.parquet"))
    if not robot_paths:
        raise FileNotFoundError(f"No {structure}-d*_robot.parquet files in {input_dir}")
    baselines = _baseline_index(baseline_dir, structure)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for robot_path in robot_paths:
        match = ROBOT_RE.match(robot_path.name)
        assert match is not None
        episode = match.group("episode")
        tracking_path = input_dir / f"{episode}_tracking.parquet"
        if not tracking_path.exists():
            raise FileNotFoundError(f"Missing tracking file for {robot_path.name}: {tracking_path}")
        output_path = output_dir / f"{episode}.parquet"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_path}; use --overwrite or another --output-dir"
            )

        robot_metadata = _metadata(robot_path)
        key = _episode_key(robot_metadata)
        if key is None:
            try:
                key = (
                    structure,
                    round(float(robot_metadata["theta_rad"]), 2),
                    round(float(robot_metadata["phi_rad"]), 2),
                )
            except (KeyError, TypeError, ValueError):
                key = None
        baseline_path = baselines.get(key) if key is not None else None
        if require_baseline and baseline_path is None:
            raise FileNotFoundError(
                f"No baseline found for {robot_path.name} "
                f"(metadata key={key!r}) in {baseline_dir}"
            )
        print(
            f"Compiling {robot_path.name} with "
            f"{baseline_path.name if baseline_path else 'no baseline'} -> {output_path}",
            flush=True,
        )
        compile_static_episode(
            robot_path,
            tracking_path,
            output_path,
            camera_ema_alpha=camera_ema_alpha,
            baseline_path=baseline_path,
            command_argv=sys.argv,
        )
        png_path = output_path.with_suffix(".png")
        viz_cmd = [
            sys.executable,
            "-m",
            "real_robot_exps.viz_static_sysid",
            "--input",
            str(output_path),
            "--save",
            str(png_path),
            "--no-show",
        ]
        print(" ".join(viz_cmd), flush=True)
        try:
            subprocess.run(viz_cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] Visualization failed for {output_path}: {exc}")
        outputs.append(output_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, help="Structure prefix, e.g. s11 or s09")
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--baseline-dir", type=Path, default=Path("baselines"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("recompiled"),
        help="Output directory; defaults to a safe recompiled/ directory",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-baseline", action="store_true")
    parser.add_argument("--camera-ema-alpha", type=float, default=1.0)
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir == Path("recompiled"):
        output_dir = output_dir / args.structure
    recompile_structure(
        args.structure,
        input_dir=args.input_dir,
        baseline_dir=args.baseline_dir,
        output_dir=output_dir,
        overwrite=args.overwrite,
        require_baseline=args.require_baseline,
        camera_ema_alpha=args.camera_ema_alpha,
    )


if __name__ == "__main__":
    main()
