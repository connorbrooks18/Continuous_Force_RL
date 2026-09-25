"""Field data collection: one guided, resumable session per orchard visit.

For every apple it walks the operator through:

    notes -> calibrate camera (board held by suction) -> place tags -> snapshots -> grasp
          -> pulls (all directions, one grasp) -> release -> baseline -> measurements

and stores everything for that apple in its own folder. Compilation (baseline
subtraction, tag->part radius correction, viz) is done later, at home:

    python -m real_robot_exps.field_session --session 2026-10-02_orchardA           # collect
    python -m real_robot_exps.field_session --session 2026-10-02_orchardA --apple A003   # resume A003
    python -m real_robot_exps.field_session --session 2026-10-02_orchardA --list
    python -m real_robot_exps.field_session --session 2026-10-02_orchardA --compile all   # at home

Layout (``--data-root``, default ~/field_data)::

    <session>/session.json                 settings, git commits, directions
    <session>/A003/apple.json              notes, step status, calibration verdict, parts, files
    <session>/A003/log.txt                 everything the subprocesses printed
    <session>/A003/calib/                  .calib, .samples, report.txt, camera_to_base.json
    <session>/A003/config/                 robot config + tracking_config.yaml used
    <session>/A003/snapshots/              under_gravity / lengthened / post_grasp_dXX (.json + .png)
    <session>/A003/tracking/               tracking_NN.parquet (detector, one per detector start)
    <session>/A003/pulls/                  dXX_robot.parquet, pull_plan.json, status.json
    <session>/A003/baseline/               dXX_baseline.parquet
    <session>/A003/compiled/               dXX.parquet + dXX.png (written by --compile)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
AT_TRACKING = REPO_ROOT / "at-tracking"
DEFAULT_CONFIG = REPO_ROOT / "real_robot_exps" / "config.yaml"
DEFAULT_DIRECTIONS = REPO_ROOT / "real_robot_exps" / "directions.json"
DEFAULT_TRACKING_CONFIG = AT_TRACKING / "tracking_config.yaml"
ROS_SETUP = "/opt/ros/humble/setup.bash"
DEFAULT_ROS_WS = Path.home() / "connor" / "franka_ros2_ws"
HANDEYE_DIR = Path.home() / ".ros2" / "easy_handeye2"

STEPS = (
    "notes",
    "calibrate",
    "tags",
    "snapshots",
    "grasp",
    "pulls",
    "baseline",
    "measurements",
)
STEP_TITLES = {
    "notes": "Notes about this fruiting system",
    "calibrate": "Camera calibration (ChArUco hand-eye)",
    "tags": "Place the tags",
    "snapshots": "Structure snapshots (under gravity, stretched)",
    "grasp": "Hand-guide the gripper onto the apple and grasp",
    "pulls": "Pull in every direction (one grasp)",
    "baseline": "Unloaded baseline replay",
    "measurements": "Measure and weigh the parts",
}


# =============================================================================
# small utilities
# =============================================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None}


def load_directions(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    entries = payload.get("directions", []) if isinstance(payload, dict) else payload
    directions = []
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            theta, phi, name = float(entry["theta"]), float(entry["phi"]), str(entry.get("name", ""))
        else:
            theta, phi, name = float(entry[0]), float(entry[1]), ""
        directions.append({"index": index, "theta": theta, "phi": phi, "name": name})
    if not directions:
        raise SystemExit(f"No directions in {path}")
    return directions


def compose_camera_to_base(base_to_camera_link: np.ndarray, camera_link_to_optical: np.ndarray) -> np.ndarray:
    """T_base_optical = T_base_camlink @ T_camlink_optical (AprilTag poses live in the optical frame)."""
    return np.asarray(base_to_camera_link, dtype=np.float64) @ np.asarray(camera_link_to_optical, dtype=np.float64)


# =============================================================================
# measurements
# =============================================================================

# (part, field, label, unit, plausible min, plausible max, required)
MEASUREMENT_FIELDS = (
    ("apple", "diameter_mm", "Apple diameter (widest)", "mm", 30.0, 120.0, True),
    ("apple", "height_mm", "Apple height (stem axis)", "mm", 30.0, 120.0, True),
    ("apple", "mass_g", "Apple mass", "g", 20.0, 600.0, True),
    ("stem", "length_mm", "Stem length", "mm", 2.0, 80.0, True),
    ("stem", "diameter_mm", "Stem diameter", "mm", 0.5, 8.0, True),
    ("stem", "mass_g", "Stem mass (blank = not weighed)", "g", 0.01, 10.0, False),
    ("spur", "length_mm", "Spur length", "mm", 5.0, 400.0, True),
    ("spur", "diameter_mm", "Spur diameter (at the tag)", "mm", 1.0, 30.0, True),
    ("spur", "mass_g", "Spur mass (blank = not weighed)", "g", 0.1, 300.0, False),
    ("primary", "diameter_mm", "Branch diameter (at the tag)", "mm", 5.0, 200.0, True),
    ("primary", "length_mm", "Branch free length (blank = unknown)", "mm", 10.0, 5000.0, False),
)
# Used only when a part was not weighed; the values match the lab structures.json defaults.
DEFAULT_DENSITY_KG_M3 = {"primary": 660.0, "spur": 1200.0, "stem": 1000.0}


def parts_from_measurements(raw: dict[str, dict[str, float | None]]) -> dict[str, dict[str, Any]]:
    """Convert operator measurements (mm, g) to the structures.json part layout (m, kg, kg/m^3)."""
    parts: dict[str, dict[str, Any]] = {}

    def cylinder(name: str, shape: str = "cylinder") -> None:
        values = raw.get(name, {})
        radius = float(values["diameter_mm"]) / 2000.0
        length = values.get("length_mm")
        entry: dict[str, Any] = {"shape": shape, "radius_m": radius, "measured": dict(values)}
        if length is not None:
            entry["length_m"] = float(length) / 1000.0
        mass = values.get("mass_g")
        if mass is not None:
            entry["mass_kg"] = float(mass) / 1000.0
            if length is not None:
                volume = math.pi * radius ** 2 * float(length) / 1000.0
                entry["density_kg_m3"] = entry["mass_kg"] / volume
        parts[name] = entry

    for name in ("stem", "spur", "primary"):
        cylinder(name)
        if "density_kg_m3" not in parts[name]:
            parts[name]["density_kg_m3"] = DEFAULT_DENSITY_KG_M3[name]
            parts[name]["density_source"] = "default (not weighed)"
        else:
            parts[name]["density_source"] = "measured mass / cylinder volume"

    apple = raw["apple"]
    radius = float(apple["diameter_mm"]) / 2000.0
    half_height = float(apple["height_mm"]) / 2000.0
    volume = 4.0 / 3.0 * math.pi * radius * radius * half_height  # spheroid
    parts["apple"] = {
        "shape": "sphere",
        "radius_m": radius,
        "length_m": 2.0 * half_height,
        "mass_kg": float(apple["mass_g"]) / 1000.0,
        "density_kg_m3": float(apple["mass_g"]) / 1000.0 / volume,
        "volume_model": "spheroid from diameter and height",
        "measured": dict(apple),
    }
    return parts


# =============================================================================
# the operator console
# =============================================================================

class Console:
    """All operator interaction goes through here so tests can script it."""

    def __init__(self, input_fn: Callable[[str], str] = input, print_fn: Callable[..., None] = print):
        self._input = input_fn
        self._print = print_fn

    def say(self, text: str = "") -> None:
        self._print(text)

    def banner(self, text: str) -> None:
        self._print("\n" + "=" * 72 + f"\n  {text}\n" + "=" * 72)

    def ask(self, prompt: str) -> str:
        return self._input(prompt).strip()

    def enter(self, prompt: str) -> None:
        self._input(f"{prompt} [Enter] ")

    def yes(self, prompt: str, default: bool = True) -> bool:
        suffix = "[Y/n]" if default else "[y/N]"
        while True:
            answer = self._input(f"{prompt} {suffix} ").strip().lower()
            if not answer:
                return default
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False

    def choose(self, prompt: str, choices: dict[str, str], default: str | None = None) -> str:
        keys = "/".join(choices)
        text = f"{prompt} (" + ", ".join(f"{k}={v}" for k, v in choices.items()) + f") [{keys}] "
        while True:
            answer = self._input(text).strip().lower()
            if not answer and default is not None:
                return default
            if answer in choices:
                return answer

    def number(self, prompt: str, unit: str, lo: float, hi: float, required: bool) -> float | None:
        while True:
            answer = self._input(f"  {prompt} [{unit}]: ").strip().replace(",", ".")
            if not answer:
                if not required:
                    return None
                self._print("    required")
                continue
            try:
                value = float(answer)
            except ValueError:
                self._print("    not a number")
                continue
            if not (lo <= value <= hi) and not self.yes(
                f"    {value:g} {unit} is outside the usual {lo:g}-{hi:g} {unit}. Keep it?", default=False
            ):
                continue
            return value


# =============================================================================
# processes
# =============================================================================

class Proc:
    """A background process in its own process group (so Ctrl-C/stop reach all children)."""

    def __init__(self, name: str, cmd: list[str], log_path: Path, *, cwd: Path | None = None, env=None):
        self.name = name
        self.cmd = cmd
        self.log_path = log_path
        self.log = log_path.open("a", encoding="utf-8")
        self.log.write(f"\n--- {now_utc()} start {name}: {' '.join(cmd)}\n")
        self.log.flush()
        self._log_offset = self.log.tell()
        self.popen = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True
        )

    def alive(self) -> bool:
        return self.popen.poll() is None

    def log_text(self) -> str:
        """What this process has written to the log so far."""
        with self.log_path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(self._log_offset)
            return stream.read()

    def stop(self, timeout_s: float = 20.0) -> int | None:
        if self.alive():
            for sig, wait in ((signal.SIGINT, timeout_s), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
                try:
                    os.killpg(self.popen.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.popen.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        code = self.popen.poll()
        self.log.write(f"--- {now_utc()} stop {self.name}: exit {code}\n")
        self.log.close()
        return code


def ros_command(inner: str, ros_ws: Path) -> list[str]:
    """Run a ROS 2 command with the humble + workspace overlays (system Python, not conda).

    Always pair with ``env=ros_env()``: sourcing ROS on top of an active conda env
    makes conda's libtiff shadow the system one, and rqt_image_view / cv_bridge die
    with ``libgdal.so.30: undefined symbol: TIFFReadRGBATileExt``.
    """
    return ["bash", "-c", f"source {ROS_SETUP} && source {ros_ws}/install/setup.bash && exec {inner}"]


# Variables a ROS process needs from the operator's session (display, DDS, locale);
# everything else, in particular conda's PATH/LD_LIBRARY_PATH/PYTHONPATH/QT_*, is dropped.
_ROS_ENV_KEEP = (
    "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "LC_ALL",
    "DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "XDG_SESSION_TYPE", "DBUS_SESSION_BUS_ADDRESS",
    "ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY", "RMW_IMPLEMENTATION", "CYCLONEDDS_URI", "FASTRTPS_DEFAULT_PROFILES_FILE",
)


def ros_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in _ROS_ENV_KEEP if key in os.environ}
    env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    return env


# =============================================================================
# session and apple state
# =============================================================================

class Session:
    def __init__(self, root: Path, name: str):
        self.name = name
        self.dir = root / name
        self.path = self.dir / "session.json"
        self.data = read_json(self.path, {})

    def exists(self) -> bool:
        return self.path.exists()

    def create(self, args, directions: list[dict[str, Any]]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data = {
            "session": self.name,
            "created_utc": now_utc(),
            "config_path": str(Path(args.config).resolve()),
            "config_sha256": sha256_file(Path(args.config)),
            "tracking_config_path": str(Path(args.tracking_config).resolve()),
            "overrides": list(args.override),
            "kp": float(args.kp),
            "distance_m": float(args.distance),
            "stops": int(args.stops),
            "hold_duration_s": float(args.hold),
            "settle_sec": float(args.settle),
            "slip_threshold_m": float(args.slip_threshold),
            "directions_source": str(Path(args.directions).resolve()),
            "directions": directions,
            "mock": bool(args.mock),
            "git": {"repo": git_state(REPO_ROOT), "at_tracking": git_state(AT_TRACKING)},
        }
        write_json(self.path, self.data)

    def apple_ids(self) -> list[str]:
        return sorted(p.name for p in self.dir.glob("A[0-9][0-9][0-9]") if (p / "apple.json").exists())

    def next_apple_id(self) -> str:
        ids = [int(name[1:]) for name in self.apple_ids()]
        return f"A{(max(ids) + 1) if ids else 1:03d}"


class Apple:
    def __init__(self, session: Session, apple_id: str):
        self.session = session
        self.id = apple_id
        self.dir = session.dir / apple_id
        self.path = self.dir / "apple.json"
        self.data = read_json(self.path, None)
        if self.data is None:
            self.data = {
                "apple_id": apple_id,
                "session": session.name,
                "created_utc": now_utc(),
                "steps": {step: {"status": "pending"} for step in STEPS},
                "notes": "",
                "files": {},
            }
        for sub in ("calib", "config", "snapshots", "tracking", "pulls", "baseline", "compiled"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.save()

    def save(self) -> None:
        write_json(self.path, self.data)

    def log(self, text: str) -> None:
        with (self.dir / "log.txt").open("a", encoding="utf-8") as stream:
            stream.write(f"[{now_utc()}] {text}\n")

    def step(self, name: str) -> dict[str, Any]:
        return self.data["steps"].setdefault(name, {"status": "pending"})

    def mark(self, name: str, status: str, **extra: Any) -> None:
        entry = self.step(name)
        entry["status"] = status
        entry[f"{status}_utc"] = now_utc()
        entry.update(extra)
        self.save()
        self.log(f"step {name}: {status} {extra if extra else ''}")

    def next_step(self) -> str | None:
        for name in STEPS:
            if self.step(name)["status"] not in {"done", "skipped"}:
                return name
        return None

    def complete(self) -> bool:
        return self.next_step() is None

    # common paths
    @property
    def camera_to_base_path(self) -> Path:
        return self.dir / "calib" / "camera_to_base.json"

    @property
    def request_dir(self) -> Path:
        return self.dir / "tracking" / "requests"

    def direction_files(self, index: int) -> dict[str, Path]:
        return {
            "robot": self.dir / "pulls" / f"d{index:02d}_robot.parquet",
            "baseline": self.dir / "baseline" / f"d{index:02d}_baseline.parquet",
            "compiled": self.dir / "compiled" / f"d{index:02d}.parquet",
        }


# =============================================================================
# the per-apple workflow
# =============================================================================

class FieldSession:
    def __init__(self, args, console: Console | None = None):
        self.args = args
        self.console = console or Console()
        self.session = Session(Path(args.data_root).expanduser(), args.session)
        self.detector: Proc | None = None
        self.ros_ws = Path(args.ros_ws).expanduser()

    # --- settings -------------------------------------------------------------
    @property
    def settings(self) -> dict[str, Any]:
        return self.session.data

    def overrides(self) -> list[str]:
        overrides = list(self.settings.get("overrides", []))
        if self.settings.get("mock") and "robot.use_mock=true" not in overrides:
            overrides.append("robot.use_mock=true")
        return overrides

    # --- detector -------------------------------------------------------------
    def check_ee(self, apple: Apple, profile: str, step: str) -> None:
        """Refuse to continue until Desk's active end effector is ``profile`` (board/gripper)."""
        if self.settings.get("mock") or self.args.no_ee_check:
            return
        from real_robot_exps.ee_profiles import require_profile
        from real_robot_exps.field_config import apply_overrides

        config = yaml.safe_load((apple.dir / "config" / "robot_config.yaml").read_text(encoding="utf-8"))
        apply_overrides(config, self.overrides())
        result = require_profile(
            profile, config=config, ask=self.console.ask, say=self.console.say,
            profiles_path=Path(self.args.ee_profiles),
        )
        apple.data.setdefault("ee_checks", {})[step] = {"expected": profile, "utc": now_utc(), **result}
        apple.save()
        if result.get("skipped"):
            apple.log(f"end-effector check for '{profile}' skipped by operator at {step}")

    def start_detector(self, apple: Apple) -> None:
        if self.detector is not None and self.detector.alive():
            return
        if self.args.no_detector:
            return
        if not apple.camera_to_base_path.exists():
            raise RuntimeError("No camera_to_base.json for this apple; run the calibration step first")
        existing = sorted((apple.dir / "tracking").glob("tracking_*.parquet"))
        output = apple.dir / "tracking" / f"tracking_{len(existing):02d}.parquet"
        apple.request_dir.mkdir(parents=True, exist_ok=True)
        for stale in apple.request_dir.glob("*.request.json"):
            stale.unlink()
        cmd = [
            sys.executable, str(AT_TRACKING / "Detecting.py"),
            "--output", str(output),
            "--config", str(apple.dir / "config" / "tracking_config.yaml"),
            "--camera-to-base", str(apple.camera_to_base_path),
            "--snapshot-dir", str(apple.request_dir),
            "--headless",
        ]
        if self.args.record_video:
            cmd.append("--record")
        self.detector = Proc("detector", cmd, apple.dir / "log.txt", cwd=AT_TRACKING)
        apple.data["files"].setdefault("tracking", [])
        if str(output) not in apple.data["files"]["tracking"]:
            apple.data["files"]["tracking"].append(str(output))
        apple.save()
        self.console.say(f"Detector started -> {output.name} (waiting for the camera)")
        time.sleep(3.0)
        if not self.detector.alive():
            raise RuntimeError(f"Detector exited immediately; see {apple.dir / 'log.txt'}")

    def stop_detector(self) -> None:
        if self.detector is not None:
            self.console.say("Stopping the detector (writes the tracking file)...")
            self.detector.stop()
            self.detector = None

    def snapshot(self, apple: Apple, label: str) -> dict:
        from real_robot_exps.snapshot_geometry import request_snapshot

        output = apple.dir / "snapshots" / f"{label}.json"
        if self.args.no_detector:
            write_json(output, {"label": label, "error": "no detector (--no-detector)"})
            return {"label": label, "error": "no detector"}
        return request_snapshot(
            apple.request_dir, label, output, frames=5, timeout_s=10.0,
            detector_alive=lambda: self.detector is not None and self.detector.alive(),
        )

    # --- steps ----------------------------------------------------------------
    def step_notes(self, apple: Apple) -> None:
        c = self.console
        c.say("Describe the fruiting system (variety, row/tree, branch, anything unusual).")
        c.say("Finish with an empty line.")
        lines = []
        while True:
            line = c.ask("> ")
            if not line:
                break
            lines.append(line)
        apple.data["notes"] = "\n".join(lines)
        apple.data["location"] = c.ask("Row / tree label (optional): ")
        apple.save()

    def step_calibrate(self, apple: Apple) -> None:
        c = self.console
        calib_dir = apple.dir / "calib"
        if self.args.skip_calibration:
            from real_robot_exps.static_constants import CAMERA_TO_BASE_4X4_DEFAULT

            write_json(apple.camera_to_base_path, {
                "camera_to_base_4x4": np.asarray(CAMERA_TO_BASE_4X4_DEFAULT).tolist(),
                "source": "static_constants.CAMERA_TO_BASE_4X4_DEFAULT (--skip-calibration)",
            })
            apple.data["calibration"] = {"verdict": "SKIPPED"}
            apple.save()
            c.say("Calibration skipped: using the static camera matrix.")
            return

        name = re.sub(r"[^A-Za-z0-9_]", "_", f"{self.session.name}_{apple.id}")
        c.say("Place the camera so the apple, spur and branch are in view (the arm must")
        c.say("also be able to show the ChArUco board to it).")
        if c.yes("Open a live camera view to aim the camera?", default=True):
            view = Proc(
                "camera_view",
                ros_command("ros2 launch easy_handeye2_charuco charuco_view.launch.py use_rviz:=false use_image_view:=true", self.ros_ws),
                apple.dir / "log.txt",
                env=ros_env(),
            )
            time.sleep(6.0)
            if not view.alive() or "process has died" in view.log_text():
                c.say(f"!! The live view did not start; see {apple.dir / 'log.txt'} (camera_view).")
            c.enter("Camera aimed (this closes the live view)")
            view.stop()
        self.check_ee(apple, "gripper", "calibrate")
        self._hold_board()
        try:
            self._run_calibration(apple, name, calib_dir)
        finally:
            self._release_board()
        apple.data["calibration"]["board_mount"] = "suction (air on, fingers in), gripper end-effector profile"
        apple.save()

    def _run_calibration(self, apple: Apple, name: str, calib_dir: Path) -> None:
        c = self.console
        while True:
            launch = Proc(
                "calib_launch",
                ros_command(f"ros2 launch easy_handeye2_charuco eye_on_base_calib.launch.py name:={name}", self.ros_ws),
                apple.dir / "log.txt",
                env=ros_env(),
            )
            try:
                time.sleep(5.0)
                if not launch.alive():
                    raise RuntimeError("calibration launch exited; see log.txt")
                c.say("The calibration will ask you to free-drive the board to the image centre.")
                robot_config = (
                    "$(ros2 pkg prefix easy_handeye2_franka_auto)/share/easy_handeye2_franka_auto/config/robot.yaml"
                )
                code = subprocess.call(ros_command(
                    "ros2 run easy_handeye2_franka_auto handeye_auto_calibrate "
                    f"--robot-config {robot_config} --name {name} "
                    "--robot-base-frame fr3_link0 --robot-effector-frame handeye_ee "
                    f"--n-poses {int(self.args.calib_poses)} "
                    f"--rotation-delta-degrees {float(self.args.calib_rotation_deg):g} --seed 0 --return-home",
                    self.ros_ws,
                ), env=ros_env())
                if code != 0:
                    raise RuntimeError(f"handeye_auto_calibrate exited with {code}")
                optical_path = calib_dir / "camera_link_to_optical.json"
                code = subprocess.call(
                    ros_command(
                        f"/usr/bin/python3 -m real_robot_exps.field_tf_lookup --parent camera_link "
                        f"--child camera_color_optical_frame --output {optical_path}",
                        self.ros_ws,
                    ),
                    cwd=REPO_ROOT,
                    env=ros_env(),
                )
                if code != 0:
                    raise RuntimeError("could not read camera_link -> camera_color_optical_frame from TF")
            finally:
                launch.stop()

            report = subprocess.run(
                ros_command(f"ros2 run easy_handeye2_franka_auto evaluate_calibration --name {name}", self.ros_ws),
                capture_output=True, text=True, env=ros_env(),
            )
            (calib_dir / "report.txt").write_text(report.stdout + report.stderr, encoding="utf-8")
            c.say(report.stdout[-2500:])
            match = re.search(r"Verdict:\s*(\w+)", report.stdout)
            verdict = match.group(1) if match else "UNKNOWN"

            for kind, suffix in (("calibrations", ".calib"), ("samples", ".samples")):
                source = HANDEYE_DIR / kind / f"{name}{suffix}"
                if source.exists():
                    shutil.copy2(source, calib_dir / source.name)
            self._write_camera_to_base(apple, calib_dir / f"{name}.calib", optical_path, verdict)

            if verdict == "GOOD":
                break
            choice = c.choose(
                f"Calibration verdict {verdict}.", {"r": "redo", "a": "accept anyway"}, default="r"
            )
            if choice == "a":
                apple.data["calibration"]["accepted_despite_verdict"] = True
                apple.save()
                break

    # --- gripper air (valve only; fingers untouched) -------------------------------
    def _gripper_call(self, service: str, value: bool, what: str) -> None:
        from real_robot_exps.gripper_test import GripperClient

        gripper = GripperClient(
            mock=bool(self.settings.get("mock") or self.args.mock_gripper), timeout_s=30.0, service=service
        )
        try:
            response = gripper.send_request(value)
        finally:
            gripper.terminate()
        if response is not None and not response.success:
            raise RuntimeError(f"{what} rejected by the gripper: {response.message}")

    def air_on(self) -> None:
        from real_robot_exps.gripper_test import VALVE_SERVICE

        self._gripper_call(VALVE_SERVICE, True, "air on")

    def air_off(self) -> None:
        from real_robot_exps.gripper_test import VALVE_SERVICE

        self._gripper_call(VALVE_SERVICE, False, "air off")

    def _hold_board(self) -> None:
        """Operator presses the ChArUco board to the gripper; suction holds it for the calibration."""
        c = self.console
        while True:
            c.enter("Hold the ChArUco board flat against the gripper (fingers in)")
            try:
                self.air_on()
            except Exception as exc:
                c.say(f"!! Air on failed: {exc}")
                if c.yes("Retry?", default=True):
                    continue
                raise
            if c.yes("Let go gently: does suction hold the board without slipping?", default=True):
                return
            c.enter("Hold the board again; Enter turns the air off so you can reposition it")
            self.air_off()

    def _release_board(self) -> None:
        c = self.console
        try:
            c.enter("Hold the ChArUco board; Enter turns the air off and releases it")
        except (KeyboardInterrupt, EOFError):
            c.say("!! Air left ON (board still held). Release it with: python -m real_robot_exps.gripper_test air-off")
            raise
        self.air_off()
        c.say("Air off, board released.")

    def _write_camera_to_base(self, apple: Apple, calib_path: Path, optical_path: Path, verdict: str) -> None:
        from real_robot_exps.calibrate_camera_to_base import _load_handeye_calibration

        base_to_camera_link, calib_meta = _load_handeye_calibration(calib_path)
        camera_link_to_optical = np.asarray(read_json(optical_path)["matrix_4x4"], dtype=np.float64)
        camera_to_base = compose_camera_to_base(base_to_camera_link, camera_link_to_optical)
        write_json(apple.camera_to_base_path, {
            "camera_to_base_4x4": camera_to_base.tolist(),
            "semantics": "maps camera_color_optical_frame points into fr3_link0 (Franka base)",
            "composition": "T_base_optical = T_base_camera_link (.calib) @ T_camera_link_optical (RealSense TF)",
            "base_to_camera_link_4x4": base_to_camera_link.tolist(),
            "camera_link_to_optical_4x4": camera_link_to_optical.tolist(),
            "calib_file": str(calib_path),
            "calib_parameters": calib_meta.get("parameters", {}),
            "verdict": verdict,
            "created_utc": now_utc(),
        })
        apple.data["calibration"] = {
            "verdict": verdict,
            "calib_file": calib_path.name,
            "camera_position_in_base_m": camera_to_base[:3, 3].tolist(),
        }
        apple.save()
        self.console.say(
            f"Camera optical centre in robot base: {np.round(camera_to_base[:3, 3], 3).tolist()} m "
            "(check against a tape measure)"
        )

    def step_tags(self, apple: Apple) -> None:
        c = self.console
        self.check_ee(apple, "gripper", "tags")
        from real_robot_exps.gripper_test import GripperClient

        # Grasp starts from a known state: fingers in, air off (also checks gripper_grab is up).
        c.say("Checking the gripper service...")
        gripper = GripperClient(mock=bool(self.settings.get("mock") or self.args.mock_gripper), timeout_s=30.0)
        try:
            gripper.send_request(False)
        finally:
            gripper.terminate()
        c.say("Gripper open.")
        c.say("Place the tags, each facing the camera:  Branch = tag 0,  Spur = tag 1,  Apple = tag 2")
        c.say("(tags sit on the surface; compile moves them inward by the measured radius).")
        c.enter("Tags placed and facing the camera")

    def step_snapshots(self, apple: Apple) -> None:
        c = self.console
        self.start_detector(apple)
        for label, prompt in (
            ("under_gravity", "Let the apple hang naturally under gravity (hands off)"),
            ("lengthened", "Stretch the structure so the branch->spur->apple segments are straight"),
        ):
            while True:
                c.enter(prompt)
                try:
                    snapshot = self.snapshot(apple, label)
                except Exception as exc:
                    c.say(f"Snapshot failed: {exc}")
                    if c.yes("Turn the tags toward the camera and retry?", default=True):
                        continue
                    raise
                seen = snapshot.get("tracker_seen_counts", {})
                c.say(f"  {label}: {snapshot.get('camera_frame_count', 0)} frames, seen {seen}; "
                      f"apple at {np.round(snapshot.get('apple_pos', [np.nan] * 3), 3).tolist()} m")
                if snapshot.get("error") or c.yes("Keep this snapshot?", default=True):
                    break
        apple.data["files"]["snapshots"] = [str(apple.dir / "snapshots" / f"{x}.json") for x in ("under_gravity", "lengthened")]
        apple.save()

    def step_grasp(self, apple: Apple) -> None:
        c = self.console
        self.check_ee(apple, "gripper", "grasp")
        self.start_detector(apple)
        c.say("Put the robot in hand-guiding mode and bring the open gripper around the apple.")
        c.enter("Gripper positioned around the apple; hands off the arm")
        from real_robot_exps.gripper_test import GripperClient

        gripper = GripperClient(mock=bool(self.settings.get("mock") or self.args.mock_gripper), timeout_s=30.0)
        try:
            gripper.send_request(True)
        finally:
            gripper.terminate()
        time.sleep(2.0)
        if not c.yes("Is the apple held firmly?", default=True):
            raise RuntimeError("grasp not accepted; open the gripper and repeat the grasp step")

    def _pull_plan(self, apple: Apple, directions: list[dict[str, Any]]) -> dict:
        s = self.settings
        pre_grasp_geometry = {
            "under_gravity_snapshot": read_json(apple.dir / "snapshots" / "under_gravity.json", {}),
            "lengthened_snapshot": read_json(apple.dir / "snapshots" / "lengthened.json", {}),
            "parts": {},
            "parts_note": "measured after the pulls; merged in by compile (field_session --compile)",
        }
        return {
            "config_path": str(apple.dir / "config" / "robot_config.yaml"),
            "overrides": self.overrides(),
            "kp": s["kp"],
            "distance_m": s["distance_m"],
            "stops": s["stops"],
            "hold_duration_s": s["hold_duration_s"],
            "settle_sec": s["settle_sec"],
            "slip_threshold_m": s["slip_threshold_m"],
            "confirm_each": bool(self.args.confirm_each),
            "num_directions": len(s["directions"]),
            "mock_gripper": bool(s.get("mock") or self.args.mock_gripper),
            "snapshot_dir": None if self.args.no_detector else str(apple.request_dir),
            "snapshot_output_dir": str(apple.dir / "snapshots"),
            "status_path": str(apple.dir / "pulls" / "status.json"),
            "directions": [
                {**d, "output": str(apple.direction_files(d["index"])["robot"])} for d in directions
            ],
            "run_metadata": {
                "apple_id": apple.id,
                "session": self.session.name,
                "notes": apple.data.get("notes", ""),
                "location": apple.data.get("location", ""),
                "calibration": apple.data.get("calibration", {}),
                "camera_to_base_path": str(apple.camera_to_base_path),
                "pre_grasp_geometry": pre_grasp_geometry,
            },
        }

    def step_pulls(self, apple: Apple) -> None:
        c = self.console
        s = self.settings
        done = set(apple.step("pulls").get("completed_directions", []))
        remaining = [d for d in s["directions"] if d["index"] not in done]
        c.say(f"Plan: {len(remaining)} direction(s), {s['distance_m'] * 100:.1f} cm in {s['stops']} stops, "
              f"kp={s['kp']:g}, {s['hold_duration_s']:g} s holds, {s['settle_sec']:g} s settle between directions")
        for d in remaining:
            c.say(f"  d{d['index']:02d}: theta={d['theta']:.2f} phi={d['phi']:.2f} {d.get('name', '')}")
        if done:
            c.say(f"Already recorded: {sorted(done)}")
        if not c.yes("The apple is held. Start the pulls?", default=True):
            raise RuntimeError("pulls not started by operator")
        self.check_ee(apple, "gripper", "pulls")
        self.start_detector(apple)
        plan = self._pull_plan(apple, remaining)
        plan_path = apple.dir / "pulls" / f"pull_plan_{len(list((apple.dir / 'pulls').glob('pull_plan_*.json'))):02d}.json"
        write_json(plan_path, plan)
        status_path = Path(plan["status_path"])
        status_path.unlink(missing_ok=True)
        code = subprocess.call(
            [sys.executable, "-m", "real_robot_exps.field_pull", "--plan", str(plan_path)], cwd=REPO_ROOT
        )
        status = read_json(status_path, {}) or {}
        completed = sorted(done | set(status.get("completed_directions", [])))
        apple.step("pulls")["completed_directions"] = completed
        if status.get("partial_direction") is not None:
            apple.step("pulls").setdefault("aborted_directions", []).append(status["partial_direction"])
        apple.data["start_pose_4x4"] = status.get("start_pose_4x4", apple.data.get("start_pose_4x4"))
        apple.save()
        if code != 0 or status.get("aborted"):
            # field_pull opened the gripper, so the next attempt needs a new grasp.
            apple.mark("grasp", "pending", reason="pull series aborted; gripper was opened")
            raise RuntimeError(
                f"pull series stopped ({status.get('error', f'exit {code}')}); gripper was opened. "
                f"Recorded directions: {completed}. Re-grasp and resume to continue."
            )
        missing = [d["index"] for d in s["directions"] if d["index"] not in completed]
        if missing and not status.get("stopped_by_operator"):
            raise RuntimeError(f"directions {missing} were not recorded")
        self.stop_detector()

    def step_baseline(self, apple: Apple) -> None:
        c = self.console
        self.stop_detector()
        robots = [
            (d, apple.direction_files(d["index"]))
            for d in self.settings["directions"]
            if apple.direction_files(d["index"])["robot"].exists()
        ]
        if not robots:
            raise RuntimeError("no pull recordings to baseline")
        c.say("The baseline replays every pull with nothing held. The gripper closes empty")
        c.say("at the grasp pose, so the apple must be out of the gripper's path.")
        c.enter("Gripper open and clear of the apple (cut it or move it aside)")
        self.check_ee(apple, "gripper", "baseline")
        for d, files in robots:
            if files["baseline"].exists():
                continue
            c.say(f"Baseline d{d['index']:02d}...")
            cmd = [
                sys.executable, "-m", "real_robot_exps.collect_joint_velocity_baseline",
                "--actual-robot", str(files["robot"]),
                "--output", str(files["baseline"]),
                "--config", str(apple.dir / "config" / "robot_config.yaml"),
                "--gripper", "closed",
            ]
            for override in self.overrides():
                cmd += ["--override", override]
            if self.settings.get("mock") or self.args.mock_gripper:
                cmd.append("--mock-gripper")
            if subprocess.call(cmd, cwd=REPO_ROOT) != 0:
                raise RuntimeError(f"baseline d{d['index']:02d} failed")

    def step_measurements(self, apple: Apple) -> None:
        c = self.console
        c.say("Measure and weigh the parts (blank = not measured, only where allowed).")
        while True:
            raw: dict[str, dict[str, float | None]] = {}
            for part, field, label, unit, lo, hi, required in MEASUREMENT_FIELDS:
                raw.setdefault(part, {})[field] = c.number(label, unit, lo, hi, required)
            c.say("\nYou entered:")
            for part, values in raw.items():
                c.say(f"  {part:<8} " + ", ".join(f"{k}={v:g}" if v is not None else f"{k}=-" for k, v in values.items()))
            if c.yes("Correct?", default=True):
                break
        extra = c.ask("Anything else to note (optional): ")
        apple.data["measurements_raw"] = raw
        apple.data["parts"] = parts_from_measurements(raw)
        if extra:
            apple.data["notes_after"] = extra
        apple.save()
        write_json(apple.dir / "parts.json", {"parts": apple.data["parts"], "measurements_raw": raw})

    # --- driver ---------------------------------------------------------------
    def prepare_apple(self, apple: Apple) -> None:
        config_dir = apple.dir / "config"
        if not (config_dir / "robot_config.yaml").exists():
            shutil.copy2(self.settings["config_path"], config_dir / "robot_config.yaml")
        if not (config_dir / "tracking_config.yaml").exists():
            shutil.copy2(self.settings["tracking_config_path"], config_dir / "tracking_config.yaml")

    def run_apple(self, apple: Apple) -> bool:
        c = self.console
        self.prepare_apple(apple)
        handlers = {name: getattr(self, f"step_{name}") for name in STEPS}
        try:
            while True:
                name = apple.next_step()
                if name is None:
                    break
                index = STEPS.index(name) + 1
                c.banner(f"{apple.id}  step {index}/{len(STEPS)}: {STEP_TITLES[name]}")
                apple.mark(name, "started")
                try:
                    handlers[name](apple)
                except (KeyboardInterrupt, EOFError):
                    apple.mark(name, "failed", error="interrupted")
                    raise
                except Exception as exc:
                    apple.mark(name, "failed", error=f"{type(exc).__name__}: {exc}")
                    c.say(f"\n!! {name} failed: {exc}")
                    choice = c.choose("What now?", {"r": "retry", "s": "skip step", "q": "quit"}, default="r")
                    if choice == "s":
                        apple.mark(name, "skipped")
                    elif choice == "q":
                        return False
                    continue
                apple.mark(name, "done")
        finally:
            self.stop_detector()
        c.banner(f"{apple.id} complete. Data in {apple.dir}")
        c.say(f"At home: python -m real_robot_exps.field_session --session {self.session.name} --compile {apple.id}")
        return True

    def collect(self) -> None:
        c = self.console
        if not self.session.exists():
            directions = load_directions(Path(self.args.directions))
            self.session.create(self.args, directions)
            c.say(f"New session {self.session.name} at {self.session.dir} ({len(directions)} directions)")
        else:
            c.say(f"Session {self.session.name}: {len(self.session.apple_ids())} apple(s) so far")
        apple_id = self.args.apple
        if apple_id is None:
            unfinished = [a for a in self.session.apple_ids() if not Apple(self.session, a).complete()]
            if unfinished and c.yes(f"Resume unfinished apple {unfinished[-1]}?", default=True):
                apple_id = unfinished[-1]
        while True:
            apple = Apple(self.session, apple_id or self.session.next_apple_id())
            c.say(f"\nApple {apple.id} -> {apple.dir}")
            if not self.run_apple(apple):
                return
            apple_id = None
            if not c.yes("Next apple?", default=True):
                return

    def list_apples(self) -> None:
        for apple_id in self.session.apple_ids():
            apple = Apple(self.session, apple_id)
            steps = " ".join(
                f"{name}:{apple.step(name)['status'][0]}" for name in STEPS
            )
            compiled = len(list((apple.dir / "compiled").glob("d*.parquet")))
            self.console.say(f"{apple_id}  {steps}  compiled={compiled}  {apple.data.get('location', '')}")


# =============================================================================
# offline compile (at home)
# =============================================================================

def _tracking_for(robot_path: Path, tracking_paths: list[Path]) -> Path:
    import pyarrow.parquet as pq

    stamps = pq.read_table(robot_path, columns=["timestamp"]).column("timestamp").to_pylist()
    stamps = [t for t in stamps if t is not None]
    start, end = min(stamps), max(stamps)
    best, best_overlap = None, -1.0
    for path in tracking_paths:
        times = pq.read_table(path, columns=["timestamp"]).column("timestamp").to_pylist()
        if not times:
            continue
        overlap = min(end, max(times)) - max(start, min(times))
        if overlap > best_overlap:
            best, best_overlap = path, overlap
    if best is None or best_overlap < 0.9 * (end - start):
        raise ValueError(f"No tracking file covers {robot_path.name}")
    return best


def compile_apple(apple: Apple, *, overwrite: bool = False, viz: bool = True, console: Console | None = None) -> list[Path]:
    from real_robot_exps.compile_static_sysid import compile_static_episode

    console = console or Console()
    parts = apple.data.get("parts")
    if not parts:
        raise ValueError(f"{apple.id}: no measured parts yet (measurement step not done)")
    tracking_paths = sorted((apple.dir / "tracking").glob("tracking_*.parquet"))
    if not tracking_paths:
        raise ValueError(f"{apple.id}: no tracking files")
    outputs = []
    for robot_path in sorted((apple.dir / "pulls").glob("d[0-9][0-9]_robot.parquet")):
        index = int(robot_path.name[1:3])
        files = apple.direction_files(index)
        if files["compiled"].exists() and not overwrite:
            console.say(f"{apple.id} d{index:02d}: already compiled")
            outputs.append(files["compiled"])
            continue
        if not files["baseline"].exists():
            console.say(f"{apple.id} d{index:02d}: no baseline, skipped")
            continue
        tracking = _tracking_for(robot_path, tracking_paths)
        console.say(f"{apple.id} d{index:02d}: compiling with {tracking.name}")
        compile_static_episode(
            robot_path, tracking, files["compiled"],
            baseline_path=files["baseline"], parts=parts, command_argv=sys.argv,
        )
        outputs.append(files["compiled"])
        if viz:
            subprocess.call([
                sys.executable, "-m", "real_robot_exps.viz_static_sysid",
                "--input", str(files["compiled"]), "--save", str(files["compiled"].with_suffix(".png")), "--no-show",
            ], cwd=REPO_ROOT)
    apple.data["compiled"] = {"utc": now_utc(), "files": [str(p) for p in outputs]}
    apple.save()
    return outputs


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="Session name, e.g. 2026-10-02_orchardA")
    parser.add_argument("--data-root", default="~/field_data")
    parser.add_argument("--apple", default=None, help="Resume/run a specific apple id, e.g. A003")
    parser.add_argument("--redo", action="append", default=[], choices=STEPS, metavar="STEP",
                        help=f"With --apple: mark STEP pending again so it is re-run ({', '.join(STEPS)})")
    parser.add_argument("--list", action="store_true", help="List the session's apples and their step status")
    parser.add_argument("--compile", metavar="APPLE|all", default=None, help="Offline: compile an apple (or all)")
    parser.add_argument("--overwrite", action="store_true", help="With --compile: recompile existing outputs")
    parser.add_argument("--no-viz", action="store_true", help="With --compile: skip the PNG plots")
    # session settings (fixed when the session is created)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--tracking-config", default=str(DEFAULT_TRACKING_CONFIG))
    parser.add_argument("--directions", default=str(DEFAULT_DIRECTIONS))
    parser.add_argument("--override", action="append", default=[], help="Robot config override key.path=value")
    parser.add_argument("--kp", type=float, default=100.0)
    parser.add_argument("--distance", type=float, default=0.04, help="Pull distance [m]")
    parser.add_argument("--stops", type=int, default=4)
    parser.add_argument("--hold", type=float, default=1.0, help="Hold duration per stop [s]")
    parser.add_argument("--settle", type=float, default=5.0, help="Settle time at the start pose before each direction [s]")
    parser.add_argument("--slip-threshold", type=float, default=0.01, help="Apple-to-gripper drift that pauses the series [m]")
    # run options
    parser.add_argument("--confirm-each", action="store_true", help="Pause (apple held) between directions")
    parser.add_argument("--calib-poses", type=int, default=15)
    parser.add_argument("--calib-rotation-deg", type=float, default=25.0,
                        help="Board tilt of the calibration poses; lower it if the suction-held board slips")
    parser.add_argument("--record-video", action="store_true", help="Also record the detector camera feed")
    parser.add_argument("--ros-ws", default=str(DEFAULT_ROS_WS))
    # testing without hardware
    parser.add_argument("--mock", action="store_true", help="Mock robot + gripper (new sessions only)")
    parser.add_argument("--mock-gripper", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true", help="Use the static camera matrix")
    parser.add_argument("--no-detector", action="store_true", help="Run without the camera (no snapshots/tracking)")
    parser.add_argument("--ee-profiles", default=str(REPO_ROOT / "real_robot_exps" / "ee_profiles.yaml"),
                        help="Captured Desk end-effector profiles (only 'gripper' is used), see ee_profiles.py")
    parser.add_argument("--no-ee-check", action="store_true", help="Do not check Desk's active end effector")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    field = FieldSession(args)
    if args.redo:
        if not args.apple or args.apple not in field.session.apple_ids():
            raise SystemExit("--redo needs --apple with an existing apple id")
        apple = Apple(field.session, args.apple)
        for step in args.redo:
            apple.mark(step, "pending", reason="--redo")
            print(f"{args.apple}: {step} will be re-run")
    if args.list:
        field.list_apples()
        return
    if args.compile:
        if not field.session.exists():
            raise SystemExit(f"No session {args.session} under {args.data_root}")
        ids = field.session.apple_ids() if args.compile == "all" else [args.compile]
        for apple_id in ids:
            apple = Apple(field.session, apple_id)
            try:
                compile_apple(apple, overwrite=args.overwrite, viz=not args.no_viz)
            except Exception as exc:
                print(f"{apple_id}: {exc}")
        return
    try:
        field.collect()
    except (KeyboardInterrupt, EOFError):
        field.stop_detector()
        print(
            f"\nStopped. Everything recorded so far is saved; resume with:\n"
            f"  python -m real_robot_exps.field_session --session {args.session}"
            + (f" --data-root {args.data_root}" if args.data_root != "~/field_data" else "")
        )
        sys.exit(130)


if __name__ == "__main__":
    main()
