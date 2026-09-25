"""Check which Franka Desk end-effector profile is active (board vs gripper).

Desk's end-effector profiles (flange->EE transform F_T_NE, mass, centre of mass,
inertia) cannot be switched through FCI or any documented API, so the operator
selects them in Desk. This module makes sure the right one is active before the
robot moves: it captures each profile's fingerprint once from RobotState and
later compares the live values against it.

    # once per profile, with that profile selected in Desk
    python -m real_robot_exps.ee_profiles capture --name board
    python -m real_robot_exps.ee_profiles capture --name gripper
    # what is active right now
    python -m real_robot_exps.ee_profiles show

The robot must not be under FCI control by another program while this reads it.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

DEFAULT_PROFILES_PATH = Path(__file__).resolve().with_name("ee_profiles.yaml")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")

# Two profiles for different tools differ by far more than these.
MASS_TOL_KG = 0.01
TRANSLATION_TOL_M = 0.001
ROTATION_TOL_DEG = 0.5
COM_TOL_M = 0.002


def read_ee_state(config: dict) -> dict[str, Any]:
    """Desk end-effector parameters as the robot reports them (no motion, no control)."""
    robot_cfg = config["robot"]
    if robot_cfg.get("use_mock", False):
        raise RuntimeError("mock robot has no Desk end-effector profile")
    import pylibfranka as plf

    robot = plf.Robot(robot_cfg["ip"])
    try:
        state = robot.read_once()
        return {
            "F_T_NE": np.asarray(state.F_T_NE, dtype=np.float64).reshape(4, 4).T.tolist(),  # row-major
            "m_ee": float(state.m_ee),
            "F_x_Cee": np.asarray(state.F_x_Cee, dtype=np.float64).tolist(),
            "I_ee": np.asarray(state.I_ee, dtype=np.float64).tolist(),
        }
    finally:
        robot.stop()


def load_profiles(path: Path = DEFAULT_PROFILES_PATH) -> dict[str, dict[str, Any]]:
    if not Path(path).exists():
        return {}
    return dict((yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("profiles") or {})


def save_profile(name: str, state: dict[str, Any], path: Path = DEFAULT_PROFILES_PATH) -> None:
    path = Path(path)
    profiles = load_profiles(path)
    profiles[name] = {**state, "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    header = (
        "# Franka Desk end-effector profiles, captured from RobotState with\n"
        "#   python -m real_robot_exps.ee_profiles capture --name <name>\n"
        "# F_T_NE is row-major (flange -> nominal EE). Used to check that Desk has the\n"
        "# right profile selected before the robot moves.\n"
    )
    path.write_text(header + yaml.safe_dump({"profiles": profiles}, sort_keys=True), encoding="utf-8")


def profile_differences(state: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    """Human-readable reasons why ``state`` is not ``profile`` (empty list = match)."""
    reasons = []
    mass_delta = abs(float(state["m_ee"]) - float(profile["m_ee"]))
    if mass_delta > MASS_TOL_KG:
        reasons.append(f"mass {state['m_ee']:.3f} kg vs {profile['m_ee']:.3f} kg")
    live = np.asarray(state["F_T_NE"], dtype=np.float64)
    expected = np.asarray(profile["F_T_NE"], dtype=np.float64)
    translation_delta = float(np.linalg.norm(live[:3, 3] - expected[:3, 3]))
    if translation_delta > TRANSLATION_TOL_M:
        reasons.append(f"EE offset differs by {translation_delta * 1000:.1f} mm")
    cos_angle = (np.trace(live[:3, :3].T @ expected[:3, :3]) - 1.0) / 2.0
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    if angle_deg > ROTATION_TOL_DEG:
        reasons.append(f"EE rotation differs by {angle_deg:.1f} deg")
    com_delta = float(np.linalg.norm(np.asarray(state["F_x_Cee"]) - np.asarray(profile["F_x_Cee"])))
    if com_delta > COM_TOL_M:
        reasons.append(f"centre of mass differs by {com_delta * 1000:.1f} mm")
    return reasons


def identify(state: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> str | None:
    for name, profile in profiles.items():
        if not profile_differences(state, profile):
            return name
    return None


def require_profile(
    name: str,
    *,
    config: dict,
    ask: Callable[[str], str],
    say: Callable[[str], None],
    profiles_path: Path = DEFAULT_PROFILES_PATH,
    read_state: Callable[[dict], dict[str, Any]] = read_ee_state,
) -> dict[str, Any]:
    """Block until Desk's active end effector is profile ``name``; return the live state.

    If ``name`` was never captured, the operator selects it in Desk and it is
    captured on the spot. 's' at the prompt skips the check (recorded by the caller).
    """
    while True:
        profiles = load_profiles(profiles_path)
        if name not in profiles:
            answer = ask(
                f"No '{name}' end-effector profile captured yet. Select the '{name}' end effector "
                "in Desk, then press Enter to capture it ('s' to skip the check): "
            ).strip().lower()
            if answer == "s":
                return {"skipped": True}
            save_profile(name, read_state(config), profiles_path)
            say(f"Captured Desk profile '{name}' into {profiles_path}")
            continue
        state = read_state(config)
        reasons = profile_differences(state, profiles[name])
        if not reasons:
            say(f"Desk end effector: '{name}' ({state['m_ee']:.3f} kg) - OK")
            return {**state, "profile": name}
        active = identify(state, profiles)
        say(f"!! Desk end effector is {repr(active) if active else 'unknown'}, expected '{name}': "
            + "; ".join(reasons))
        answer = ask(f"Select the '{name}' end effector in Desk, then Enter to re-check ('s' to skip): ")
        if answer.strip().lower() == "s":
            return {**state, "profile": active, "skipped": True, "expected": name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("capture", "show"))
    parser.add_argument("--name", help="profile name for capture, e.g. board or gripper")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    state = read_ee_state(config)
    if args.command == "capture":
        if not args.name:
            parser.error("capture needs --name")
        save_profile(args.name, state, args.profiles)
        print(f"Captured '{args.name}' into {args.profiles}")
    active = identify(state, load_profiles(args.profiles))
    print(json.dumps({"active_profile": active, **state}, indent=2))


if __name__ == "__main__":
    main()
