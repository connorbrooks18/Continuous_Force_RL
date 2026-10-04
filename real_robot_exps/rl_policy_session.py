"""Guided real-arm run of the VIC-harvest policy: grasp an apple, then run the policy.

1. grasp       hand-guide the open gripper around the apple, calibrate F/T (gripper still
               free), close the gripper, and confirm the apple is held. A rejected grasp
               reopens the gripper and starts the step over.
2. run_policy  start torque mode at the grasped pose and run the policy (rl_policy_env).
               Press any key to stop it (the policy never ends the episode itself); the
               arm holds its last target while you answer the next prompt.
3. next        after a pick, ask whether to move to the next apple. Yes releases the
               apple, leaves torque mode, and repeats from the grasp (F/T is calibrated
               again with the gripper free). No shuts the arm down.

The gripper controller stack is launched first, the same way field_session does it
(stray processes killed, ``lfd_gripper.launch.py`` relaunched, gripper opened and checked).

The checkpoint must be a [D17] tool-frame one (``action_frame: tool`` and the tool-frame
``actor_layout`` in meta.json). It is loaded before the gripper stack or the arm is touched,
so a world-frame or mismatched checkpoint is refused up front (exit 2).

    python -m real_robot_exps.rl_policy_session --checkpoint <ckpt_dir>
    python -m real_robot_exps.rl_policy_session --checkpoint <ckpt_dir> --mock --mock-gripper   # dry run
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from real_robot_exps.field_session import DEFAULT_ROS_WS, Console, Proc, ros_env
from real_robot_exps.gripper_stack import (
    DEFAULT_PASSWORD,
    DEFAULT_SSID,
    gripper_stack_ready,
    kill_stray_gripper_processes,
    launch_gripper_stack,
    run_gripper_command,
)
from real_robot_exps.rl_policy_env import (
    FrankaVicHarvestEnv,
    HarvestPolicy,
    KeypressStop,
    new_rollout_log,
    run_rollout,
    save_rollout_log,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANUAL_RELEASE = "python -m real_robot_exps.gripper_test open"


class GraspRejected(RuntimeError):
    """The operator said the apple is not held; the gripper has been reopened."""


class GripperError(RuntimeError):
    """A gripper_test command timed out or was rejected."""


class PolicySession:
    def __init__(self, args, console: Console | None = None, *, policy=None, env=None):
        self.args = args
        self.console = console or Console()
        self.log_dir = Path(args.log_dir).expanduser()
        self.policy = policy
        self.env = env
        self._stack: Proc | None = None
        self.gripper_closed = False  # last commanded state, drives the release prompt at the end

    # ------------------------------------------------------------------ gripper

    def gripper(self, mode: str) -> None:
        """``close`` / ``open`` through a fresh gripper_test process (see gripper_stack)."""
        if mode == "close":
            self.gripper_closed = True  # set before the call: a timed-out close may still have closed
        if self.args.mock_gripper:
            self.console.say(f"[mock gripper] {mode}")
        else:
            try:
                run_gripper_command(mode)
            except (TimeoutError, RuntimeError) as exc:
                raise GripperError(str(exc)) from exc
        if mode == "open":
            self.gripper_closed = False

    def ensure_gripper_stack(self, *, force_restart: bool = False) -> None:
        """Kill stray gripper processes, launch one clean stack, then open the gripper and
        have the operator confirm it released. Raises if the controller doesn't come up."""
        c = self.console
        if self.args.mock_gripper:
            return
        if not self.args.no_gripper_stack:
            if not force_restart and self._stack is not None and self._stack.alive():
                if gripper_stack_ready(timeout_s=2.0)[0]:
                    return
            c.say("Starting the gripper controller (killing any stray instances first)...")
            kill_stray_gripper_processes()
            if self._stack is not None:
                self._stack.stop()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / "gripper_stack.log"
            self._stack = launch_gripper_stack(
                Path(self.args.ros_ws).expanduser(), log_path,
                ssid=self.args.gripper_ssid, password=self.args.gripper_password, env=ros_env(),
            )
            ready, error = gripper_stack_ready(timeout_s=45.0)
            if not ready:
                raise RuntimeError(f"gripper controller did not come up: {error}. See {log_path}")
            c.say("Gripper controller is up (gripper_grab responding).")
        self.open_and_confirm()

    def open_and_confirm(self) -> None:
        c = self.console
        while True:
            try:
                self.gripper("open")
                error = None
            except GripperError as exc:
                error = exc
                c.say(f"!! open failed: {exc}")
            if c.yes("Is the gripper released (fingers in, air off)?", default=error is None):
                return
            if c.choose("Gripper not released.", {"r": "try opening again", "c": "continue anyway"},
                        default="r") == "c":
                return

    # ------------------------------------------------------------------ steps

    def step_grasp(self) -> None:
        c = self.console
        c.say("Put the robot in hand-guiding mode and bring the open gripper around the apple.")
        c.enter("Gripper around the apple, not touching it; hands off the arm")
        # Before closing: the policy was trained seeing the apple/stem load at episode start.
        c.say("Calibrating F/T (gripper free)...")
        c.say(f"F/T bias: {np.round(self.env.calibrate_ft_bias(), 3)}")
        self.gripper("close")
        time.sleep(0.0 if self.args.mock_gripper else 2.0)
        if not c.yes("Is the apple within the grasp and held firmly?", default=True):
            self.gripper("open")
            raise GraspRejected("grasp not accepted; gripper reopened")

    def step_run_policy(self, log: dict) -> bool:
        """False if the start pose was refused as out of distribution (nothing commanded)."""
        self.console.say("Starting torque mode at the grasped pose...")
        obs, _ = self.env.reset()
        with KeypressStop() as stop:  # the policy never terminates; the operator ends the run
            return run_rollout(
                self.policy, self.env, obs, self.args.steps, log,
                allow_ood=self.args.allow_ood, say=self.console.say, should_stop=stop,
            )

    # ------------------------------------------------------------------ driver

    def grasp_until_accepted(self, apple: int) -> bool:
        """Retry shape of field_session.run_apple. False if the operator quits."""
        c = self.console
        c.banner(f"apple {apple}, step 1/2: hand-guide to the apple and grasp")
        while True:
            try:
                self.step_grasp()
                return True
            except (KeyboardInterrupt, EOFError):
                raise
            except Exception as exc:
                c.say(f"\n!! grasp: {type(exc).__name__}: {exc}")
                choice = c.choose(
                    "What now?",
                    {"r": "retry the grasp", "g": "restart the gripper controller, then retry", "q": "quit"},
                    default="g" if isinstance(exc, GripperError) else "r",
                )
                if choice == "q":
                    return False
                if choice == "g":
                    self.ensure_gripper_stack(force_restart=True)

    def release(self) -> bool:
        """Operator-confirmed open at the end. True if the gripper was opened."""
        c = self.console
        try:
            c.enter("Hold the apple (or have something under it); Enter opens the gripper")
            self.gripper("open")
            return True
        except (KeyboardInterrupt, EOFError):
            c.say(f"\n!! Gripper still closed. Release it with: {MANUAL_RELEASE}")
            return False
        except GripperError as exc:
            c.say(f"!! open failed ({exc}). Release it with: {MANUAL_RELEASE}")
            return False

    def air_off(self) -> None:
        """Switch the valve off directly. The node's release turns the air off on a timer
        *after* it has already answered (lfd_automatic_gripper.fingers_and_valve_reset),
        so stopping the stack right after an ``open`` would leave the air on."""
        try:
            self.gripper("air-off")
        except GripperError as exc:
            self.console.say(f"!! air-off failed ({exc}). Turn it off with: python -m real_robot_exps.gripper_test air-off")

    def rollout_path(self, apple: int) -> Path:
        """``rollout.npz`` for the first apple, ``rollout_02.npz`` and so on after that."""
        base = Path(self.args.log) if self.args.log else self.log_dir / "rollout.npz"
        if apple <= 1:
            return base
        return base.with_name(f"{base.stem}_{apple:02d}{base.suffix}")

    def run(self) -> int:
        c = self.console
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_gripper_stack()
        log = new_rollout_log()
        apple = 1
        try:
            while True:
                if not self.grasp_until_accepted(apple):
                    return 1
                c.banner(f"apple {apple}, step 2/2: run the policy")
                if self.step_run_policy(log):
                    # A finished pick used to return here, which shut the arm down.
                    if not c.yes("Move to the next apple?", default=False):
                        return 0
                    save_rollout_log(log, self.rollout_path(apple), say=c.say)
                    log = new_rollout_log()
                    apple += 1
                    self.env.robot.end_control()  # hand-guiding needs torque mode off
                    if not self.release():
                        return 1
                    c.say("F/T is calibrated again with the gripper free, before the next grasp.")
                    continue
                # Refused as out of distribution: nothing ran, so let the operator reposition.
                if c.choose("Reposition?", {"g": "open the gripper and redo the grasp", "q": "quit"},
                            default="g") == "q":
                    return 2
                self.env.robot.end_control()  # hand-guiding needs torque mode off
                self.gripper("open")
        finally:
            self.env.close()
            save_rollout_log(log, self.rollout_path(apple), say=c.say)
            released = self.release() if self.gripper_closed else True
            # Leave the stack up while the apple is still held, so MANUAL_RELEASE works.
            if released:
                self.air_off()
                if self._stack is not None:
                    self._stack.stop()


def build_parser() -> argparse.ArgumentParser:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="checkpoint dir (ckpt_<timestep>/, agent.pt + meta.json)")
    p.add_argument("--config", default=str(REPO_ROOT / "real_robot_exps" / "config.yaml"))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--mock", action="store_true", help="mock the arm (mock_pylibfranka)")
    p.add_argument("--control-rate-hz", type=float, default=60.0)
    p.add_argument("--cage-pos-m", type=float, default=0.12)
    p.add_argument("--cage-rot-rad", type=float, default=0.5)
    p.add_argument("--allow-ood", action="store_true",
                   help="run even if the start pose is outside the training distribution")
    p.add_argument("--log-dir", default=f"~/policy_runs/{stamp}", help="gripper stack log + rollout.npz")
    p.add_argument("--log", default=None, help="rollout .npz path (default: <log-dir>/rollout.npz)")
    p.add_argument("--mock-gripper", action="store_true", help="mock the gripper (no ROS)")
    p.add_argument("--no-gripper-stack", action="store_true",
                   help="don't launch the gripper controller (you started lfd_gripper.launch.py yourself)")
    p.add_argument("--gripper-ssid", default=DEFAULT_SSID, help="Wi-Fi hotspot SSID for the ESP32 gripper controller")
    p.add_argument("--gripper-password", default=DEFAULT_PASSWORD, help="Wi-Fi hotspot password")
    p.add_argument("--ros-ws", default=str(DEFAULT_ROS_WS))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy = HarvestPolicy(args.checkpoint)
    except RuntimeError as exc:  # rl_policy_env.load_actor_checkpoint's contract checks
        print(f"Refusing checkpoint: {exc}")
        return 2
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.mock:
        config.setdefault("robot", {})["use_mock"] = True
    env = FrankaVicHarvestEnv(
        config, policy.action_bounds, policy.max_episode_steps,
        cage_pos_m=args.cage_pos_m, cage_rot_rad=args.cage_rot_rad,
        control_rate_hz=args.control_rate_hz,
    )
    session = PolicySession(args, policy=policy, env=env)
    try:
        return session.run()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
