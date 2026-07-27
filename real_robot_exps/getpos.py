"""
First Real Robot Test — Safe, interactive validation of FrankaInterface on real FR3.

Runs 5 sequential phases, printing all values for manual inspection.
Waits for Enter between phases. Robot holds zero torques throughout.

Usage:
    python real_robot_exps/first_real_robot_test.py
    python real_robot_exps/first_real_robot_test.py --config real_robot_exps/config.yaml
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_robot_exps.robot_interface import FrankaInterface, make_ee_target_pose, make_ee_target_pose_from_matrix, SafetyViolation


def wait_for_enter(msg: str = "Press Enter to continue to next phase..."):
    """Block until user presses Enter."""
    input(f"\n>>> {msg}\n")


def print_separator(phase: int, title: str):
    print(f"\n{'=' * 70}")
    print(f"  PHASE {phase}: {title}")
    print(f"{'=' * 70}\n")


def phase1_connect_and_read(config: dict):
    """Phase 1: Connect to robot, read raw state (no motion)."""
    print_separator(1, "CONNECT & READ RAW STATE (no motion)")

    robot_cfg = config['robot']
    ip = robot_cfg['ip']

    # Import real pylibfranka (never mock for this test)
    import pylibfranka
    print(f"Connecting to robot at {ip}...")
    raw_robot = pylibfranka.Robot(ip)

    # Set frames
    NE_T_EE = robot_cfg['NE_T_EE']
    EE_T_K = robot_cfg['EE_T_K']
    raw_robot.set_EE(NE_T_EE)
    raw_robot.set_K(EE_T_K)
    #print("NE_T_EE and EE_T_K set.")

    # Read raw state (outside control loop)
    state = raw_robot.read_once()

    #print("\n--- Raw O_T_EE (column-major, 16 elements) ---")
    T = state.O_T_EE

    for row in range(4):
        cols = [T[row + col * 4] for col in range(4)]
        print(f"  [{cols[0]:+.6f}  {cols[1]:+.6f}  {cols[2]:+.6f}  {cols[3]:+.6f}]")
    """
    print(f"\n--- Raw joint positions q (7) ---")
    print(f"  {[f'{v:+.4f}' for v in state.q]}")

    print(f"\n--- Raw joint velocities dq (7) ---")
    print(f"  {[f'{v:+.6f}' for v in state.dq]}")

    print(f"\n--- Raw O_F_ext_hat_K (6) ---")
    print(f"  {[f'{v:+.4f}' for v in state.O_F_ext_hat_K]}")
    """
    # Extract EE position
    ee_pos = [T[12], T[13], T[14]]
    print(f"\n--- EE position from O_T_EE[12:15] ---")
    print(f"  x={ee_pos[0]:.4f}m, y={ee_pos[1]:.4f}m, z={ee_pos[2]:.4f}m")
    """
    # Sanity check: is position in reasonable workspace?
    x, y, z = ee_pos
    checks = []
    checks.append(("x in [0.1, 0.8]", 0.1 <= x <= 0.8))
    checks.append(("y in [-0.5, 0.5]", -0.5 <= y <= 0.5))
    checks.append(("z in [0.0, 0.8]", 0.0 <= z <= 0.8))

    
    print("\n--- Workspace sanity checks ---")
    all_ok = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {status}: {name} (value: {eval(name[0])})")

    if not all_ok:
        print("\n  WARNING: EE position outside expected workspace!")
        print("  Visually confirm the robot is in a safe configuration before continuing.")

    print("\n  MANUAL CHECK: Does the printed EE position match where the")
    print("  fingertip midpoint physically is? Measure with a ruler if unsure.")
    """

    # YAML-ready calibration output (cyan for visibility)
  
    raw_robot.stop()
    return ee_pos




def main():
    parser = argparse.ArgumentParser(description="First Real Robot Test")
    parser.add_argument("--config", type=str, default="real_robot_exps/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--skip-to", type=int, default=1,
                        help="Skip to phase N (1-6)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override to real robot
    if config['robot'].get('use_mock', False):
        print("WARNING: config has use_mock=true. This test requires a REAL robot.")
        print("Overriding to use_mock=false for this test.\n")
        config['robot']['use_mock'] = False

    print("=" * 70)
    print("  FIRST REAL ROBOT TEST")
    """
    print("  Safe, interactive validation of FrankaInterface on real FR3")
    print("  Robot IP:", config['robot']['ip'])
    print("=" * 70)
    print("\n  This test has 6 phases:")
    print("  1. Connect & read raw state (no motion)")
    print("  2. Frame validation (no motion)")
    print("  3. Torque control + snapshot (zero torques, background thread)")
    print("  4. Force/torque sign convention (manual push)")
    print("  5. Cartesian reset motion (2cm up)")
    print("  6. 1kHz torque recomputation stress test")
    print("\n  Each phase waits for Enter before proceeding.")
    print("  You can Ctrl+C at any time to abort.\n")
    """

    initial_ee_pos = None
    robot = None  # shared FrankaInterface for phases 3-5
    R_mat = None

    try:
        if args.skip_to <= 1:
            initial_ee_pos = phase1_connect_and_read(config)


        print("\n" + "=" * 70)
        print("  ALL PHASES COMPLETE")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nAborted by user (Ctrl+C). Robot should be safe.")
        print("If torque control was active, the robot will stop on communication timeout.")

    except SafetyViolation as e:
        print(f"\n\nSAFETY VIOLATION: {e}")
        print("Robot should stop automatically.")

    except Exception as e:
        print(f"\n\nERROR: {type(e).__name__}: {e}")
        print("Robot should stop automatically on communication timeout.")
        raise

    finally:
        if robot is not None:
            robot.shutdown()


if __name__ == "__main__":
    main()
