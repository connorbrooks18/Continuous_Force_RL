"""
Torque-Mode F/T Calibration Script

Moves to the apple pull starting pose, actively holds position in torque mode 
(with gripper closed to simulate payload), and computes an accurate F/T bias.

Usage:
    python -m real_robot_exps.calibrate_torque_bias --config real_robot_exps/config.yaml
"""

import argparse
import sys
import time
import numpy as np
import torch
import yaml

from real_robot_exps.pro_robot_interface import FrankaInterface, make_ee_target_pose_from_matrix
from real_robot_exps.apple_pullto_static import load_gains_from_config, build_position_targets, hold_and_record, update_gains

def main():
    parser = argparse.ArgumentParser(description="Torque-Mode F/T Calibration")
    parser.add_argument("--config", type=str, default="real_robot_exps/config.yaml", help="Path to config")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--kp", type=int, default=80, help="Kp gain for active hold")
    parser.add_argument("--duration", type=float, default=5.0, help="Calibration duration in seconds")
    args = parser.parse_args()

    device = args.device
    kp = args.kp

    print("=" * 80)
    print("TORQUE-MODE F/T BIAS CALIBRATION")
    print("=" * 80)

    # 1. Load config and temporarily clear existing bias to read raw values
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # We pass None for bias to the interface so we can read the raw un-biased values
    raw_ft_bias = [0.0] * 6

    # 2. Setup gains
    gains = load_gains_from_config(config, device)
    gains = update_gains(gains, [kp, kp, kp, 30, 30, 30], device)

    # 3. Connect to gripper service (needs gripper closed to capture payload weight)
    from real_robot_exps.gripper_test import GripperClient
    gc = GripperClient()

    # 4. Initialize robot with raw F/T tracking (zeroed bias)
    print("\nInitializing robot interface (bypassing old config bias)...")
    robot = FrankaInterface(config, device=device)
    robot._ft_bias = raw_ft_bias 

    # Define the exact apple pose
    apple_rot = np.array([
        [-0.994, -0.110,  0.000],
        [ 0.000,  0.000,  1.000],
        [-0.110,  0.991,  0.000]
    ])
    apple_pos = np.array([0.0, 0.9262, 0.41])
    apple_pose_4x4 = make_ee_target_pose_from_matrix(apple_pos, apple_rot)

    # 5. Move to apple pose under safe Cartesian position control
    print("\nMoving to apple pose...")
    robot.reset_to_start_pose(apple_pose_4x4)
    time.sleep(1.0)

    snap = robot.get_state_snapshot()
    target_pos = snap.ee_pos.clone()
    target_quat = snap.ee_quat.clone()
    default_dof_pos = snap.joint_pos.clone()

    # 6. Close gripper in free space to mimic exact pull conditions
    input("\n[Action] Ensure gripper is NOT touching the apple. Press Enter to close gripper...")
    gc.send_request(True)
    time.sleep(2.0)  # wait for grip to settle

    # 7. Run Active Hold in Torque Mode and record raw forces
    print(f"\nActively holding pose in torque mode for {args.duration}s to calibrate...")
    robot.start_torque_mode()
    
    raw_ft_data = hold_and_record(
        robot, gains, target_pos, target_quat, default_dof_pos, 
        duration_sec=args.duration, device=device
    )
    
    robot.end_control()
    gc.send_request(False)  # open gripper for safety

    # 8. Compute and display bias results
    # Negate the captured force_torque because get_state_snapshot() flips the sign
    # to represent env-on-robot forces. We need raw O_F_ext_hat_K values.
    mean_raw_forces = -np.mean(raw_ft_data, axis=0) 
    
    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE")
    print("=" * 80)
    print(f"Calculated ft_bias: {[round(x, 4) for x in mean_raw_forces.tolist()]}")
    print("\nPaste this line directly into your config.yaml control_gains section:")
    print(f"  ft_bias: {[round(x, 4) for x in mean_raw_forces.tolist()]}")
    print("=" * 80)

    # 9. Shutdown
    robot.shutdown()
    gc.terminate()

if __name__ == "__main__":
    main()