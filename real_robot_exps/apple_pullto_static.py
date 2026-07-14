"""
Controller Verification Script

Moves the robot 5cm along each axis (X, Y, Z) using the same torque control
pipeline as pro_real_robot_eval.py, then returns home. Verifies the controller
converges by checking for 10 consecutive frames with position change < 0.1mm.

Tracks orientation throughout — the robot should maintain its home orientation
(from hand_init_orn config) during all moves. Reports per-axis RPY error in degrees.

All gains are loaded from the real robot config (control_gains section).
No WandB tag or training checkpoints required.

Usage:
    python -m real_robot_exps.controller_test --config real_robot_exps/config.yaml
"""

import argparse
import math
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import yaml

from real_robot_exps.pro_robot_interface import FrankaInterface, make_ee_target_pose, make_ee_target_pose_from_matrix
from real_robot_exps.hybrid_controller import (
    ControlTargets, get_euler_xyz, compute_pose_error, compute_pose_task_wrench,
)


CONVERGE_THRESHOLD = 1e-4  # 0.1mm
CONVERGE_FRAMES = 10
MAX_STEPS = 500             # ~33s at 15Hz safety cap
MOVE_DISTANCE = 0.02     # 5cm


def load_gains_from_config(real_config: dict, device: str = "cpu") -> dict:
    """Load controller gains from real robot config control_gains section.

    Returns dict with all tensors needed to build ControlTargets.
    """
    gains_cfg = real_config.get('control_gains', {})

    if 'task_prop_gains' not in gains_cfg:
        raise RuntimeError("control_gains.task_prop_gains not found in config")
    if 'task_deriv_gains' not in gains_cfg:
        raise RuntimeError("control_gains.task_deriv_gains not found in config")

    task_prop_gains = torch.tensor(gains_cfg['task_prop_gains'], device=device, dtype=torch.float32)
    task_deriv_gains = torch.tensor(gains_cfg['task_deriv_gains'], device=device, dtype=torch.float32)
    kp_null = gains_cfg.get('kp_null', 0.0)
    kd_null = gains_cfg.get('kd_null', 0.0)

    if gains_cfg.get('singularity_damping_enabled', False):
        singularity_damping = gains_cfg.get('singularity_damping_lambda', 0.01)
    else:
        singularity_damping = 0.0

    partial_inertia_decoupling = gains_cfg.get('partial_inertia_decoupling', False)
    sep_ori = gains_cfg.get('sep_ori', False)
    pose_ki = torch.tensor(gains_cfg.get('pose_ki',  [2.0, 2.0, 2.0, 0.0, 0.0, 0.0]), device=device, dtype=torch.float32)
    pose_integral_clamp = gains_cfg.get('pose_integral_clamp', 50.0)
    pose_integral_reset_on_target = gains_cfg.get('pose_integral_reset_on_target', True)

    # Mutual exclusion check
    if partial_inertia_decoupling and sep_ori:
        raise RuntimeError(
            "partial_inertia_decoupling and sep_ori are mutually exclusive — "
            "set only one to true in control_gains"
        )

    # print(f"  task_prop_gains:  {task_prop_gains.tolist()}")
    # print(f"  task_deriv_gains: {task_deriv_gains.tolist()}")
    # print(f"  kp_null: {kp_null}, kd_null: {kd_null}")
    # print(f"  singularity_damping: {singularity_damping}")
    # print(f"  partial_inertia_decoupling: {partial_inertia_decoupling}")
    # print(f"  sep_ori: {sep_ori}")

    return {
        'task_prop_gains': task_prop_gains,
        'task_deriv_gains': task_deriv_gains,
        'kp_null': kp_null,
        'kd_null': kd_null,
        'singularity_damping': singularity_damping,
        'partial_inertia_decoupling': partial_inertia_decoupling,
        'sep_ori': sep_ori,
        'pose_ki': pose_ki,
        'pose_integral_clamp': pose_integral_clamp,
        'pose_integral_reset_on_target': pose_integral_reset_on_target
    }


def build_position_targets(
    gains: dict,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    default_dof_pos: torch.Tensor,
    device: str = "cpu",
) -> ControlTargets:
    """Build ControlTargets for pure position control to a fixed target.

    Sets goal_position = target_pos so bounds constraint doesn't interfere.
    """
    # pos_bounds set large enough to never clamp
    pos_bounds = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)

    return ControlTargets(
        target_pos=target_pos,
        target_quat=target_quat,
        target_force=torch.zeros(6, device=device),
        sel_matrix=torch.zeros(6, device=device),
        task_prop_gains=gains['task_prop_gains'],
        task_deriv_gains=gains['task_deriv_gains'],
        force_kp=torch.zeros(6, device=device),
        force_di_wrench=torch.zeros(6, device=device),
        default_dof_pos=default_dof_pos,
        kp_null=gains['kp_null'],
        kd_null=gains['kd_null'],
        pos_bounds=pos_bounds,
        goal_position=target_pos,
        ctrl_mode="force_only",
        singularity_damping=gains['singularity_damping'],
        partial_inertia_decoupling=gains['partial_inertia_decoupling'],
        sep_ori=gains['sep_ori'],
        pose_ki=gains['pose_ki'],
        pose_integral_clamp=gains['pose_integral_clamp'],
        pose_integral_reset_on_target=gains['pose_integral_reset_on_target']
    )


def _quat_to_rpy_deg(quat: torch.Tensor) -> list:
    """Convert quaternion (w,x,y,z) to [roll, pitch, yaw] in degrees."""
    roll, pitch, yaw = get_euler_xyz(quat)
    return [math.degrees(roll.item()), math.degrees(pitch.item()), math.degrees(yaw.item())]


def run_move(
    robot: FrankaInterface,
    gains: dict,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    default_dof_pos: torch.Tensor,
    label: str,
    device: str = "cpu",
    prnt: bool = True,
    manage_control: bool = True,
) -> dict:
    """Run torque control until robot converges to target_pos.

    Returns dict with position/orientation start, target, achieved, errors, and steps.
    """
    targets = build_position_targets(gains, target_pos, target_quat, default_dof_pos, device)

    if manage_control:
        robot.start_torque_mode()

    snap = robot.get_state_snapshot()
    start_pos = snap.ee_pos.clone()
    start_quat = snap.ee_quat.clone()
    prev_pos = snap.ee_pos.clone()
    converge_count = 0

    for step in range(MAX_STEPS):
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        robot.check_safety(snap)

        robot.set_control_targets(targets)

        # Debug: replicate wrench computation from compute process for visibility
        if step == 0 or step % 50 == 0 or converge_count == CONVERGE_FRAMES - 1:
            pos_err, aa_err = compute_pose_error(
                snap.ee_pos, snap.ee_quat, target_pos, target_quat,
            )
            wrench = compute_pose_task_wrench(
                snap.ee_pos, snap.ee_quat, snap.ee_linvel, snap.ee_angvel,
                target_pos, target_quat,
                gains['task_prop_gains'], gains['task_deriv_gains'],
            )
            if(prnt):
                print(f"    [step {step:3d}] orn_error (axis-angle, base frame): "
                    f"[{aa_err[0].item():.6f}, {aa_err[1].item():.6f}, {aa_err[2].item():.6f}]")
                print(f"    [step {step:3d}] wrench [Fx,Fy,Fz,Tx,Ty,Tz] (base frame, pre-Lambda): "
                    f"[{wrench[0].item():.4f}, {wrench[1].item():.4f}, {wrench[2].item():.4f}, "
                    f"{wrench[3].item():.4f}, {wrench[4].item():.4f}, {wrench[5].item():.4f}]")
                print(f"    [step {step:3d}] frame: geometric Jacobian base frame "
                    f"(q_err = q_target * q_current^-1 -> axis-angle)")

        pos_delta = torch.norm(snap.ee_pos - prev_pos).item()
        prev_pos = snap.ee_pos.clone()

        if pos_delta < CONVERGE_THRESHOLD:
            converge_count += 1
        else:
            converge_count = 0

        if converge_count >= CONVERGE_FRAMES:
            break
    if manage_control:
        robot.end_control()

    # Position results
    achieved_pos = snap.ee_pos.clone()
    pos_error = (achieved_pos - target_pos).tolist()
    pos_error_norm = torch.norm(achieved_pos - target_pos).item()

    # Orientation results
    achieved_quat = snap.ee_quat.clone()
    target_rpy_deg = _quat_to_rpy_deg(target_quat)
    start_rpy_deg = _quat_to_rpy_deg(start_quat)
    achieved_rpy_deg = _quat_to_rpy_deg(achieved_quat)
    orn_error_deg = [(achieved_rpy_deg[i] - target_rpy_deg[i] + 180.0) % 360.0 - 180.0 for i in range(3)]

    steps_used = step + 1
    converged = converge_count >= CONVERGE_FRAMES

    if(prnt):
        print(f"  [{label}]")
        print(f"    Start Pos:      [{start_pos[0].item():.5f}, {start_pos[1].item():.5f}, {start_pos[2].item():.5f}]")
        print(f"    Target Pos:     [{target_pos[0].item():.5f}, {target_pos[1].item():.5f}, {target_pos[2].item():.5f}]")
        print(f"    Achieved Pos:   [{achieved_pos[0].item():.5f}, {achieved_pos[1].item():.5f}, {achieved_pos[2].item():.5f}]")
        print(f"    Pos Error:      [{pos_error[0]:.5f}, {pos_error[1]:.5f}, {pos_error[2]:.5f}] (norm={pos_error_norm*1000:.2f}mm)")
        print(f"    Orn Target  (RPY deg): [{target_rpy_deg[0]:.2f}, {target_rpy_deg[1]:.2f}, {target_rpy_deg[2]:.2f}]")
        print(f"    Orn Start   (RPY deg): [{start_rpy_deg[0]:.2f}, {start_rpy_deg[1]:.2f}, {start_rpy_deg[2]:.2f}]")
        print(f"    Orn Achieved(RPY deg): [{achieved_rpy_deg[0]:.2f}, {achieved_rpy_deg[1]:.2f}, {achieved_rpy_deg[2]:.2f}]")
        print(f"    Orn Error   (RPY deg): [{orn_error_deg[0]:.2f}, {orn_error_deg[1]:.2f}, {orn_error_deg[2]:.2f}]")
        print(f"    Steps:    {steps_used} ({'converged' if converged else 'MAX STEPS'})")

    return {
        'label': label,
        'start_pos': start_pos,
        'target_pos': target_pos,
        'achieved_pos': achieved_pos,
        'pos_error': pos_error,
        'pos_error_norm': pos_error_norm,
        'target_rpy_deg': target_rpy_deg,
        'start_rpy_deg': start_rpy_deg,
        'achieved_rpy_deg': achieved_rpy_deg,
        'orn_error_deg': orn_error_deg,
        'steps': steps_used,
        'converged': converged,
    }

import pandas as pd
import matplotlib.pyplot as plt

def hold_and_record(robot: FrankaInterface, gains, target_pos, target_quat, default_dof_pos, duration_sec, device="cpu", base_ft=np.array([0,0,0,0,0,0])):
    targets = build_position_targets(gains, target_pos, target_quat, default_dof_pos, device)
    steps = int(duration_sec * robot._control_rate_hz)
    ft_history = []
    
    for _ in range(steps):
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        robot.check_safety(snap)
        robot.set_control_targets(targets)
        
        # Grab the raw body-frame force and subtract the torque-mode tare
        
        ft = snap.force_torque.cpu().numpy()

        ft -= base_ft
        ft_history.append(ft)
    
    # print(ft_history)

        
    return np.array(ft_history)

def plot_and_save_data(raw_ft_data, label="pull", window_size=5, baseline=False, plot=True):
    """Saves raw/smooth CSVs and plots the Fx, Fy, Fz forces."""
    # Create DataFrame
    cols = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    df_raw = pd.DataFrame(raw_ft_data, columns=cols)
   
    # Save to CSV
    df_raw.to_csv(f"{label}.csv", index=False)
    
    # Plot forces
    if(plot):
        plt.figure(figsize=(10, 5))
        for axis, color in zip(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"], ['r', 'g', 'b', 'yellow', 'teal', 'purple']):
            plt.plot(df_raw[axis], color=color, alpha=1.0, label=f"Raw {axis}")
            
        plt.title(f"Force/Torque Profile: {label}")
        plt.xlabel("Policy Steps (15Hz)")
        plt.ylabel("Force (N) / Torque (Nm)")
        plt.legend()
        plt.grid(True)
        plt.show()

def hold_position(
    robot: FrankaInterface,
    gains: dict,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    default_dof_pos: torch.Tensor,
    duration_sec: float,
    device: str = "cpu"
):
    """Actively hold a Cartesian pose while maintaining the 15Hz safety/timing loop."""
    targets = build_position_targets(gains, target_pos, target_quat, default_dof_pos, device)
    
    # 15Hz * duration = number of steps
    steps = int(duration_sec * robot._control_rate_hz)
    
    for _ in range(steps):
        robot.wait_for_policy_step()
        snap = robot.get_state_snapshot()
        robot.check_safety(snap)
        robot.set_control_targets(targets)

def update_gains(gains, new_prop_gains, device):

    gains["task_prop_gains"] = torch.tensor(new_prop_gains, device=device, dtype=torch.float32)
    derivs = [0, 0, 0, 0, 0, 0]
    for i in range(len(new_prop_gains)):
        derivs[i] = 1.75 * math.sqrt(new_prop_gains[i])
    gains["task_deriv_gains"] = torch.tensor(derivs, device=device, dtype=torch.float32)
    return gains

def pull_test(theta, phi, robot: FrankaInterface, apple_pose_4x4, default_dof_pos, gains, home_pose_4x4, gc, device: str = "cpu", baseline: bool = False, debug: bool = False, to_plot: bool = False, distance: float = 0.05, stops: int = 5):
    
    time.sleep(2.0) # let it settle
    robot.reset_to_start_pose(apple_pose_4x4)
    snap = robot.get_state_snapshot()
    #base_ft = snap.force_torque.cpu().numpy()
    #print(f"post cartesian... base_ft {base_ft}")


    #new_ft_bias = robot.calibrate_ft_bias()
    #print(f"new ft_bias is {new_ft_bias}")
    gc.send_request(True)
    time.sleep(2)
    
    # distance = .05
    # stops = 5
    steps = stops
    if(distance/steps > 0.01):
        steps *= 2
    print(f"steps is... {steps}")

    robot.start_torque_mode()
    
    # if debug:
    #     print("Settling torque controller...")
    # time.sleep(1.0) 
    
    
    snap = robot.get_state_snapshot()
    #base_ft = snap.force_torque.cpu().numpy()
    #print(f"base_ft {base_ft}")
    target = snap.ee_pos.clone()
    #theta = math.pi/2 #'roll' pi/3 to 2pi/3
    #phi = math.pi/4 # 'pitch'
    dx = distance * math.sin(theta) * math.cos(phi)
    dy = distance * math.sin(theta) * math.sin(phi)
    dz = distance * math.cos(theta)

    pull_data = []

    for i in range(steps):
        if(debug):
            print(f"Starting {i+1} of {steps}...")
        
        
        target[0] -= (dx/steps)
        target[1] -= (dy/steps)
        target[2] -= (dz/steps)
        apple_quat = snap.ee_quat.clone()
        #gains[""]
        run_move(robot, gains, target, apple_quat, default_dof_pos, f"closer #{i}", prnt=debug, manage_control=False)
        
        if((i+1) % (steps/stops) == 0):
            s = 1
            if(debug):
                print(f"Holding position for {s}s...")
            data = hold_and_record(robot, gains, target, apple_quat, default_dof_pos, duration_sec=s, device=device)
            pull_data.append(data)
            #hold_position(robot, gains, target, apple_quat, default_dof_pos, duration_sec=s, device=device)
        
    
    # 1. Zero out the PD error so the arm stops trying to pull
    if(debug):
        print("Relaxing tension before release...")
    snap = robot.get_state_snapshot()
    hold_position(robot, gains, snap.ee_pos, snap.ee_quat, default_dof_pos, duration_sec=0.5, device=device)
    
    # 2. Safely open gripper (arm will not snap because PD error is 0)
    gc.send_request(False)
    
    # 3. Hold position a little longer to let the physical shake settle
    hold_position(robot, gains, snap.ee_pos, snap.ee_quat, default_dof_pos, duration_sec=0.5, device=device)
    
    # 4. Safely drop out of torque mode
    robot.end_control()

    # 5. Wait a moment before the Cartesian reset
    full_pull_data = np.concatenate(pull_data, axis=0)
    label = f"pull_theta{theta:.2f}_phi{phi:.2f}"
    if baseline:
        label += "_baseline"
    plot_and_save_data(full_pull_data, label=label, plot=to_plot)
    
    time.sleep(2)
    robot.reset_to_start_pose(home_pose_4x4)



def main():


    from real_robot_exps.gripper_test import GripperClient
    gc = GripperClient()
    



    parser = argparse.ArgumentParser(description="Controller Verification Test")
    parser.add_argument("--config", type=str, default="real_robot_exps/config.yaml", help="Real robot config path")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--override", action="append", default=[], help="Override config values")
    parser.add_argument("--mode", type=str, default="collect", choices=["collect", "baseline"], help="collect/baseline")
    parser.add_argument("--plot", default=None, action="store_true", help="True/False[default]")
    parser.add_argument("--debug", default="none", help="none/all/...")
    parser.add_argument("--kp", type=int, default=80, help="kp from 20-120 (kd is auto calculated)")
    parser.add_argument("--distance", type=float, default=0.05, help="pull distance in meters (0.01 to 0.075)")
    parser.add_argument("--stops", type=int, default=5, help="number of stops to record data during pull")
    parser.add_argument("--theta", type=float, default=2.36, help="angle determining height of pull (z-direction) in radians")
    parser.add_argument("--phi", type=float, default=1.57, help="angle determining left/right of pull (circle on xy plane) in radians")
    args = parser.parse_args()

    device = args.device
    mode = args.mode # collect or baseline
    to_plot = args.plot is not None
    debug = args.debug
    kp = args.kp
    distance = args.distance
    stops = args.stops
    theta = args.theta
    phi = args.phi
    is_baseline = (mode == "baseline")

    if (mode != "collect") and (mode != "baseline"):
        print("Invalid mode command. Should be 'collect' or 'baseline'")
        sys.exit()

    if(debug != "none"):
        print("=" * 80)
        print("CONTROLLER VERIFICATION TEST")
        print("=" * 80)

    # 1. Load config
    if(debug != "none"):
        print(f"\nLoading config: {args.config}")

    with open(args.config, 'r') as f:
        real_config = yaml.safe_load(f)

    if args.override:
        for override in args.override:
            if '=' not in override:
                raise ValueError(f"Override must be 'key=value', got: {override}")
            key_path, value_str = override.split('=', 1)
            keys = key_path.split('.')
            parent = real_config
            for k in keys[:-1]:
                parent = parent[k]
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    if value_str.lower() == 'true':
                        value = True
                    elif value_str.lower() == 'false':
                        value = False
                    else:
                        value = value_str
            parent[keys[-1]] = value
            print(f"  Override: {key_path} = {value}")

    # 2. Load gains from config
    if debug != "none":
        print("\nLoading controller gains...")
    gains = load_gains_from_config(real_config, device)

   
    import pylibfranka as plf

    robot_cfg = real_config['robot']
    diag_robot = plf.Robot(robot_cfg['ip'])

    # Set NE_T_EE and EE_T_K exactly as FrankaInterface does
    NE_T_EE_cfg = robot_cfg.get('NE_T_EE', [
        0.7071, -0.7071, 0.0, 0.0,
        0.7071,  0.7071, 0.0, 0.0,
        0.0,     0.0,    1.0, 0.0,
        0.0,     0.0,    0.1034, 1.0,
    ])
    EE_T_K_cfg = robot_cfg.get('EE_T_K', [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
    
    diag_robot.set_EE(NE_T_EE_cfg)
    diag_robot.set_K(EE_T_K_cfg)


    diag_state = diag_robot.read_once()

    T = np.array(diag_state.O_T_EE)
    R = np.array([
        [T[0], T[4], T[8]],
        [T[1], T[5], T[9]],
        [T[2], T[6], T[10]],
    ])
    pos = T[12:15]

   
    diag_robot.stop()


    # 3. Initialize robot
    if(debug != "none"):
        print("\nInitializing robot interface...")
    robot = FrankaInterface(real_config, device=device)

 

    home_pose_4x4 = make_ee_target_pose_from_matrix(pos, R)

    # arbitrarily chosen 'home'
    home_rot = np.array([[-1, 0, 0.0], [0.0, 0.0, 1.0], [0, 1, 0]])
    home_pos = np.array([0.0, 0.85, 0.42])
    home_pose_4x4 = make_ee_target_pose_from_matrix(home_pos, home_rot)

    apple_rot = np.array([
                 [ -0.994, -.110, 0.00],
                 [0, 0, 1.000],
                 [-0.11,  .991,  0 ]
                 ])
    # print(R)
    # print(apple_rot)
    apple_pose_4x4 = make_ee_target_pose_from_matrix(np.array([0, .9262, .41]), apple_rot)

    # 6. Move to home and wait for user
    print("\nMoving to home position...")
    robot.reset_to_start_pose(home_pose_4x4)
    snap = robot.get_state_snapshot()
    home_actual = snap.ee_pos.clone()
    home_quat = snap.ee_quat.clone()
    default_dof_pos = snap.joint_pos.clone()
    home_rpy_deg = _quat_to_rpy_deg(home_quat)
    # print(f"  Home Pos: [{home_actual[0].item():.5f}, {home_actual[1].item():.5f}, {home_actual[2].item():.5f}]")
    # print(f"  Home Orn (RPY deg): [{home_rpy_deg[0]:.2f}, {home_rpy_deg[1]:.2f}, {home_rpy_deg[2]:.2f}]")   

    input("  Press Enter to begin apple pull run...")


    #target = home_actual.clone()
    #target[axis] += MOVE_DISTANCE

    # Move to target (orientation goal = home_quat, should not change)

    target = apple_pose_4x4
    

    #run_move(robot, gains, home_actual, home_quat, default_dof_pos, "Apple", device)
    
   

    # snap = robot.get_state_snapshot()
    # target = snap.ee_pos.clone()
    # target[1] -= .04
    # apple_quat = snap.ee_quat.clone()

    # kp = 80 # 50 is default
    gains = update_gains(gains, [kp, kp, kp, 30, 30, 30], device)
    # print(gains["task_prop_gains"])
    # print(gains["task_deriv_gains"])
    
    # run_move(robot, gains, target, apple_quat, default_dof_pos, "pull apple", device)
    # time.sleep(1.5)

    pi = math.pi
    left = (pi/2, 0)
    back_left = (pi/2, pi/4)
    back = (pi/2, pi/2)
    back_right = pi/2, 3*pi/4
    right = (pi/2, pi)
    up_back_left = (3*pi/4, pi/4)
    up_back = (3*pi/4, pi/2)
    up_back_right = (3*pi/4, 3*pi/4)
    angles = [up_back_left, up_back, up_back_right, back_left, back, back_right]
    angles = [up_back_left, up_back, up_back_right]
    angles = [(theta, phi)]
    for (theta, phi) in angles:
        pull_test(theta, phi, robot, apple_pose_4x4, default_dof_pos, gains, home_pose_4x4, gc, device=device, baseline=is_baseline, to_plot=to_plot, debug=(debug != "none"), distance=distance, stops=stops)

     

    # 9. Shutdown
    robot.shutdown()
    gc.terminate()
   

if __name__ == "__main__":
    main()
