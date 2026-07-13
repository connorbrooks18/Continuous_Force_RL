# Apple Pull Test & Controller Verification

This script executes a parameterized, multi-stop pull test using a Franka FR3 robot to simulate pulling an apple from a branch. It utilizes a hybrid force/position controller to execute trajectory steps, records Wrench (Force/Torque) data at each stop, and exports the results to CSV files and plots.

## Prerequisites
* Franka arm configured with `pylibfranka`.
* Active ROS 2 environment with the `gripper_grab_client` service running.
* Required Python packages: `torch`, `numpy`, `pandas`, `matplotlib`, `pyyaml`.

## Usage
Run the script as a module from the root of your workspace:

```bash
python -m real_robot_exps.controller_test [OPTIONS]
python -m real_robot_exps.controller_test --mode collect --plot
python -m real_robot_exps.controller_test --mode baseline --kp 100
python -m real_robot_exps.controller_test --distance 0.05 --stops 5

## Command Line Arguments

### Positional Arguments:

- distance (float, default: 0.05): The total pull distance in meters.

- stops (int, default: 5): The number of discrete stops along the pull trajectory where data is recorded.

###Optional Arguments:

- --config (str, default: real_robot_exps/config.yaml): Path to the real robot configuration YAML file.

- --device (str, default: cpu): Torch device to use for tensor operations.

- --mode (str, default: collect): Operation mode. Use collect for standard runs or baseline to append a baseline tag to the output files.

- --plot: If provided, displays a matplotlib graph of the F/T data at the end of each pull.

- --debug (str, default: none): Set to "all" to print verbose step-by-step wrench and trajectory data.

- --kp (int, default: 80): The proportional gain for the controller (recommended 20-120). Derivative gains are automatically calculated.

- --override (str): Append config overrides in key=value format (e.g., --override robot.gripper_force_n=60).

## Outputs

For each angle tested, the script generates:

- CSV Data (pull_thetaX_phiY[_baseline].csv): Contains the concatenated raw F/T readings (Fx, Fy, Fz, Tx, Ty, Tz) across all stops.

- Plots: A 6-axis line chart of the force/torque profile over the 15Hz policy steps (if --plot is enabled).