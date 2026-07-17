# Apple Pull Test & Controller Verification

This script executes a parameterized, multi-stop pull test using a Franka FR3 robot to simulate pulling an apple from a branch. It utilizes a hybrid force/position controller to execute trajectory steps, records Wrench (Force/Torque) data at each stop, and exports the results to CSV files and plots.

Eventually, the pipeline will include doing a dry run / baseline run, calculating the interaction force `F_{int} = F_m - F_b` where `F_m and F_b` are measured and baseline force/torque measurements respectively.

## Prerequisites
* Franka arm configured with `pylibfranka`.
* Active ROS 2 environment with the `gripper_grab_client` service running from https://github.com/connorbrooks18/lfd_apples.
* Required Python packages: `torch`, `numpy`, `pandas`, `matplotlib`, `pyyaml`.

## Usage
Run the script as a module from the root of your workspace:

python -m real_robot_exps.apple_pullto_static [OPTIONS]

### Examples

```bash
# Standard Data Collection (Defaults: 5cm, 5 stops, up-back pull):
python -m real_robot_exps.apple_pullto_static --mode collect --plot`

# Baseline Collection (High Stiffness, Custom Distance & Stops):
python -m real_robot_exps.apple_pullto_static --mode baseline --kp 100 --distance 0.075 --stops 10

# Custom Angle Pull (Horizontal Back-Left):
python -m real_robot_exps.apple_pullto_static --distance 0.05 --stops 5 --theta 1.57 --phi 0.79

# Full Example
python -m real_robot_exps.apple_pullto_static --config real_robot_exps/config.yaml --mode collect --theta 2.36 --phi 2.36 --distance 0.04 --stops 4 --plot --kp 100
```

## Command Line Arguments

All arguments are passed as optional flags.

### Core Pull Parameters:
* `--distance` (float, default: 0.05): The total pull distance in meters (recommended 0.01 to 0.075).
* `--stops` (int, default: 5): The number of discrete stops along the pull trajectory where data is recorded.
* `--theta` (float, default: 2.36): The inclination angle determining the Z-axis height of the pull in radians.
* `--phi` (float, default: 1.57): The azimuthal angle determining the left/right direction on the XY plane in radians.

### Configuration & System:
* `--config` (str, default: real_robot_exps/config.yaml): Path to the real robot configuration YAML file.
* `--device` (str, default: cpu): Torch device to use for tensor operations.
* `--mode` (str, default: collect): Operation mode. Use 'collect' for standard runs or 'baseline' to append a baseline tag to the output files.
* `--plot`: If provided, displays a matplotlib graph of the F/T data at the end of each pull.
* `--debug` (str, default: none): Set to "all" to print verbose step-by-step wrench and trajectory data.
* `--kp` (int, default: 80): The proportional gain for the controller (recommended 20-120). Derivative gains are automatically calculated.
* `--override` (str): Append config overrides in key=value format (e.g., --override robot.gripper_force_n=60).

## Understanding Pull Angles (theta & phi)

The script calculates the pull trajectory using a spherical coordinate system. Because the script subtracts the calculated vector from the target position, the arm pulls *away* from the origin (the apple). 

* `theta`: Controls the vertical trajectory (inclination/altitude).
  * 1.57 (pi/2): Purely horizontal pull.
  * 2.36 (3pi/4): Pulls upward.
* `phi`: Controls the horizontal (left/right/back) trajectory on the XY plane (azimuth). (0 to pi to stay on front semisphere)

### Common Angle Configurations (Radians)

Here are the exact arguments you can pass to `--theta` and `--phi` to achieve specific pull directions:

| Direction         | --theta | --phi  | Formula           |
|-------------------|---------|--------|-------------------|
| Left              | 1.57    | 0.00   | pi/2 / 0          |
| Back-Left         | 1.57    | 0.79   | pi/2 / pi/4       |
| Back              | 1.57    | 1.57   | pi/2 / pi/2       |
| Back-Right        | 1.57    | 2.36   | pi/2 / 3pi/4      |
| Right             | 1.57    | 3.14   | pi/2 / pi         |
| Up-Back-Left      | 2.36    | 0.79   | 3pi/4 / pi/4      |
| Up-Back (Default) | 2.36    | 1.57   | 3pi/4 / pi/2      |
| Up-Back-Right     | 2.36    | 2.36   | 3pi/4 / 3pi/4     |

## Outputs
For each test run, the script generates:
1. CSV Data (pull_thetaX_phiY[_baseline].csv): Contains the concatenated raw F/T readings (Fx, Fy, Fz, Tx, Ty, Tz) across all stops.
2. Plots: A 6-axis line chart of the force/torque profile over the 15Hz policy steps (if --plot is enabled).

# Computing Interaction Force

```bash

# This preassumens a raw and a baseline file for a given theta and phi as "pull_theta2.36_phi2.36_[raw/baseline].csv with same distance and stop #
python compute_interaction.py --theta 2.36 --phi 1.57 --plot

```

## Unified static system-ID Parquet

The camera and robot collectors run as separate processes on the same computer
and timestamp every sample with Unix wall-clock seconds from `time.time()`.
Start the camera before the robot so it records frames at the robot's rest
reference timestamp.

```bash
# Terminal 1: press q after the robot run finishes.
cd at-tracking
python Detecting.py --output tracking.parquet

# Terminal 2: also writes the legacy wrench CSV.
python -m real_robot_exps.apple_pullto_static --config real_robot_exps/config.yaml --mode collect --theta 2.36 --phi 2.36 --distance 0.04 --stops 4

# After both collection processes have stopped:
python -m real_robot_exps.compile_static_sysid \
  --robot pull_robot.parquet \
  --tracking tracking.parquet \
  --output pull_unified.parquet \
  --camera-frames 5
```

The compiler keeps all robot-rate static-hold rows and attaches a median camera
estimate made from a few complete `Branch`, `Spur`, and `Apple` frames. The
three woody parts are ordered as `fruiting_base -> Branch`, `Branch -> Spur`,
and `Spur -> Apple`. For now, the fruiting base is the reference AprilTag
origin; `--fruiting-base-pos X Y Z` can override it after calibration.

The unified file stores the model fields as fixed-size Arrow lists and embeds
collection, calibration, synchronization, topology, units, source-file hashes,
software versions, and camera-selection diagnostics in `dataset_metadata` in
the Parquet footer.

Three distinct robot-side joint-torque fields are recorded directly from the
same `pylibfranka.RobotState` sample. Each is a 7-vector in Franka joint order
`[joint_1, ..., joint_7]` (base to end effector), in N·m:

- `tau_J`: measured link-side joint torque sensor signals. This is the sensor
  measurement; this collection code does not subtract gravity from it.
- `tau_ext_hat_filtered`: libfranka's low-pass-filtered estimate of torque due
  to external forces. It is the difference between `tau_J` and model-expected
  torque and does not include configured end-effector/load or robot
  mass/dynamics contributions.
- `tau_J_d`: desired link-side joint torque signal without gravity. This is a
  desired/controller signal, not a second measurement.

The process-based robot interface transports all three signals through shared
memory. The raw robot Parquet and compiled unified Parquet retain the three
names separately; no additional bias subtraction or gravity compensation is
applied to any of them by the data-collection/compiler pipeline.

To inspect a unified file over time, run:

```bash
python -m real_robot_exps.viz_static_sysid \
  --input pull_unified.parquet \
  --save pull_unified_viz.png
```

The viewer gives `tau_J`, `tau_ext_hat_filtered`, and `tau_J_d` separate plots,
then lays out wrench, TCP velocity, action, pose, bending angles, and experiment
state on a shared timestamp axis, with hold boundaries marked for debugging.

The same viewer also works on arm-only Parquets that do not contain camera
fields. In that case it falls back to the robot signals and TCP position/time
plots, so you can still inspect a run even if AprilTag data is missing or was
not recorded.

To debug wrench zeroing and pose-dependent offsets without doing a pull, run a
short rest-pose sweep near the normal home and apple poses:

```bash
python -m real_robot_exps.ft_rest_pose_sweep \
  --config real_robot_exps/config.yaml \
  --output ft_rest_pose_sweep.parquet \
  --hold-seconds 3 \
  --offset-cm 2
```

This records the same robot-side wrench and pose fields as the pull script, but
only while the arm is stationary at a few nearby poses.
