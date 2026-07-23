# Apple Pull Test & Controller Verification

This project executes parameterized, multi-stop pull tests using a Franka FR3
robot, records robot and optional camera data, saves metadata-rich Parquet
files, compiles synchronized system-identification episodes, and generates
diagnostic plots.

The pipeline supports an unloaded dry/baseline run followed by a matched
measurement run. It calculates interaction wrench dynamically across each
hold as `F_int = F_raw - F_baseline`.

## Prerequisites
* Franka arm configured with `pylibfranka`.
* Active ROS 2 environment with the `gripper_grab_client` service running from https://github.com/connorbrooks18/lfd_apples.
* Required Python packages: `torch`, `numpy`, `pandas`, `matplotlib`, `pyyaml`,
  `pyarrow`, `scipy`, `opencv-python`, `pyrealsense2`, and `pupil-apriltags`.

## Full collection process

The normal workflow uses one unloaded dry run as a dynamic baseline followed
by one loaded run with exactly the same trajectory. Camera tracking is needed
only for the loaded run that will be compiled into the unified dataset.

The collector now records the whole episode continuously, not just the static
holds. Each run includes:

- an initial rest-geometry sample before the apple is grabbed,
- a post-grab initial hold before motion starts,
- the moving portion of each pull segment, and
- the static hold at each stop.

These stages are marked with `sample_label`. The raw robot rows also use
special `hold_index` values for the setup frames:

- `hold_index = -2`: initial rest geometry
- `hold_index = -1`: post-grab initial hold
- `hold_index >= 0`: actual pull segments / hold groups

### 1. Choose and verify the pull setup

Near the top of `real_robot_exps/apple_pullto_static.py`, select the starting
pose and baseline behavior:

```python
USE_CLOSE_PULL_START_POSE = False
USE_DYNAMIC_BASELINE_CORRECTION = True
```

Use the same values for the baseline and loaded run. Confirm the Franka Desk
end-effector/load configuration and keep the physical gripper configuration,
controller gains, pull direction, distance, and hold count unchanged between
the two runs.

### 2. Record the unloaded dynamic baseline

Remove the apple/contact load, but otherwise preserve the loaded-run setup.
Run the complete trajectory in `baseline` mode:

```bash
python -m real_robot_exps.apple_pullto_static \
  --config real_robot_exps/config.yaml \
  --mode baseline \
  --theta 2.36 --phi 1.57 \
  --distance 0.04 --stops 4 \
  --kp 100
```

With these values, the default output is
`pull_theta2.36_phi1.57_baseline_robot.parquet`. It contains the unloaded
profile and is not itself baseline-corrected.

### 3. Record the matched loaded run and camera tracking

Run the robot collector and the standalone RealSense/AprilTag detector in
separate terminals. Keep them on the same computer so both use the same
`time.time()` clock, and start the detector before or during the pull. Tag IDs,
offsets, pose math, metadata, and Parquet writing for the camera remain in the
standalone tracking program.

```bash
python -m real_robot_exps.apple_pullto_static \
  --config real_robot_exps/config.yaml \
  --mode collect \
  --theta 2.36 --phi 1.57 \
  --distance 0.04 --stops 4 \
  --kp 100
```

The collect run locates the default baseline file, verifies compatible angles,
distance, hold count, selected pose, and exact starting transform, and then
interpolates the unloaded wrench over normalized time within each actual pull
hold (`hold_index >= 0`). It saves:

- `ft_wrist_raw`: loaded-run wrench before baseline correction.
- `ft_wrist_baseline`: interpolated unloaded wrench profile.
- `ft_wrist`: interaction wrench, `ft_wrist_raw - ft_wrist_baseline`.

IMPORTANT: `ft_wrist` force components are in the end-effector frame. The
torque components, joint states, TCP pose/position, and all camera-derived
geometry are in the Franka base frame.

The default raw outputs are `pull_theta2.36_phi1.57_raw_robot.parquet` and
`pull_theta2.36_phi1.57_raw_tracking.parquet`. If an explicitly uncorrected
collect run is needed, set `USE_DYNAMIC_BASELINE_CORRECTION = False` before
running it.

Run `python at-tracking/Detecting.py` separately for the troubleshooting feed.
It draws the reference frame and the full Branch, Spur, and Apple frames (X
red, Y green, Z blue), plus their positions and orientations. Pressing `q` or
Escape ends the detector without affecting the robot process.

### 4. Compile the synchronized episode

After both raw files are saved, compile them with:

```bash
python -m real_robot_exps.compile_static_sysid \
  --robot pull_theta2.36_phi1.57_raw_robot.parquet \
  --tracking pull_theta2.36_phi1.57_raw_tracking.parquet \
  --output pull_unified.parquet \
  --camera-frames 5
```

The compiler retains every robot-rate hold row and attaches a robust median
camera estimate from a few valid Branch, Spur, and Apple frames. Source hashes,
source metadata, synchronization information, topology, units, and compiler
details are stored in the Parquet footer.

### 5. Inspect and visualize the result

```bash
# Inspect headers, footer metadata, and the first 15 rows.
python -m real_robot_exps.dump_parquet_preview pull_unified.parquet

# Create the time-series diagnostic figure.
python -m real_robot_exps.viz_static_sysid \
  --input pull_unified.parquet \
  --save pull_unified_viz.png
```

Check the episode ID, baseline source/hash, raw versus corrected wrench,
force/torque magnitudes, joint-torque signals, camera validity, hold boundaries,
absolute positions, position deltas, and woody bending angles before accepting
the episode.

## Usage
Run the script as a module from the root of your workspace:

python -m real_robot_exps.apple_pullto_static [OPTIONS]

### Examples

```bash
# Standard Data Collection (Defaults: 5cm, 5 stops, up-back pull):
python -m real_robot_exps.apple_pullto_static --mode collect --plot

# 1. Unloaded dry run; saves pull_theta2.36_phi1.57_baseline_robot.parquet.
python -m real_robot_exps.apple_pullto_static --mode baseline --theta 2.36 --phi 1.57 --distance 0.04 --stops 4

# 2. Repeat with the interaction/load present. The matching baseline is
# automatically loaded and subtracted across each hold.
python -m real_robot_exps.apple_pullto_static --mode collect --theta 2.36 --phi 1.57 --distance 0.04 --stops 4

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
* `--mode` (str, default: collect): `baseline` saves an unloaded dynamic-bias profile; `collect` loads the matching default-named baseline and applies it when `USE_DYNAMIC_BASELINE_CORRECTION` is enabled near the top of the script.
* `--plot`: Legacy compatibility flag. Use `viz_static_sysid.py` for the current Parquet visualization workflow.
* `--debug` (str, default: none): Set to "all" to print verbose step-by-step wrench and trajectory data.
* `--kp` (int, default: 80): The proportional gain for the controller (recommended 20-120). Derivative gains are automatically calculated.
* `--override` (str): Append config overrides in key=value format (e.g., --override robot.gripper_force_n=60).
* `--direction-index` / `--num-directions`: Select the index and width of the direction one-hot vector stored in each row.
* `--robot-output`: Override the raw robot Parquet filename. Dynamic baseline auto-discovery uses the default baseline filename, so custom baseline naming currently requires placing/copying it at the expected default path before `collect` mode.
* `--tracking`: Existing raw camera Parquet to compile after the robot run.
* `--camera-frames`: Number of camera frames per hold used by compilation.
* `--max-camera-delta`: Maximum camera/robot timestamp difference in seconds.
* `--unified-output`: Output filename when compiling with `--tracking`.

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
Baseline mode saves an unloaded `*_baseline_robot.parquet`. Corrected collect
runs retain three separate wrench fields:

- `ft_wrist_raw`: the original loaded-run wrench.
- `ft_wrist_baseline`: unloaded bias interpolated over normalized time within
  the corresponding hold.
- `ft_wrist`: corrected interaction wrench (`raw - baseline`), used by the
  unified system-ID schema.

Baseline compatibility is checked using pull angles, distance, hold count,
selected starting pose, and exact starting transform. A mismatch raises an
error instead of combining unrelated runs. Set
`USE_DYNAMIC_BASELINE_CORRECTION = False` for an explicitly uncorrected collect
run. The baseline source path, SHA-256, interpolation method, and field
semantics are stored in Parquet metadata.

The older CSV-only `compute_interaction.py` utility remains for legacy files,
but new Parquet runs perform the matched dynamic subtraction directly in
`apple_pullto_static.py` and preserve all three wrench representations.

The original legacy example is retained for existing CSV pairs named
`pull_theta2.36_phi1.57_raw.csv` and
`pull_theta2.36_phi1.57_baseline.csv`:

```bash
python -m real_robot_exps.compute_interaction --theta 2.36 --phi 1.57 --plot
```

## Unified static system-ID Parquet

The robot collector and AprilTag detector are separate programs. This keeps
camera/OpenCV work out of the robot control process. Both programs timestamp
samples with Unix wall-clock seconds from `time.time()` on the same host; run
the standalone detector during the pull so the two files overlap in time.

```bash
# Terminal 1: writes the raw robot Parquet.
python -m real_robot_exps.apple_pullto_static --config real_robot_exps/config.yaml --mode collect --theta 2.36 --phi 2.36 --distance 0.04 --stops 4

# Terminal 2: run this during the pull.
python at-tracking/Detecting.py --output pull_theta2.36_phi2.36_raw_tracking.parquet

# Optional manual recompilation:
python -m real_robot_exps.compile_static_sysid \
  --robot pull_theta2.36_phi2.36_raw_robot.parquet \
  --tracking pull_theta2.36_phi2.36_raw_tracking.parquet \
  --output pull_unified.parquet \
  --camera-frames 5
```

For unloaded pull trials closer to the robot, edit the one-line switch near the
top of `apple_pullto_static.py`:

```python
USE_CLOSE_PULL_START_POSE = True
```

`True` starts at `close_pose_4x4`, currently position `[0.0, 0.70, 0.35]` m.
Its orientation is the normal apple orientation rolled forward/downward by
20° about the local EE X axis to improve joint clearance. Change
`CLOSE_PULL_ROLL_FORWARD_DEG` beside the toggle to adjust it. `False` restores
the normal `apple_pose_4x4`. The chosen name, exact transform, toggle value,
roll angle, and both candidate transforms are saved in the robot Parquet
metadata.

### Hard-coded positions currently in use

Unless noted otherwise, these are Cartesian `[x, y, z]` positions in metres in
the Franka base frame. This table covers the active collection and diagnostic
scripts; files under `real_robot_exps/old tests/` are intentionally excluded.

| Name | Position `[x, y, z]` m | Used by | Source / notes |
| --- | --- | --- | --- |
| Home pose | `[0.0, 0.85, 0.42]` | `apple_pullto_static`, both F/T sweeps | Hard-coded in `apple_pullto_static`; fallback for `robot.ft_home_pos` in sweep scripts. |
| Apple pose | `[0.0, 0.9262, 0.41]` | `apple_pullto_static`, both F/T sweeps, torque calibration | Normal extended pull/diagnostic position; fallback for `robot.ft_apple_pos` in sweep scripts. |
| Close pull pose | `[0.0, 0.70, 0.35]` | `apple_pullto_static` only | Selected by `USE_CLOSE_PULL_START_POSE`; additionally rolled +20° about local EE X using `CLOSE_PULL_ROLL_FORWARD_DEG`. |
| Home up | `[0.0, 0.85, 0.44]` | `ft_rest_pose_sweep` | Derived from home using the default `--offset-cm 2`; changes with that argument. |
| Home forward | `[0.0, 0.87, 0.42]` | `ft_rest_pose_sweep` | Derived from home using the default +2 cm base-Y offset. |
| Apple up | `[0.0, 0.9262, 0.43]` | `ft_rest_pose_sweep` | Derived from apple using the default +2 cm base-Z offset. |
| Apple forward | `[0.0, 0.9462, 0.41]` | `ft_rest_pose_sweep` | Derived from apple using the default +2 cm base-Y offset. |
| Fixed asset | `[0.0676, 0.7072, 0.1146]` | Evaluation/controller task logic | `task.fixed_asset_position` in `config.yaml`; not used as the static-pull start pose. |
| Target peg base | `[0.0676, 0.7072, 0.1146]` | Evaluation success/engagement logic | `task.target_peg_base_position` in `config.yaml`. |
| Fruiting-base fallback | `[0.0, 0.0, 0.0]` | Static-system-ID compiler | In the reference-AprilTag frame, not the Franka base frame; used until a calibrated tag-to-base translation is supplied. |

The evaluation configuration also contains relative offsets rather than
absolute positions: `hand_init_pos = [0.0, 0.0, 0.047]` m relative to the
fixed asset, `hand_init_pos_noise = [0.02, 0.02, 0.01]` m, and
`ee_to_peg_base_offset = [0.0, 0.0, -0.0324]` m. Camera-derived Branch, Spur,
and Apple locations are measured per frame and therefore are not hard-coded
positions.

The compiler keeps all robot-rate static-hold rows and attaches a median camera
estimate made from a few complete `Branch`, `Spur`, and `Apple` frames. The
three schema slots are `Branch -> Spur`, `Branch -> Apple`, and `Spur -> Apple`;
there is no synthetic `fruiting_base` point in the compiled geometry.

The unified file stores the model fields as fixed-size Arrow lists and embeds
collection, calibration, synchronization, topology, units, source-file hashes,
software versions, and camera-selection diagnostics in `dataset_metadata` in
the Parquet footer. Camera-derived positions and transforms are converted into
the Franka base `O` frame before writing, while the original tag-frame inputs
remain recorded in metadata for traceability.

### Per-step Parquet fields

The principal collected/compiled fields are:

| Field | Dim | Units / meaning |
| --- | ---: | --- |
| `ft_wrist` | 6 | Interaction wrench `[Fx,Fy,Fz,Tx,Ty,Tz]` after dynamic subtraction when enabled; otherwise uncorrected. Robot EE/body convention, N and N·m. |
| `ft_wrist_raw` | 6 | Loaded-run wrench before dynamic baseline subtraction. |
| `ft_wrist_baseline` | 6 | Interpolated unloaded wrench subtracted from the raw wrench. |
| `tau_J` | 7 | Measured link-side joint torque, joints 1–7, base to EE; N·m. |
| `tau_ext_hat_filtered` | 7 | Franka low-pass-filtered external joint-torque estimate; N·m. |
| `tau_J_d` | 7 | Desired link-side torque without gravity; N·m. |
| `gravity_torques` | 7 | `Model.gravity(state)`; N·m. |
| `tcp_velocity` | 6 | TCP linear XYZ in m/s plus angular XYZ in rad/s. |
| `action` | 6 | Recorded EE velocity command; zero during the recorded static holds. |
| `tcp_pos` | 3 | Robot TCP position in metres. |
| `apple_pos` | 3 | Camera-derived Apple position in the Franka base `O` frame; metres. |
| `apple_pose_4x4` | 16 | Camera-derived Apple pose in the Franka base `O` frame, flattened row-major. |
| `woody_part_start_pos` | 9 | Three XYZ starts flattened in Branch, Spur, Apple order in the Franka base `O` frame; metres. |
| `woody_part_end_pos` | 9 | Matching three XYZ ends in the Franka base `O` frame; metres. |
| `woody_bending_angles` | 3 | Chord deflection from the frame-0 rest direction; radians. |
| `hold_number` | number of holds | One-hot hold encoding. |
| `direction` | number of directions | One-hot direction encoding. |
| `phase` | 1 | Static hold = 1, moving = 0. Recorded rows are static holds. |
| `excitation_direction` | 3 | Unit pull-direction vector. |

The unified file additionally contains episode/step identifiers and camera
selection/timestamp diagnostics. Baseline-only raw files contain the robot
fields but do not contain camera geometry.

Three distinct robot-side joint-torque fields are recorded directly from the
same `pylibfranka.RobotState` sample. Each is a 7-vector in Franka joint order
`[joint_1, ..., joint_7]` (base to end effector), in N·m:

reference: https://frankarobotics.github.io/libfranka/0.15.0/structfranka_1_1RobotState.html#acdef8005828d193e45b128085a9e363b

- `tau_J`: measured link-side joint torque sensor signals. This is the sensor
  measurement; this collection code does not subtract gravity from it.
- `tau_ext_hat_filtered`: libfranka's low-pass-filtered estimate of torque due
  to external forces. It is the difference between `tau_J` and model-expected
  torque and does not include configured end-effector/load or robot
  mass/dynamics contributions.
- `tau_J_d`: desired link-side joint torque signal without gravity. This is a
  desired/controller signal, not a second measurement.

The process-based robot interface transports all three robot-state signals,
plus the model gravity vector, through shared memory. The raw robot Parquet and
compiled unified Parquet retain the names separately. The compiler does not
apply torque or gravity corrections. Wrench baseline subtraction occurs only
in `apple_pullto_static.py` collect mode when
`USE_DYNAMIC_BASELINE_CORRECTION` is enabled, and the raw and baseline wrench
fields remain available alongside the corrected `ft_wrist`.

The collector also records and plots `gravity_torques`, the 7-vector returned
by `pylibfranka`'s `Model.gravity(state)`.

To inspect a unified file over time, run:

```bash
python -m real_robot_exps.viz_static_sysid \
  --input pull_unified.parquet \
  --save pull_unified_viz.png
```

The viewer gives `tau_J`, `tau_ext_hat_filtered`, `tau_J_d`, and model gravity
separate plots, then lays out wrench, TCP velocity, action, pose, bending
angles, and experiment state on a shared timestamp axis, with hold boundaries
marked for debugging.
It also derives and plots total force `||F||` and total torque `||T||` from the
existing six-component `ft_wrist` field, so older Parquet files work without
being regenerated.


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

To test orientation dependence while keeping the TCP at the configured apple
position, run the unloaded local-axis orientation sweep:

```bash
python -m real_robot_exps.ft_rest_orientation_sweep \
  --config real_robot_exps/config.yaml \
  --output ft_rest_orientation_sweep.parquet \
  --hold-seconds 3 \
  --angle-deg 45
```

It records exactly seven stops: the apple reference orientation and
+/-45-degree rotations about each local EE axis. It moves directly between the
ordered targets, avoiding the extra reference-pose transitions used by the
previous version. Use a smaller `--angle-deg` if any target is uncomfortable
or close to a joint limit.
