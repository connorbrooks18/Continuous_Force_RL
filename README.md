# Apple Pull System Identification

This repository collects apple-pull data from a Franka arm, optional AprilTag
tracking, and unified Parquet files for reconstruction and analysis.

The main entry point is [`real_robot_exps.runner`](/home/skand/connor/Continuous_Force_RL/real_robot_exps/runner.py).
It now assumes the apple is present first, captures a settled pre-grasp camera
snapshot, uses that snapshot to build the dynamic apple start pose from the
apple center plus one apple radius along base-frame Y, and only then decides whether a
structure-specific baseline must be collected.

## Main workflows

Normal collection:

```bash
python -m real_robot_exps.runner
```

Metadata-only reconstruction capture:

```bash
python -m real_robot_exps.runner --only-metadata
```

Useful flags:

- `--structure-index 0` selects a structure without an interactive prompt.
- `--camera-ema-alpha 0.3` enables light camera smoothing during unified compile.
- `--no-start-detector` disables the AprilTag tracking subprocess.
- `--no-expect-tracking` skips unified compile and PNG generation.
- `--only-metadata` skips baseline generation and the pull trajectory, but still captures settled geometry and post-grasp reconstruction data.
- `--manual-setup` pauses with torque mode off so you can physically place the arm on the apple surface before the pull starts.

## Current runner behavior

For a normal `collect` run, the runner does this:

1. prompts for the structure,
2. asks for the apple in its settled starting state,
3. captures the settled snapshot,
4. uses that snapshot to define the dynamic apple pose,
5. checks for missing baseline files for that structure and direction,
6. if needed, asks you to remove the apple and runs those baselines,
7. runs the actual tracked collection,
8. compiles the unified Parquet and saves a PNG.

Baseline files are structure-specific, for example:

```text
s00_pull_theta1.57_phi1.57_kp100_baseline_robot.parquet
```

## What gets saved

Raw robot Parquet files contain:

- a metadata row first,
- robot-side samples after that,
- a `dataset_metadata` JSON blob in the Parquet footer.

Unified Parquet files contain:

- robot-side data,
- camera-derived apple and woody geometry in the Franka base frame,
- timestamp alignment fields,
- `pre_grasp_geometry`,
- `post_grasp_geometry`.

Important convention: `ft_wrist` force components are in the end-effector
frame, while torque components, TCP pose, and camera-derived geometry are in
the Franka base frame.

## Visualization

To render a unified file manually:

```bash
python -m real_robot_exps.viz_static_sysid --input s00-d00.parquet --save s00-d00.png --no-show
```

The visualizer now uses a non-interactive backend for `--no-show`, so headless
PNG generation works inside the runner.

## Camera/Base Calibration Helper

Use `real_robot_exps.remake_translation_matrix` when you want to recompute the
reference-tag-to-base translation from live detections and the current robot
TCP reading. The script:

- reads the apple center from the live camera in reference-tag coordinates,
- reads the TCP position from the robot in base coordinates,
- assumes the TCP is `4 cm` negative `Y` from the apple center in base frame
  by default,
- prints the solved translation and full `4x4` matrix to the terminal,
- does not move the arm and does not write JSON.

Example:

```bash
python -m real_robot_exps.remake_translation_matrix
```

If needed, you can change the assumed apple-to-TCP distance with:

```bash
python -m real_robot_exps.remake_translation_matrix --apple-to-tcp-distance-m 0.05
```

For the camera-to-base calibration path, use
`real_robot_exps.calibrate_camera_to_base`. It reads the eye-on-base hand-eye
solution from the `.calib` file as the camera pose in the base frame and
prints a ready-to-paste
`CAMERA_TO_BASE_4X4_DEFAULT` matrix.

## Extra references

- Dense implementation notes: [REFERENCE.md](/home/skand/connor/Continuous_Force_RL/REFERENCE.md)
- Geometry-data collection instructions: [GEOMETRY_COLLECTION.md](/home/skand/connor/Continuous_Force_RL/GEOMETRY_COLLECTION.md)
