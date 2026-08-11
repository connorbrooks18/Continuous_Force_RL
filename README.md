# Apple Pull System Identification

This repository collects apple-pull data from a Franka arm, optional AprilTag
tracking, and unified Parquet files for reconstruction and analysis.

The main entry point is [`real_robot_exps.runner`](/home/skand/connor/Continuous_Force_RL/real_robot_exps/runner.py).
It captures both the natural under-gravity structure and a lengthened
structure before the arm approaches. The settled / under-gravity snapshot now
drives the dynamic apple pull staging pose, while the lengthened snapshot
remains available through the compatibility field `pre_grasp_geometry.snapshot`
for older readers. The staged target is the apple surface plus a 2 cm offset
along the pull direction, so the total center-to-stage distance is the apple
radius plus that 2 cm clearance.

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
- `--record` saves the detector video feed for each run as an `.mp4` next to the tracking output.
- `--only-metadata` skips baseline generation and the pull trajectory, but still captures under-gravity, settled, lengthened, and post-grasp reconstruction data.
- `--manual-setup` pauses with torque mode off so you can physically place the arm on the apple surface before the pull starts.

## Understanding Pull Angles (theta & phi)

The script calculates the pull trajectory using a spherical coordinate system. Because the script subtracts the calculated vector from the target position, the pull direction still points away from the apple for the actual pull, but the arm now starts directly on the target surface pose instead of first staging 2 cm off the apple.

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


## Current runner behavior

For a normal `collect` run, the runner does this:

1. prompts for the structure,
2. asks for the natural under-gravity state and captures it,
3. asks you to lengthen the apple/structure and captures connection angles and lengths,
4. writes that measured pre-grasp geometry to `metadata_cache.json` under `--output-dir`,
5. reuses that cache on later runs with the same structure so it can skip the camera prompts,
6. checks for missing baseline files for that structure and direction,
7. if needed, asks you to remove the apple and runs those baselines,
8. runs the tracked collection,
9. after grasp closure, requests a fresh camera snapshot from the running detector,
10. compiles the unified Parquet and saves a PNG.

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

The complete file and metadata contract is documented in
[PARQUET_DATA.md](/home/skand/connor/Continuous_Force_RL/PARQUET_DATA.md).

The pre-grasp camera snapshot is captured before the arm approaches and does
not contain a TCP position. The pull-start TCP is stored separately as
`pre_grasp_geometry.robot_snapshot`. The post-grasp block contains both a
measured TCP state and a fresh camera snapshot taken after the apple has moved
with the grasp.

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
- Parquet schema and metadata reference: [PARQUET_DATA.md](/home/skand/connor/Continuous_Force_RL/PARQUET_DATA.md)
