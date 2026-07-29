# Reference Notes

This file keeps the denser pipeline details that do not belong in the main
README.

## Collection modes

- `baseline`: records an unloaded dynamic baseline for the same trajectory.
- `collect`: records the loaded trial and subtracts the matching baseline when
  dynamic correction is enabled.
- `collect --only-metadata`: captures under-gravity and lengthened pre-grasp geometry plus a
  post-grasp reconstruction snapshot without running the pull trajectory or
  applying a baseline.

The runner captures an under-gravity snapshot and then a lengthened snapshot
before checking for missing baselines. The lengthened snapshot is the canonical
pre-grasp geometry and remains available through the backward-compatible
`pre_grasp_geometry.snapshot` field. The natural snapshot is retained as both
`settled_snapshot` and `under_gravity_snapshot`. The dynamic pre-grasp target
uses the lengthened apple center in the Franka base frame minus one apple
radius along base Y. The camera/base calibration helper
[`real_robot_exps.remake_translation_matrix`](/home/skand/connor/Continuous_Force_RL/real_robot_exps/remake_translation_matrix.py)
recomputes the reference-tag-to-base translation from live detections and the
current TCP reading. It assumes the TCP sits `4 cm` negative `Y` of the apple
center in base frame unless you override `--apple-to-tcp-distance-m`, and it
prints the solved translation plus the full `4x4` matrix to the terminal.
`apple_pullto_static.py --manual-setup` is the manual alternative: it keeps
torque mode off while you move the arm onto the apple surface, then records the
current TCP pose as a separate pull-start record without overwriting the
lengthened camera snapshot.
For camera/base calibration, `real_robot_exps.calibrate_camera_to_base` reads
the eye-on-base `.calib` file as the camera pose in the Franka base frame and
prints the ready-to-paste `CAMERA_TO_BASE_4X4_DEFAULT` candidate.
The live tracker writer now applies that calibration once and emits base-frame
poses directly, so `compile_static_sysid.py` only aggregates and timestamps the
tracking data instead of re-transforming it. The compiler now also validates
that tracking input is already in Franka base frame and refuses to guess if the
metadata says otherwise.

## File naming

Runner runs are saved as:

- `s00-d00_robot.parquet`
- `s00-d00_tracking.parquet`
- `s00-d00.parquet`
- `manifest.json`

Baseline files are structure-specific:

- `s00_pull_theta1.57_phi1.57_kp100_baseline_robot.parquet`

## Metadata layout

The raw robot Parquet stores a metadata row first, followed by the actual robot
samples. The footer also stores a JSON blob under `dataset_metadata`.

The top-level collection metadata is grouped into:

- `dump`
- `pre_grasp_geometry`
- `post_grasp_geometry`

The raw robot files also keep compatibility fields such as:

- `theta_rad`
- `phi_rad`
- `distance_m`
- `n_holds`
- `pull_start_pose_name`
- `robot_start_pose_4x4`

## Field conventions

Robot-side fields:

| Field | Dim | Notes |
| --- | ---: | --- |
| `ft_wrist` | 6 | Force in EE frame; torque in base frame convention used by the robot interface. |
| `ft_wrist_raw` | 6 | Raw loaded wrench before dynamic baseline subtraction. |
| `ft_wrist_baseline` | 6 | Interpolated unloaded baseline. |
| `tau_J_d` | 7 | Desired link-side joint torques without gravity. |
| `joint_pos` | 7 | Joint positions, base-to-EE order. |
| `tcp_velocity` | 6 | TCP linear + angular velocity. |
| `action` | 6 | Recorded command. |
| `tcp_pos` | 3 | TCP position. |
| `tcp_pose_4x4` | 16 | TCP pose in the Franka base frame. |
| `target_pose_4x4` | 16 | Commanded Cartesian target. |
| `hold_number` | N | One-hot hold encoding. |
| `direction` | N | One-hot direction encoding. |
| `phase` | 1 | `0` moving, `1` hold. |
| `excitation_direction` | 3 | Unit pull-direction vector. |

Unified camera fields:

| Field | Dim | Notes |
| --- | ---: | --- |
| `apple_pos` | 3 | Apple position in the base frame. |
| `apple_pose_4x4` | 16 | Apple pose in the base frame. |
| `woody_part_start_pos` | 9 | Branch / Spur / Apple start points. |
| `woody_part_end_pos` | 9 | Branch / Spur / Apple end points. |
| `woody_bending_angles` | 3 | Rest-relative chord deflection per woody segment. |
| `camera_timestamp` | 1 | Median camera timestamp used for the row. |
| `robot_camera_timestamp_offset_s` | 1 | Robot time minus camera time. |

## Hard-coded poses

Current active poses:

- Home pose: `[0.0, 0.85, 0.42]`
- Apple pose: `[0.0, 0.9262, 0.41]`
- Close pull pose: `[0.0, 0.70, 0.35]`

The close pull pose is selected with the one-line flag near the top of
`apple_pullto_static.py`:

```python
USE_CLOSE_PULL_START_POSE = False
```

## AprilTag / camera notes

- `Detecting.py` is the standalone tracking process and the sole RealSense
  owner during runner-based collection.
- After grasp closure, `apple_pullto_static.py` requests a one-shot snapshot
  from that existing detector so the apple’s moved post-grasp position is
  captured without opening a second RealSense pipeline.
- `Replay.py` reprojects compiled unified data back onto the live feed.
- The camera pipeline stores tag-frame information and the compiler converts it
  into the Franka base frame before saving the unified Parquet.
- `remake_translation_matrix.py` is a live calibration helper, not a motion
  script; it reads camera detections and the robot TCP once, then prints the
  solved reference-tag-to-base transform for debugging calibration.

## Baseline safety checks

Baseline and collect are compared using:

- pull angles,
- distance,
- number of holds,
- pull-start pose name,
- start-pose rotation,
- start-pose translation within a small tolerance,
- structure identity when present in `pre_grasp_geometry`.

If these do not match, the collect run will refuse to subtract the baseline.

## Geometry fields

The structure catalog uses:

- `length_m`
- `radius_m`
- `density_kg_m3`

instead of mass. The runner keeps those values in `pre_grasp_geometry` and the
rest of the metadata under `dump`.
