# Reference Notes

This file keeps the denser pipeline details that do not belong in the main
README.

## Collection modes

- `baseline`: records an unloaded dynamic baseline for the same trajectory.
- `collect`: records the loaded trial and subtracts the matching baseline when
  dynamic correction is enabled.

The runner (`real_robot_exps.runner`) will auto-run the baseline pass if the
expected baseline file is missing, then continue with the collect pass.

## File naming

Runner runs are saved as:

- `s00-d00_robot.parquet`
- `s00-d00_tracking.parquet`
- `s00-d00.parquet`
- `manifest.json`

## Metadata layout

The raw robot Parquet stores a metadata row first, followed by the actual robot
samples. The footer also stores a JSON blob under `dataset_metadata`.

The top-level collection metadata is grouped into:

- `dump`
- `robot_info`
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

- `Detecting.py` is the standalone tracking process.
- `Replay.py` reprojects compiled unified data back onto the live feed.
- The camera pipeline stores tag-frame information and the compiler converts it
  into the Franka base frame before saving the unified Parquet.

## Baseline safety checks

Baseline and collect are compared using:

- pull angles,
- distance,
- number of holds,
- pull-start pose name,
- exact `robot_start_pose_4x4`.

If these do not match, the collect run will refuse to subtract the baseline.

