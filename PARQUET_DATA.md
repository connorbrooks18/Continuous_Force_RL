# Parquet Data Reference

This is the authoritative description of Parquet files produced by the
apple-pull collection pipeline.

## File Types

The runner normally creates:

```text
s00-d00_robot.parquet      Raw robot samples and collection metadata
s00-d00_tracking.parquet   Raw AprilTag tracking rows
s00-d00.parquet            Unified, time-aligned robot/camera episode
```

Baseline files use the raw robot format, for example:
`s00_pull_theta1.57_phi1.57_kp100_baseline_robot.parquet`.

The current baseline format is a `joint_velocity_replay_baseline`. It is
created from an actual run's raw robot Parquet: the recorded `joint_vel`
sequence is timestamp-interpolated onto the configured control-rate grid and
replayed with the robot-side jerk limiter while the unloaded wrench is
recorded. Baseline metadata identifies this with
`collection_mode="joint_velocity_replay_baseline"`,
`baseline_replay_field="joint_vel"`, and
`baseline_command_field="joint_vel_cmd"`.

Raw robot and unified files store the metadata object in the Parquet footer
under `dataset_metadata`. Raw robot files also contain one first table row with
`row_kind="metadata"` and the same metadata in `metadata_json`; remaining rows
are robot samples.

## Coordinate And Time Conventions

- Positions are meters.
- Angles are radians unless a field says otherwise.
- Timestamps are Unix wall-clock seconds from `time.time()`.
- TCP poses and camera-derived poses use the `franka_base_o` frame.
- `tcp_velocity` is ordered `[vx, vy, vz, wx, wy, wz]`.
- `action_wrench_ee` is ordered `[Fx, Fy, Fz, Tx, Ty, Tz]` and records the per-frame pose-control wrench.
- `ft_wrist` and `ft_wrist_raw` contain `[Fx, Fy, Fz, Tx, Ty, Tz]`.
- Force values use the robot end-effector/body convention; interpret torque
  axes using the robot interface.
- `target_pose_4x4`, `task_prop_gains`, and `task_deriv_gains` are stored per
  row so `compute_pose_task_wrench(...)` can be recreated later from the saved
  sample stream.

## Raw Robot Fields

| Field | Shape | Meaning |
| --- | ---: | --- |
| `row_kind` | string | `metadata` on the metadata row; null on data rows. |
| `metadata_json` | string | Metadata JSON on the metadata row. |
| `timestamp` | scalar | Robot sample time. |
| `hold_step_idx` | scalar | Step within the current hold. |
| `hold_index` | scalar | Static hold index. |
| `hold_number` | N | One-hot hold encoding. |
| `direction_index` | scalar | Direction index. |
| `direction` | D | One-hot direction encoding. |
| `phase` | scalar | `0` moving, `1` hold. |
| `phase_name` | string | Human-readable phase name. |
| `sample_label` | string | Human-readable sample label. |
| `amplitude_m` | scalar | Requested displacement amplitude. |
| `target_pose_4x4` | 16 | Commanded TCP target, reshape to `[4, 4]`. |
| `task_prop_gains` | 6 | Pose proportional gains used for the sample. |
| `task_deriv_gains` | 6 | Pose derivative gains used for the sample. |
| `ft_wrist` | 6 | Wrench used by the run. |
| `ft_wrist_raw` | 6 | Wrench before dynamic baseline subtraction. |
| `tau_J_d` | 7 | Desired joint torques. |
| `joint_pos` | 7 | Measured joint positions. |
| `tcp_velocity` | 6 | TCP linear and angular velocity. |
| `joint_vel` | 7 | Measured joint velocities. |
| `joint_vel_cmd` | 7 | Joint-velocity command sent during a replay baseline, when available. |
| `joint_vel_target` | 7 | Joint-velocity target used by the replay controller, when available. |
| `action_wrench_ee` | 6 | Per-frame pose-control wrench `[Fx, Fy, Fz, Tx, Ty, Tz]`. |
| `tcp_pos` | 3 | Measured TCP position. |
| `tcp_pose_4x4` | 16 | Measured TCP pose, reshape to `[4, 4]`. |
| `excitation_direction` | 3 | Unit pull direction. |

`ft_wrist_baseline` is added to the unified schema and may be absent from old
raw robot files. Replay-baseline files can additionally contain
`tau_J`, `tau_ext_hat_filtered`, `joint_vel_cmd`, and `joint_vel_target`; these
are diagnostic/controller fields and are not required in ordinary collected
raw files.

## Baseline subtraction

When a baseline is supplied during compilation, baseline rows are grouped by
`(hold_index, phase)` and ordered by `hold_step_idx`. For each corresponding
loaded-run segment, the compiler takes the baseline `ft_wrist` samples by
frame index, truncating an overlong baseline or repeating its final sample for
a short overrun. The segment-length difference must be no greater than 50%.

For every loaded row:

```text
ft_wrist_baseline = matched baseline ft_wrist
ft_wrist_raw      = measured loaded-run ft_wrist_raw
ft_wrist          = ft_wrist_raw - ft_wrist_baseline
```

The baseline `ft_wrist` field is the filtered (smoothed) unloaded wrench saved
by the replay collector; it is the field used for the current compiler
subtraction. `ft_wrist_raw` remains available in both baseline and collected
files for diagnostics. A baseline must contain every `(hold_index, phase)`
segment present in the loaded run and must pass the metadata compatibility
checks before collection-time subtraction is allowed.

## Unified Fields

The unified file has one row per robot sample. Robot fields above are retained,
including the replay-critical target pose and gain vectors, and these
camera/alignment fields are added:

| Field | Shape | Meaning |
| --- | ---: | --- |
| `episode_id` | string | Episode UUID. |
| `step_idx` | scalar | Sequential row index. |
| `apple_pos` | 3 | Camera-derived apple position. |
| `apple_pose_4x4` | 16 | Camera-derived apple pose. |
| `branch_pose_4x4` | 16 | Branch/Spur junction pose in the base frame. |
| `spur_pose_4x4` | 16 | Stem/Spur junction pose in the base frame. |
| `camera_timestamp` | scalar | Median selected camera timestamp. |
| `robot_camera_timestamp_offset_s` | scalar | Robot time minus camera time. |
| `camera_window_start_timestamp` | scalar | Earliest selected camera frame. |
| `camera_window_end_timestamp` | scalar | Latest selected camera frame. |
| `camera_frame_count` | scalar | Number of selected camera frames. |
| `camera_selected_timestamps` | list | Selected camera timestamps. |
| `camera_data_valid` | boolean | Whether camera geometry was available. |

Structure geometry should use the pose fields: `branch_pose_4x4` is the
Branch/Spur junction, `spur_pose_4x4` is the stem/Spur junction, and
`apple_pose_4x4` is the apple center. The metadata snapshot fields
`woody_part_start_pos`, `woody_part_end_pos`, and `woody_bending_angles` are
deprecated compatibility fields. They remain in camera/pre-grasp snapshots
for now, but are intentionally absent from per-row schemas.

## Metadata Groups

Important `dataset_metadata` groups are:

- `dump`: structure catalog, direction, controller, and runner metadata.
- `pre_grasp_geometry`: structure geometry and pre-grasp snapshots.
- `post_grasp_geometry`: robot and camera state after grasp closure.
- `dynamic_baseline`: baseline lifecycle and whether force correction was applied.
  A raw collect file may use `role="baseline_pending"` because the current
  joint-velocity replay baseline is collected after the robot run. The unified
  compiled file changes this to `role="corrected_collect_run"` with
  `applied=true` when subtraction succeeds.
- `field_layout`: dimensions and meanings of model-facing fields.
- `camera_aggregation`: frame selection and smoothing settings.
- `topology`: Branch/Spur/Apple chord ordering.
- `compiler`: software versions and source commits in unified files.

## Snapshot Semantics

| Field | Capture event | Contents |
| --- | --- | --- |
| `pre_grasp_geometry.snapshot` | Before arm approach | Compatibility alias for the lengthened camera snapshot. |
| `pre_grasp_geometry.lengthened_snapshot` | Before arm approach, after lengthening | Camera geometry used for angles, lengths, and dynamic targeting. |
| `pre_grasp_geometry.settled_snapshot` | Before arm approach, natural under gravity | Natural unloaded camera geometry. |
| `pre_grasp_geometry.under_gravity_snapshot` | Same event as settled snapshot | Explicit name for the same natural unloaded state. |
| `pre_grasp_geometry.robot_snapshot` | Arm at pull-start pose, before closure | Pull-start TCP state; not the pre-grasp camera snapshot. |
| `post_grasp_geometry.tcp_pos` | After closure and settling, before motion | Measured post-grasp TCP position. |
| `post_grasp_geometry.snapshot` | After closure and settling, before motion | Fresh camera snapshot containing the apple’s moved position. |
| `post_grasp_geometry.robot_snapshot` | Same post-grasp event | Copy of the post-grasp robot state. |

The pre-grasp camera snapshot is intentionally captured before the arm
approaches, so it has no TCP position. A pre-grasp TCP-to-apple distance is
therefore unavailable by design. Do not combine
`pre_grasp_geometry.snapshot.apple_pos` with
`pre_grasp_geometry.robot_snapshot.tcp_pos` and call that simultaneous
pre-grasp distance.

The post-grasp distance is valid because its TCP and apple position come from
the post-grasp event. During runner-based collection the detector remains the
sole RealSense owner; the robot process sends it a one-shot snapshot request.

## Compatibility Rules

- Readers that historically used `pre_grasp_geometry["snapshot"]` should keep
  using it; it now means the lengthened snapshot.
- `settled_snapshot` remains for older readers.
- `under_gravity_snapshot` is the explicit equivalent of `settled_snapshot`.
- Older files with an empty `lengthened_snapshot` and populated `snapshot`
  remain readable.
- Older files may not contain a post-grasp camera snapshot or
  `ft_wrist_baseline`.

## Inspection Commands

```bash
python -m real_robot_exps.print_apple_tcp_base --parquet s00-d00_robot.parquet
python -m real_robot_exps.dump_parquet_preview s00-d00.parquet
python -m real_robot_exps.preview_geometry s00-d00.parquet
```

Manual compilation:

```bash
python -m real_robot_exps.compile_static_sysid \
  --robot s00-d00_robot.parquet \
  --tracking s00-d00_tracking.parquet \
  --output s00-d00.parquet
```
