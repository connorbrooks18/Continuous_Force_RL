# Field data collection pipeline

One command walks through every apple, and each apple gets its own folder.
Everything that needs a person happens in the field. Post-processing (baseline
subtraction, moving tag positions to the part centres, plots) runs at home.

```bash
# in the field (conda env, repo root)
python -m real_robot_exps.field_session --session 2026-10-02_orchardA

# resume a specific apple (e.g. after a crash or a skipped step)
python -m real_robot_exps.field_session --session 2026-10-02_orchardA --apple A003

# what has been collected so far
python -m real_robot_exps.field_session --session 2026-10-02_orchardA --list

# at home
python -m real_robot_exps.field_session --session 2026-10-02_orchardA --compile all
```

Session settings (`--kp --distance --stops --hold --settle --slip-threshold
--directions --config --override`) are fixed when the session is created and
stored in `session.json`, so every apple in a session is collected the same way.

## Per-apple steps

| # | Step | You | The script |
|---|---|---|---|
| 1 | notes | describe the fruiting system, row/tree label | `apple.json` |
| 2 | calibrate | aim the camera (optional live view); mount the ChArUco board; free-drive it to the image centre when asked | ChArUco launch → `handeye_auto_calibrate` → reads the `camera_link → camera_color_optical_frame` transform → `evaluate_calibration`. GOOD continues; otherwise redo or accept. Saves `calib/` |
| 3 | tool_and_tags | swap the board for the gripper; stick tags **Branch = 0, Spur = 1, Apple = 2** facing the camera | checks the gripper service and opens the gripper |
| 4 | snapshots | let the apple hang → Enter; stretch the structure → Enter | **starts the detector** (runs until the pulls are done); takes both snapshots through it (median of 5 frames, plus a PNG) |
| 5 | grasp | hand-guide the open gripper around the apple → Enter | closes the gripper, asks whether the grasp is firm |
| 6 | pulls | confirm the plan | `field_pull`: for each direction: settle → post-grasp snapshot → slip check → pull + holds → return to the start pose. The apple stays held; the gripper opens at the end, or immediately on any error |
| 7 | baseline | move the apple out of the gripper's path (cut it or hold it aside) → Enter | replays every pull from the recorded start joint angles with the gripper closed on nothing |
| 8 | measurements | enter apple diameter/height/mass, stem, spur and branch dimensions (and masses if weighed) | checks ranges, repeats the values back, computes radii and densities → `parts` |

**Desk end-effector check.** Desk's end-effector profiles (EE offset, mass, CoM,
inertia) can't be switched from code: no FCI or documented Desk API call does it.
So you switch them in Desk, and the session checks them. Before calibrating it
requires the `board` profile. After the tool swap, and before the grasp, pulls
and baseline, it requires the `gripper` profile. It compares the live `RobotState`
values (`F_T_NE`, `m_ee`, `F_x_Cee`) against `ee_profiles.yaml` and won't
continue until they match. `s` skips the check, and the skip is recorded in
`apple.json` (`ee_checks`). `--no-ee-check` turns the check off.

A step that fails offers **retry / skip / quit**. The status is stored in
`apple.json`, so quitting and re-running resumes at the first unfinished step.
If the pull series aborts, the gripper is opened, the finished directions are
kept, and resuming goes back to *grasp* and records only the missing directions.
Partial rows are kept as `dXX_robot.aborted-NN.parquet`; baseline and compile
ignore them.

## Data layout

```
~/field_data/<session>/session.json        settings, directions, git commits of both repos
~/field_data/<session>/A003/
  apple.json            notes, location, step status/errors, calibration verdict, parts, files
  parts.json            measured parts (also used by compile_static_sysid --parts-json)
  log.txt               output of every subprocess (detector, ROS launches, …)
  config/               robot_config.yaml + tracking_config.yaml used for this apple
  calib/                <name>.calib, <name>.samples, report.txt,
                        camera_link_to_optical.json, camera_to_base.json (optical → base 4x4)
  snapshots/            under_gravity.json/.png, lengthened.json/.png, post_grasp_dXX.json/.png
  tracking/             tracking_NN.parquet (one per detector start), requests/
  pulls/                dXX_robot.parquet, pull_plan_NN.json, status.json
  baseline/             dXX_baseline.parquet
  compiled/             dXX.parquet + dXX.png            (written by --compile)
```

`camera_to_base.json` maps AprilTag poses, which are in the colour optical
frame, into the robot base: `T_base_optical = T_base_camera_link (.calib) @
T_camera_link_optical (RealSense TF)`. The detector stores the matrix it used in
each tracking file (`camera_to_base_4x4_used`), and compile reads it from there.

## What each file stores

- **`dXX_robot.parquet`** (`field_pull`)
  - Rows at 1 kHz: wrist force/torque, joints, TCP pose/velocity, commanded wrench, target, hold/direction one-hots, phase.
  - Top-level metadata: `episode_id`, `rest_reference_timestamp` (end of the settle), `theta_rad`/`phi_rad`, `robot_start_pose_4x4`, `robot_start_joint_pos`, `apple_id`, `session`.
  - `dump`: gains, EE config, hold ranges, config hash.
  - `pre_grasp_geometry`: the two snapshots only (`under_gravity_snapshot`, `lengthened_snapshot`); parts are filled in at compile.
  - `post_grasp_geometry`: robot state + `camera_snapshot` after the settle, `slip_check`, `pull_origin_pose_4x4`.
  - `field_session`: notes, calibration verdict, camera matrix path.
- **`dXX_baseline.parquet`**: the unloaded replay, labelled with the pull's hold/phase by elapsed time. It carries the pull's metadata plus `baseline_start_method` and `baseline_gripper_state`.
- **`tracking_NN.parquet`**: Branch/Spur/Apple tag poses in the robot base for every frame (NaN rows where a tag was missed), plus the tracking config and camera matrix used. It is rewritten every 30 s, so a crash loses at most 30 s.
- **`compiled/dXX.parquet`**:
  - `ft_wrist = ft_wrist_raw − ft_wrist_baseline`, matched by (hold, phase).
  - Part positions: tag position + measured radius along the tag's z axis. The raw tag values are kept as `apple_pos_tag`, `*_pose_4x4_tag`.
  - The same correction is applied to the snapshots, and the spur/stem connection angles are recomputed from the corrected stretched snapshot.
  - Metadata: `tag_to_part_correction` (radii, sign), `pull_start_tracking`.

## Before going out

- [ ] `colcon build` the `easy_handeye2` packages in `~/connor/franka_ros2_ws`; `ros2 launch easy_handeye2_charuco charuco_view.launch.py` shows the board.
- [ ] Commit the `at-tracking` changes, and set `tracking_config.yaml` sizes to the **measured** printed tag size.
- [ ] The gripper service (`gripper_grab`) node starts on the field laptop.
- [ ] Franka Desk has end-effector profiles for both the board and the gripper. Capture
      each one once (with that profile selected in Desk, nothing else controlling the robot):
      `python -m real_robot_exps.ee_profiles capture --name board`, then `--name gripper`
      (writes `real_robot_exps/ee_profiles.yaml`; commit it). `... ee_profiles show` prints
      which one is active. If you skip this, the session captures them the first time it
      needs them.
- [ ] Mock rehearsal: `python -m real_robot_exps.field_session --session rehearsal --mock --skip-calibration --no-detector --stops 2 --settle 1`.
- [ ] Lab rehearsal with the real robot, camera and one apple (2 directions). Check:
  - the pull step prints `gripper TCP to apple tag … OK` when it starts;
  - the arm returns to the start pose between directions;
  - the slip check stays under 10 mm;
  - `--compile` gives an apple position ≈ tag position + radius, *further* from the camera.

  The last check confirms the sign convention (`TAG_INTO_SURFACE_SIGN` in `compile_static_sysid.py`).

## Hot path

```
field_session.py
├─ [ROS, system python] easy_handeye2_charuco eye_on_base_calib.launch.py / charuco_view.launch.py
│                       easy_handeye2_franka_auto handeye_auto_calibrate, evaluate_calibration
├─ field_tf_lookup.py            camera_link → optical transform (run with /usr/bin/python3)
├─ calibrate_camera_to_base.py   _load_handeye_calibration (.calib → 4x4)
├─ gripper_test.py               GripperClient (ROS service gripper_grab)
├─ ee_profiles.py (+ ee_profiles.yaml)  checks the active Desk end effector (board/gripper)
├─ at-tracking/Detecting.py      steps 4–6: tracking + snapshot requests
│    ├─ snapshot_requests.py, tracking_config.py (+ tracking_config.yaml), Tracker.py
│    └─ DataCollector.py, annotate.py, real_robot_exps/frame_transforms.py
├─ field_pull.py                 the pull series
│    ├─ apple_pullto_static.py   run_move, hold_and_record, save_robot_hold_parquet, …
│    └─ pro_robot_interface.py → robot_interface.py, hybrid_controller.py
├─ collect_joint_velocity_baseline.py
├─ snapshot_geometry.py          snapshot client + connection angles
├─ field_config.py               --override parsing
└─ (at home) compile_static_sysid.py → viz_static_sysid.py
```

**Lab path, not used in the field:** `runner.py`, `apple_pullto_static.py main()`
(dynamic lineup / manual setup), `camera_snapshot.py` (opens the camera itself;
never run it next to the detector), `structure_building.py`, `structures.json`,
`structure_constants.json`, `metadata_cache.py`, `recompile_static_sysid_batch.py`.

**Tools:** `dump_parquet_preview.py`, `preview_geometry.py`,
`check_timestamp_alignment.py`, `manifest_run_length.py`, `read_state.py`,
`print_apple_tcp_base.py`, `mock_pylibfranka.py` (dry runs),
`at-tracking/{Replay,read_apple_pose,apple_from_tag_assumption,DataCollector}.py`.

**Legacy, safe to remove once the lab rehearsal passes** (nothing on the hot path imports them):

| File(s) | Why |
|---|---|
| `remake_translation_matrix.py` + test, `calibrate_reference_tag_to_base.py` + test | reference-tag calibration, replaced by the ChArUco hand-eye; old tag IDs |
| `publish_camera_to_apple_tf.py`, `temp_ros2_tracker_tf_from_image.py` | temporary RViz debugging with hard-coded old tag IDs |
| `check_ft_bias_old.py`, `calibrate_torque_bias.py` | F/T bias; the baseline replay covers this now (`ft_bias` is zero) |
| `ft_rest_pose_sweep.py`, `ft_rest_orientation_sweep.py`, `compute_interaction.py` | one-off F/T experiments |
| `controller_test.py`, `getpos.py` | early bring-up scripts |
| `real_robot_exps/old tests/` | already retired |
| repo root `s0x-d0x_metadata.tmp-*.json` | leftovers from crashed `runner.py` runs |
| `static_constants.CAMERA_TO_BASE_4X4_DEFAULT` | only a fallback now (lab tools, `--skip-calibration`) |
