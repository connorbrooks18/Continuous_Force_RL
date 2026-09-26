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

# check that everything an apple needs was saved and is usable (read-only)
python -m real_robot_exps.field_session --session 2026-10-02_orchardA --verify A003   # or all

# redo a step (pulls/baseline move the old files to A003/superseded-<time>/ first)
python -m real_robot_exps.field_session --session 2026-10-02_orchardA --apple A003 --redo baseline

# at home
python -m real_robot_exps.field_session --session 2026-10-02_orchardA --compile all
```

`--verify` reports PASS / WARN / FAIL for steps, calibration, snapshots, tracking and
video, each pull (rate, holds, metadata, **camera frames with all three tags during the
pull**), baselines (present, filtered, duration), and parts (including a plausibility
check that catches typos, e.g. an apple density of 59,000 kg/m³). It ends with the exact
`--redo` commands to fix what failed. Run it after each apple, before cutting the next one.

Everything a step runs is also written to the apple's `log.txt`: the pull series, the
baselines, the calibration and the TF lookup, not just the terminal. Known noise lines
(AprilTag "more than one new minima", micro-ROS agent UDP restarts) are dropped from the logs.

**Grasp tag check:** after closing the gripper, the session takes a snapshot with the apple
held. If a tag is hidden (in the first lab trial the gripper covered the apple tag for every
pull, so nothing could be compiled), it asks you to re-grasp or continue anyway.

Session settings (`--kp --distance --stops --hold --settle --slip-threshold
--directions --config --override`) are fixed when the session is created and
stored in `session.json`, so every apple in a session is collected the same way.

## Per-apple steps

| # | Step | You | The script |
|---|---|---|---|
| 1 | notes | describe the fruiting system, row/tree label | `apple.json` |
| 2 | calibrate | aim the camera (optional live view); hold the ChArUco board flat against the gripper (the air comes on and holds it; fingers stay in), confirm it doesn't slip; free-drive it to the image centre when asked; at the end hold the board, and Enter turns the air off | air on (`/microROS/toggle_valve`) → ChArUco launch → `handeye_auto_calibrate` → reads the `camera_link → camera_color_optical_frame` transform → `evaluate_calibration`. GOOD continues; otherwise redo or accept, with the board still held. Air off only after you confirm, also when the calibration fails. Saves `calib/` |
| 3 | tags | stick tags **Branch = 0, Spur = 1, Apple = 2** facing the camera | checks the gripper service and opens the gripper (fingers in, air off) |
| 4 | snapshots | let the apple hang → Enter; stretch the structure → Enter | **starts the detector** (runs until the pulls are done); takes both snapshots through it (median of 5 frames, plus a PNG) |
| 5 | grasp | hand-guide the open gripper around the apple → Enter | closes the gripper, asks whether the grasp is firm |
| 6 | pulls | confirm the plan | `field_pull`: for each direction: settle → post-grasp snapshot → slip check → pull + holds → return to the start pose. The apple stays held; the gripper opens at the end, or immediately on any error |
| 7 | baseline | move the apple out of the gripper's path (cut it or hold it aside) → Enter | replays every pull from the recorded start joint angles with the gripper closed on nothing |
| 8 | measurements | enter apple diameter/height/mass, stem, spur and branch dimensions (and masses if weighed) | checks ranges, repeats the values back, computes radii and densities → `parts` |

**One tool, one Desk profile.** The gripper stays mounted for the whole session,
and it holds the ChArUco board by suction during calibration. That's fine for the
eye-on-base solve: the board's offset from the end effector is one of the
unknowns it estimates, as long as the board doesn't move on the gripper while the
poses run. Desk's end-effector profile can't be switched from code, so the
session only checks that the `gripper` profile is active, before calibration,
tags, grasp, pulls and baseline. It compares the live `RobotState` values
(`F_T_NE`, `m_ee`, `F_x_Cee`) against `ee_profiles.yaml` and won't continue until
they match. `s` skips the check, and the skip is recorded in
`apple.json` (`ee_checks`). The check is **off by default**; `--ee-check` turns it on.

**The gripper is finicky: here's why, and what the session now does about it.**
The valve and finger stepper are on an ESP32 that talks to ROS over a Wi-Fi hotspot
this laptop hosts (`lfd_apples_ws/src/lfd_apples/launch/lfd_gripper.launch.py`). Two
things cause the flakiness:
1. Fast robot motion (`handeye_auto_calibrate`'s free-drive-to-pose moves in
   particular) can jostle the ESP32 enough to drop the Wi-Fi link for a moment, so
   whatever call was in flight gets no reply.
2. A launch that was never cleanly stopped (closed terminal, crashed run) leaves its
   processes running. The next launch then runs *alongside* the old one: two
   `automatic_gripper` nodes, and worse, two `micro_ros_agent` processes bound to the
   same UDP port, splitting the ESP32's traffic between them — this was observed
   directly (two agents, one from a completely different workspace) and is the most
   likely cause of "sometimes it just doesn't respond".

`field_session` now handles both: it kills any stray `lfd_automatic_gripper` /
`micro_ros_agent` process and runs `ros2 launch lfd_apples lfd_gripper.launch.py
ssid:=<--gripper-ssid> password:=<--gripper-password>` fresh, once at the start of the session. At the start and after every restart it opens
the gripper (fingers in, air off) and asks **"Is the gripper released?"**; if not, you can
retry the open, restart the controller, or continue anyway (logged in `gripper_stack.log`).
With `--no-gripper-stack` it still opens and asks at the start (`--gripper-ssid`/`--gripper-password` default to `alejos`/`harvesting`,
matching the launch file's defaults). Every gripper call (`close`/`open`/`air-on`/
`air-off`, and the calibration's board release) goes through one place
(`FieldSession._gripper_call`). **If the gripper stops responding** (timeout or
rejection), the session recovers it:
1. it asks you to hold anything the gripper is holding (board / apple), because the
   recovery ends with the gripper open;
2. it kills and relaunches the controller;
3. it tests the gripper, closing and then opening it, and asks whether it actually moved
   (if not: restart and test again, or give up);
4. it restarts what was interrupted. A failed *open* or *air off* is already done by the
   test, so the step carries on. A failed *close* or *air on* restarts its step from the
   beginning: the board hold asks you to hold the board again and turns the air back on,
   and any other step (e.g. the grasp) is re-run from its first prompt.
Every recovery is logged in the session's `gripper_stack.log`. `field_pull` (the pull series,
which runs as its own process) retries its final release a few times on its own,
without the interactive restart, since it isn't attached to a console.
`--no-gripper-stack` turns this management off if you'd rather run the launch
yourself.

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
- [ ] The gripper controller stack (Wi-Fi hotspot + micro-ROS agent + `automatic_gripper`
      node, from `lfd_apples`) starts on the field laptop. `field_session` now kills any
      stray processes and relaunches this stack itself, once at the start of the session
      and again automatically if a gripper call gets no reply (see "The gripper is finicky"
      below); this only needs manual attention if you see repeated failures.
- [ ] Capture the Desk `gripper` end-effector profile once (gripper profile selected in Desk,
      nothing else controlling the robot): `python -m real_robot_exps.ee_profiles capture --name gripper`
      (writes `real_robot_exps/ee_profiles.yaml`; commit it). `... ee_profiles show` prints
      which profile is active. If you skip this, the session captures it the first time.
- [ ] Suction holds the ChArUco board through the tilted calibration poses: run a test
      session with `--calib-poses 3` and watch the board. If it slips, lower
      `--calib-rotation-deg` (default 25°; it's passed to `handeye_auto_calibrate
      --rotation-delta-degrees`). `python -m real_robot_exps.gripper_test air-on` / `air-off`
      switch the air by hand.
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
├─ gripper_test.py               GripperClient: gripper_grab (air + fingers), /microROS/toggle_valve (air only)
├─ ee_profiles.py (+ ee_profiles.yaml)  checks the active Desk end effector (gripper)
├─ gripper_stack.py              kills stray lfd_automatic_gripper/micro_ros_agent, relaunches lfd_gripper.launch.py
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
