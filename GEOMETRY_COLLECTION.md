# Geometry Collection Instructions

This file is for someone to collect more pre-grasp and
post-grasp geometry for simulation reconstruction.

## Setup

POWER: Arm, Laptop, Gripper

AIR: make sure air valve is connected to air source and the gripper

CONNECTION: Ethernet between Arm and Laptop; Usb from camera to laptop

ARM LOCATION: White tape at 90degrees should more or less be the front left wheel. 

FRANKA UI: Unlock joints; Set mode to "Exec"; Activate FCI (in menu where shutdown is); End Effector Selected is "Dual Mode gripper (connor)" if not already

STRUCTURES: Add whatever other structure setups you'd like to test.

DIRECTIONS: Not really relevant to the metadata collection. The post_grasp is always in same orientation.

## Goal

Use the runner in `--only-metadata` mode to capture:

- settled pre-grasp geometry with the apple present,
- dynamic apple start pose metadata,
- post-grasp robot and camera geometry after the grasp has settled,
- structure-linked metadata that can be reused for simulation setup.

This mode does **not** do baseline force collection and does **not** run the
pull trajectory.

## Standard command

```bash
python -m real_robot_exps.runner --only-metadata
```

## What to do

1. Mount the structure and choose the correct structure index.
2. Leave the apple attached and let it settle naturally.
3. When prompted, press Enter to capture the settled snapshot.
4. Let the robot move to the dynamic apple start pose and perform the grasp.
5. Wait for the grasp to settle.
6. Let the run finish and write the robot/tracking/unified files.

## What this mode saves

- `pre_grasp_geometry.settled_snapshot`
- `post_grasp_geometry`
- a small raw robot Parquet with one post-grasp sample
- a unified Parquet if tracking is enabled and overlaps correctly

The single robot row is intentional. It exists so the unified compiler can
attach camera geometry to a real robot timestamp without running the pull.

## Important notes

- Keep the apple present during the settled snapshot capture.
- If you only care about metadata and reconstruction, do not remove the apple.
- If the AprilTag snapshot fails, stop and fix visibility before collecting.
- If the run writes the unified Parquet but the PNG fails, the geometry data is
  still usable.

## Files to check after a run

- raw robot: `s00-d00_robot.parquet`
- tracking: `s00-d00_tracking.parquet`
- unified: `s00-d00.parquet` (this is the only one that matters)

NOTE: if s00-d00.parquet (or whatever other structure and direction) file already exists, a new one will be created with -01 appended to the name.

Inspect the metadata with:

```bash
python -m real_robot_exps.dump_parquet_preview s00-d00.parquet
```

or focus on geometry with:

```bash
python -m real_robot_exps.preview_geometry s00-d00.parquet
```

## Troubleshooting

### Gripper 

- To test the gripper 
```bash
python -m real_robot_exps.gripper_test # to reset the gripper
python -m real_robot_exps.gripper_test o # to close the gripper
```

If this doesn't do anything. End the ros2 program. Restart the gripper (remove and add power cable). Start the ros2 program again.

### Arm Reset Pose

Sometimes, the arm hits an acceleration or velocity discontinuity while doing reset_to_start_pose (the slow movement to the apple_pose or home_pose). If the robot has been running a while and you start to get these errors, just reboot the robot.

## Doing Full Runs

If you want to not just collect metadata, it should all be working without the tag. If it's a new structure or direction, it'll force you to remove the apple at some point (so that it can do the baseline run), and you need to put the apple back before the actual run.
