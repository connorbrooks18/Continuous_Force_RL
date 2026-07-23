# Apple Pull System Identification

This repository collects quasi-static apple-pull data from a Franka arm,
optionally records AprilTag camera tracking, compiles the two into a unified
Parquet episode, and provides a viewer for inspection.

The normal workflow is:

1. run a baseline pass with no apple/contact load,
2. run the matching collect pass,
3. optionally compile in camera tracking,
4. inspect the resulting Parquet file.

## Quick start

Baseline:

```bash
python -m real_robot_exps.apple_pullto_static \
  --mode baseline \
  --theta 2.36 --phi 1.57 \
  --distance 0.04 --stops 4
```

Collect:

```bash
python -m real_robot_exps.apple_pullto_static \
  --mode collect \
  --theta 2.36 --phi 1.57 \
  --distance 0.04 --stops 4
```

If camera tracking is recorded separately, compile afterward:

```bash
python -m real_robot_exps.compile_static_sysid \
  --robot pull_theta2.36_phi1.57_raw_robot.parquet \
  --tracking pull_theta2.36_phi1.57_raw_tracking.parquet \
  --output pull_unified.parquet
```

For batches, use:

```bash
python -m real_robot_exps.runner \
  --structure-index 0 \
  --structures real_robot_exps/structures.json \
  --directions real_robot_exps/directions.json \
  --expect-tracking \
  --start-detector
```

## What gets saved

Raw robot Parquet files include:

- robot timestamps and controller state,
- wrench, joint torque, joint position, TCP pose, and TCP velocity data,
- a metadata row at the top of the file,
- a `dataset_metadata` footer blob with the full run metadata.

Unified Parquet files add:

- camera-derived apple and woody geometry,
- camera/robot timestamp alignment fields,
- compiled episode metadata in the Parquet footer.

Important convention: `ft_wrist` force components are in the end-effector
frame, while torque components, joint state, TCP pose, and camera-derived
geometry are in the Franka base frame.

## Visualization

Inspect a unified file with:

```bash
python -m real_robot_exps.viz_static_sysid \
  --input pull_unified.parquet \
  --save pull_unified_viz.png
```

The viewer shows wrench, magnitudes, timestamps, TCP/fruit geometry, bending
angles, and phase shading.

## Helpful scripts

- `at-tracking/Detecting.py` — standalone AprilTag tracking.
- `at-tracking/Replay.py` — replay unified data back onto the camera feed.
- `real_robot_exps/dump_parquet_preview.py` — print schema, metadata, and the
  first rows of a Parquet file.

## Detailed reference

The implementation details, metadata layout, hard-coded poses, field ordering,
and collection notes live in [REFERENCE.md](REFERENCE.md).

