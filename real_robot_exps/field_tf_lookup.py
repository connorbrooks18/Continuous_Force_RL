"""Print one TF transform as JSON (run under ROS, e.g. while the RealSense launch is up).

The hand-eye calibration is ``fr3_link0 -> camera_link``; AprilTag poses are in
``camera_color_optical_frame``. The factory transform between the two camera
frames comes from the RealSense driver's static TF, read here once so the field
session can compose the full optical->base matrix.

    python3 -m real_robot_exps.field_tf_lookup --parent camera_link \
        --child camera_color_optical_frame --output camera_link_to_optical.json

The printed/written ``matrix_4x4`` maps child-frame coordinates into the parent
frame (T_parent_child).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def quat_xyzw_to_matrix(x: float, y: float, z: float, w: float, t) -> list[list[float]]:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), float(t[0])],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), float(t[1])],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), float(t[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", default="camera_link")
    parser.add_argument("--child", default="camera_color_optical_frame")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = Node("field_tf_lookup")
    buffer = Buffer()
    TransformListener(buffer, node)
    deadline = time.monotonic() + args.timeout
    transform = None
    last_error = None
    try:
        while time.monotonic() < deadline and transform is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            try:
                transform = buffer.lookup_transform(args.parent, args.child, Time())
            except Exception as exc:  # not published yet
                last_error = exc
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if transform is None:
        print(f"TF {args.parent} -> {args.child} not available after {args.timeout:.0f} s: {last_error}",
              file=sys.stderr)
        return 1

    t = transform.transform.translation
    q = transform.transform.rotation
    payload = {
        "parent": args.parent,
        "child": args.child,
        "translation": [t.x, t.y, t.z],
        "rotation_xyzw": [q.x, q.y, q.z, q.w],
        "matrix_4x4": quat_xyzw_to_matrix(q.x, q.y, q.z, q.w, (t.x, t.y, t.z)),
        "semantics": "T_parent_child: maps child-frame points into the parent frame",
    }
    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
