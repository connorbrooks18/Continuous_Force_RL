"""Publish live camera -> apple TF for RViz troubleshooting.

This node uses the same AprilTag detector and tag-to-apple offsets as the
headless apple pose reader, but broadcasts the result as TF instead of printing
poses. It also optionally publishes the camera -> reference-tag transform so the
TF tree can be inspected in RViz.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from geometry_msgs.msg import TransformStamped
from pupil_apriltags import Detector
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AT_TRACKING_ROOT = REPOSITORY_ROOT / "at-tracking"
if str(AT_TRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(AT_TRACKING_ROOT))

TAG_SIZE_M = 0.0170
REFERENCE_TAG_ID = 1
DEFAULT_CAMERA_FRAME = "camera_color_optical_frame"
DEFAULT_APPLE_CHILD_FRAME = "apple"
DEFAULT_REFERENCE_CHILD_FRAME = "reference_tag"

# Same apple attachment hypotheses used by read_apple_pose.py.
APPLE_TAG_OFFSETS = {
    7: {
        "pos": np.array([0.0, 0.0, 0.11], dtype=np.float64),
        "rot": np.array([[-0.7071, 0.0, -0.7071], [0.0, 1.0, 0.0], [0.7071, 0.0, -0.7071]], dtype=np.float64),
    },
    0: {
        "pos": np.array([0.085, 0.0, 0.0], dtype=np.float64),
        "rot": np.array([[0.7071, 0.0, -0.7071], [0.0, 1.0, 0.0], [0.7071, 0.0, 0.7071]], dtype=np.float64),
    },
}


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    tf[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return tf


def _detect_tags(detector, frame, camera_params, decision_margin, allowed_ids):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raw_tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=TAG_SIZE_M,
    )
    return {
        tag.tag_id: tag
        for tag in raw_tags
        if tag.decision_margin > decision_margin and tag.tag_id in allowed_ids
    }


def _fuse_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("Cannot fuse an empty transform list")
    translations = np.stack([tf[:3, 3] for tf in transforms], axis=0)
    rotations = np.stack([tf[:3, :3] for tf in transforms], axis=0)
    translation = np.mean(translations, axis=0)
    rotation_mean = np.mean(rotations, axis=0)
    u, _, vt = np.linalg.svd(rotation_mean)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return _make_transform(rotation, translation)


def _tag_to_apple_transform(tag_id: int) -> np.ndarray:
    offset = APPLE_TAG_OFFSETS.get(tag_id)
    if offset is None:
        raise KeyError(f"No apple offset configured for tag {tag_id}")
    return _make_transform(offset["rot"], offset["pos"])


def _compute_camera_to_apple(tags: dict[int, object]) -> tuple[np.ndarray | None, np.ndarray | None, list[int]]:
    if REFERENCE_TAG_ID not in tags:
        return None, None, []

    ref_tag = tags[REFERENCE_TAG_ID]
    t_cam_ref = _make_transform(ref_tag.pose_R, ref_tag.pose_t.reshape(3))
    t_ref_cam = np.linalg.inv(t_cam_ref)

    apple_candidates = []
    visible_apple_ids = []
    for tag_id in APPLE_TAG_OFFSETS:
        tag = tags.get(tag_id)
        if tag is None:
            continue
        visible_apple_ids.append(tag_id)
        t_cam_tag = _make_transform(tag.pose_R, tag.pose_t.reshape(3))
        t_ref_tag = t_ref_cam @ t_cam_tag
        t_ref_apple = t_ref_tag @ _tag_to_apple_transform(tag_id)
        apple_candidates.append(t_cam_ref @ t_ref_apple)

    if not apple_candidates:
        return t_cam_ref, None, visible_apple_ids

    t_cam_apple = _fuse_transforms(apple_candidates)
    return t_cam_ref, t_cam_apple, visible_apple_ids


def _publish_tf(broadcaster: TransformBroadcaster, parent: str, child: str, transform_4x4: np.ndarray, stamp_msg) -> None:
    msg = TransformStamped()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(transform_4x4[0, 3])
    msg.transform.translation.y = float(transform_4x4[1, 3])
    msg.transform.translation.z = float(transform_4x4[2, 3])
    rot = transform_4x4[:3, :3]
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
    norm = max(np.linalg.norm([qx, qy, qz, qw]), 1e-12)
    msg.transform.rotation.x = float(qx / norm)
    msg.transform.rotation.y = float(qy / norm)
    msg.transform.rotation.z = float(qz / norm)
    msg.transform.rotation.w = float(qw / norm)
    broadcaster.sendTransform(msg)


class CameraToAppleTfPublisher(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("camera_to_apple_tf_publisher")
        self.args = args
        self._broadcaster = TransformBroadcaster(self)
        self._detector = Detector(
            families="tag36h11",
            quad_decimate=1.0,
            nthreads=24,
            refine_edges=1,
            quad_sigma=0.2,
            decode_sharpening=1.0,
        )
        self._pipeline, self._camera_params = self._init_camera(
            args.camera_fps, args.camera_width, args.camera_height, args.exposure
        )
        self._last_print = 0.0
        self._timer = self.create_timer(1.0 / float(args.publish_hz), self._tick)
        self.get_logger().info(
            f"Publishing TF from {args.camera_frame} to {args.apple_child_frame}"
        )

    def destroy_node(self):
        try:
            self._pipeline.stop()
        except Exception:
            pass
        super().destroy_node()

    def _init_camera(self, camera_fps: int, width: int, height: int, exposure: int):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, camera_fps)
        profile = pipeline.start(config)
        sensor = profile.get_device().query_sensors()[1]
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, exposure)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        camera_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
        return pipeline, camera_params

    def _tick(self):
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        frame = np.asanyarray(color_frame.get_data())
        tags = _detect_tags(
            self._detector,
            frame,
            self._camera_params,
            self.args.decision_margin,
            allowed_ids=(REFERENCE_TAG_ID, *APPLE_TAG_OFFSETS.keys()),
        )
        camera_to_ref, camera_to_apple, visible_apple_ids = _compute_camera_to_apple(tags)
        stamp = self.get_clock().now().to_msg()

        if camera_to_ref is not None:
            _publish_tf(
                self._broadcaster,
                self.args.camera_frame,
                self.args.reference_child_frame,
                camera_to_ref,
                stamp,
            )

        if camera_to_apple is not None:
            _publish_tf(
                self._broadcaster,
                self.args.camera_frame,
                self.args.apple_child_frame,
                camera_to_apple,
                stamp,
            )

        now = time.time()
        if (now - self._last_print) >= self.args.print_every:
            self._last_print = now
            if camera_to_apple is not None:
                pos = camera_to_apple[:3, 3]
                self.get_logger().info(
                    f"camera->apple m=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] "
                    f"tags_seen={sorted(tags.keys())} apple_tags={visible_apple_ids}"
                )
            else:
                self.get_logger().info(f"apple unavailable tags_seen={sorted(tags.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-frame", default=DEFAULT_CAMERA_FRAME)
    parser.add_argument("--apple-child-frame", default=DEFAULT_APPLE_CHILD_FRAME)
    parser.add_argument("--reference-child-frame", default=DEFAULT_REFERENCE_CHILD_FRAME)
    parser.add_argument("--decision-margin", type=float, default=3.0)
    parser.add_argument("--publish-hz", type=float, default=15.0)
    parser.add_argument("--print-every", type=float, default=1.0)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--exposure", type=int, default=100)
    args = parser.parse_args()

    rclpy.init()
    node = CameraToAppleTfPublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
