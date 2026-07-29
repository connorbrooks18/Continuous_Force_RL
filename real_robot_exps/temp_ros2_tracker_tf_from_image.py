"""Temporary ROS2 image subscriber that runs the AprilTag tracker and publishes TF.

This is a drop-in diagnostic node for the existing tracker math:
- subscribes to a ROS2 Image topic
- detects AprilTags in the callback
- updates Branch / Spur / Apple poses with the same offsets as Detecting.py
- publishes TF for detected raw tags, the reference tag, and the tracked objects

The transform tree is:
- camera frame -> raw tag frames
- camera frame -> reference_apriltag
- reference_apriltag or camera frame -> Branch / Spur / Apple

If ``--use-reference-frame`` is enabled, tracker poses are expressed in the
reference-tag frame once the reference tag has been seen at least once.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import rclpy
import cv2
from geometry_msgs.msg import TransformStamped
from pupil_apriltags import Detector
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from scipy import linalg
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformBroadcaster

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AT_TRACKING_ROOT = REPOSITORY_ROOT / "at-tracking"
if str(AT_TRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(AT_TRACKING_ROOT))

TAG_SIZE_M = 0.0170
REFERENCE_TAG_ID = 1
DEFAULT_CAMERA_FRAME = "camera_color_optical_frame"
DEFAULT_REFERENCE_FRAME = "reference_apriltag"


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    tf[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return tf


class Tracker:
    """Minimal local copy of the tracker logic used by Detecting.py."""

    def __init__(self, name: str, ids: Iterable[int], id_offsets: list[dict[str, object]]):
        self.name = name
        self.ids = tuple(ids)
        self.pose: dict[str, np.ndarray] | None = None
        self.offsets = {
            tag_id: _make_transform(offset["rot"], offset["pos"])
            for tag_id, offset in zip(self.ids, id_offsets)
        }

    def update_pose(self, tags_in_frame: dict[int, dict[str, np.ndarray]]):
        positions = []
        first_rot = None

        for tag_id in self.ids:
            tag = tags_in_frame.get(tag_id)
            if tag is None:
                continue

            t_frame_tag = _make_transform(tag["rot"], tag["pos"])
            t_frame_obj = t_frame_tag @ self.offsets[tag_id]
            positions.append(t_frame_obj[:3, 3])

            if first_rot is None:
                first_rot = t_frame_obj[:3, :3]
                if linalg.det(first_rot) < 0:
                    first_rot = -1 * first_rot

        if not positions:
            self.pose = None
            return self.pose

        self.pose = {
            "pos": np.mean(positions, axis=0),
            "rot": first_rot,
        }
        return self.pose


def _build_default_trackers() -> list[Tracker]:
    apple_offsets = [
        {
            "pos": [0, 0.0, 0.11],
            "rot": [[-0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, -0.7071]],
        },
        {
            "pos": [0.085, 0.00, 0.0],
            "rot": [[0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, 0.7071]],
        },
    ]
    spur_offsets = [
        {"pos": [0.0, 0.035, 0.03], "rot": np.eye(3)},
        {"pos": [0.0, 0.035, 0.03], "rot": [[0, 0, -1], [0, 1, 0], [1, 0, 0]]},
        {"pos": [0.0, 0.035, 0.03], "rot": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]},
    ]
    branch_offsets = [
        {
            "pos": [0, -0.015, 0.03],
            "rot": np.eye(3),
        }
    ]
    return [
        Tracker("Branch", ids=(2,), id_offsets=branch_offsets),
        Tracker("Spur", ids=(3, 4, 5), id_offsets=spur_offsets),
        Tracker("Apple", ids=(7, 0), id_offsets=apple_offsets),
    ]


def _make_detector() -> Detector:
    return Detector(
        families="tag36h11",
        quad_decimate=1.0,
        nthreads=24,
        refine_edges=1,
        quad_sigma=0.2,
        decode_sharpening=1.0,
    )


def _detect_tags(
    detector: Detector,
    frame: np.ndarray,
    camera_params,
    decision_margin: float,
    allowed_ids,
    tag_size_m: float,
):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raw_tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size_m,
    )
    return {
        tag.tag_id: tag
        for tag in raw_tags
        if tag.decision_margin > decision_margin and tag.tag_id in allowed_ids
    }


def _transform_to_output_frame(
    tags: dict[int, object],
    reference_pose,
    use_reference_frame: bool,
) -> dict[int, dict[str, np.ndarray]]:
    if not use_reference_frame:
        return {
            tag_id: {
                "pos": np.asarray(tag.pose_t, dtype=np.float64).reshape(3),
                "rot": np.asarray(tag.pose_R, dtype=np.float64),
            }
            for tag_id, tag in tags.items()
        }

    if reference_pose is None:
        return {}

    r_ref_inv = reference_pose.pose_R.T
    t_ref = np.asarray(reference_pose.pose_t, dtype=np.float64).reshape(3)
    tags_in_ref = {}
    for tag_id, tag in tags.items():
        tags_in_ref[tag_id] = {
            "pos": (r_ref_inv @ (np.asarray(tag.pose_t, dtype=np.float64).reshape(3) - t_ref)).astype(
                np.float64
            ),
            "rot": (r_ref_inv @ np.asarray(tag.pose_R, dtype=np.float64)).astype(np.float64),
        }
    return tags_in_ref


def _quat_from_rot(rot: np.ndarray) -> tuple[float, float, float, float]:
    quat = R.from_matrix(np.asarray(rot, dtype=np.float64)).as_quat()
    return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])


def _publish_tf(
    broadcaster: TransformBroadcaster,
    parent: str,
    child: str,
    transform_4x4: np.ndarray,
    stamp_msg,
) -> None:
    msg = TransformStamped()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(transform_4x4[0, 3])
    msg.transform.translation.y = float(transform_4x4[1, 3])
    msg.transform.translation.z = float(transform_4x4[2, 3])
    qx, qy, qz, qw = _quat_from_rot(transform_4x4[:3, :3])
    msg.transform.rotation.x = qx
    msg.transform.rotation.y = qy
    msg.transform.rotation.z = qz
    msg.transform.rotation.w = qw
    broadcaster.sendTransform(msg)


class Ros2TrackerTfNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("ros2_tracker_tf_from_image")
        self.args = args
        self._broadcaster = TransformBroadcaster(self)
        self._detector = _make_detector()
        self._trackers = _build_default_trackers()
        self._last_reference_pose = None
        self._last_log_time = 0.0
        self._camera_params = None
        self._warned_missing_camera_info = False

        self._subscription = self.create_subscription(
            Image,
            args.image_topic,
            self._image_callback,
            10,
        )
        self._camera_info_subscription = self.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self._camera_info_callback,
            10,
        )

        self.get_logger().info(
            f"Listening on {args.image_topic}; publishing raw tags, {DEFAULT_REFERENCE_FRAME}, "
            f"and {[tracker.name for tracker in self._trackers]}"
        )

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self._camera_params = (
            float(msg.k[0]),
            float(msg.k[4]),
            float(msg.k[2]),
            float(msg.k[5]),
        )

    def _image_callback(self, msg: Image) -> None:
        if self._camera_params is None:
            if not self._warned_missing_camera_info:
                self._warned_missing_camera_info = True
                self.get_logger().warn(
                    f"Waiting for CameraInfo on {self.args.camera_info_topic} before detecting tags"
                )
            return

        try:
            frame = self._image_msg_to_bgr8(msg)
        except Exception as exc:
            self.get_logger().warn(f"image conversion failed: {exc}")
            return

        tags = _detect_tags(
            self._detector,
            frame,
            self._camera_params,
            self.args.decision_margin,
            allowed_ids=tuple(range(0, 9)),
            tag_size_m=self.args.tag_size_m,
        )

        if self.args.use_reference_frame and REFERENCE_TAG_ID in tags:
            self._last_reference_pose = tags[REFERENCE_TAG_ID]

        tags_out = _transform_to_output_frame(
            tags,
            self._last_reference_pose,
            self.args.use_reference_frame,
        )

        for tracker in self._trackers:
            tracker.update_pose(tags_out)

        stamp = msg.header.stamp if msg.header.stamp.sec or msg.header.stamp.nanosec else self.get_clock().now().to_msg()
        self._publish_visible_transforms(tags, tags_out, stamp)

        now = time.time()
        if now - self._last_log_time >= self.args.print_every:
            self._last_log_time = now
            tracker_summary = []
            for tracker in self._trackers:
                if tracker.pose is None:
                    tracker_summary.append(f"{tracker.name}=missing")
                    continue
                pos = tracker.pose["pos"]
                tracker_summary.append(
                    f"{tracker.name}=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]"
                )
            self.get_logger().info(
                f"tags_seen={sorted(tags.keys())} "
                f"frame_mode={'reference' if self.args.use_reference_frame else 'camera'} "
                + " ".join(tracker_summary)
            )

    def _image_msg_to_bgr8(self, msg: Image) -> np.ndarray:
        encoding = (msg.encoding or "").lower()
        if encoding not in {"bgr8", "rgb8", "mono8"}:
            raise ValueError(f"Unsupported image encoding: {msg.encoding!r}")

        if encoding == "mono8":
            gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        expected_step = msg.width * 3
        if msg.step < expected_step:
            raise ValueError(
                f"Image step {msg.step} is too small for width {msg.width} and encoding {msg.encoding!r}"
            )

        frame = np.frombuffer(msg.data, dtype=np.uint8)
        frame = frame.reshape(msg.height, msg.step)
        frame = frame[:, :expected_step].reshape(msg.height, msg.width, 3)
        if encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def _publish_visible_transforms(
        self,
        tags: dict[int, object],
        tags_out: dict[int, dict[str, np.ndarray]],
        stamp_msg,
    ) -> None:
        camera_frame = self.args.camera_frame

        for tag_id, tag in tags.items():
            t_cam_tag = _make_transform(tag.pose_R, tag.pose_t.reshape(3))
            _publish_tf(self._broadcaster, camera_frame, f"tag_{tag_id}", t_cam_tag, stamp_msg)

        if self._last_reference_pose is not None:
            t_cam_ref = _make_transform(self._last_reference_pose.pose_R, self._last_reference_pose.pose_t.reshape(3))
            _publish_tf(self._broadcaster, camera_frame, self.args.reference_frame, t_cam_ref, stamp_msg)

        object_parent = (
            self.args.reference_frame
            if self.args.use_reference_frame and self._last_reference_pose is not None
            else camera_frame
        )
        for tracker in self._trackers:
            if tracker.pose is None:
                continue
            t_parent_obj = _make_transform(tracker.pose["rot"], tracker.pose["pos"])
            _publish_tf(self._broadcaster, object_parent, tracker.name, t_parent_obj, stamp_msg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--camera-frame", default=DEFAULT_CAMERA_FRAME)
    parser.add_argument("--reference-frame", default=DEFAULT_REFERENCE_FRAME)
    parser.add_argument("--decision-margin", type=float, default=3.0)
    parser.add_argument(
        "--use-reference-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Express tracker poses in the reference-tag frame once the reference has been seen.",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=1.0,
        help="Throttle console logging in seconds.",
    )
    parser.add_argument(
        "--tag-size-m",
        type=float,
        default=TAG_SIZE_M,
        help="Declared for parity with the tracker code path.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = Ros2TrackerTfNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
