"""Shared pose transforms for lifting camera/reference poses into base frame."""

from __future__ import annotations

import numpy as np

from real_robot_exps.static_constants import CAMERA_TO_BASE_4X4_DEFAULT


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    tf[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return tf


def pose_dict_to_transform(pose: dict[str, np.ndarray]) -> np.ndarray:
    return make_transform(pose["rot"], pose["pos"])


def transform_pose_to_base(
    pose: dict[str, np.ndarray],
    *,
    camera_to_base: np.ndarray | None = None,
) -> np.ndarray:
    camera_to_base = np.asarray(
        CAMERA_TO_BASE_4X4_DEFAULT if camera_to_base is None else camera_to_base,
        dtype=np.float64,
    ).reshape(4, 4)
    return camera_to_base @ pose_dict_to_transform(pose)


def load_camera_to_base(path) -> np.ndarray:
    """Read a camera(optical)->base 4x4 from JSON: a bare 4x4 list or {"camera_to_base_4x4": ...}."""
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload["camera_to_base_4x4"]
    matrix = np.asarray(payload, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{path}: expected a finite 4x4 camera_to_base matrix")
    rotation_error = float(np.abs(matrix[:3, :3].T @ matrix[:3, :3] - np.eye(3)).max())
    if rotation_error > 1e-3 or np.linalg.det(matrix[:3, :3]) < 0.0:
        raise ValueError(f"{path}: camera_to_base rotation is not a proper rotation")
    return matrix


def median_pose_4x4(samples: list[np.ndarray]) -> np.ndarray | None:
    """Return a robust 4x4 pose estimate from a list of homogeneous transforms."""
    if not samples:
        return None
    stack = np.asarray(samples, dtype=np.float64).reshape(-1, 4, 4)
    pose = np.median(stack, axis=0)
    rotation = pose[:3, :3]
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    pose[:3, :3] = rotation
    pose[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return pose
