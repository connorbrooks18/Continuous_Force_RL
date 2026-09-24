"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.6523129925650190, 0.0166855965168138, 0.7577660262902710, -0.3017811774041026],
    [-0.7579076455260610, 0.0248917765500636, 0.6518868002293992, 0.4734662505081224],
    [-0.0079850224803827, -0.9995508943165725, 0.0288833704218878, 0.4577124177134332],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT

# Single-tag clothespin mounts. The part position is the tag pose chained with a
# fixed translation in the tag frame: p_part = t_tag + R_tag @ (clip_length_m * clip_dir).
# pupil_apriltags tag frame: +x right, +y down (as printed), +z into the tag.
# clip_length_m: tag center -> part point (e.g. branch centerline), measured along the clip.
# clip_dir: direction the clip runs from the tag center, in the tag frame.
CLIP_TAGS = {
    "Branch": {"id": 2, "clip_length_m": 0.05, "clip_dir": (0.0, 1.0, 0.0)},
    "Spur": {"id": 3, "clip_length_m": 0.05, "clip_dir": (0.0, 1.0, 0.0)},
}


def clip_tag_offset(name: str) -> dict:
    """Tracker id_offsets entry for a clip-mounted tag: pure translation, identity rotation."""
    clip = CLIP_TAGS[name]
    direction = np.asarray(clip["clip_dir"], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    return {"pos": (clip["clip_length_m"] * direction).tolist(), "rot": np.eye(3)}
