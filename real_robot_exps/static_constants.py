"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.7027310506750747, 0.0377260447738239, 0.7104546544029599, -0.3525209637573953],
    [-0.7113784249794097, 0.0225513042272376, 0.7024472757093341, 0.5174558586349199],
    [0.0104788783235628, -0.9990336251715326, 0.0426849960256055, 0.4961313727987144],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
