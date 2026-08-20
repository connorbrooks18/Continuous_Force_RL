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
