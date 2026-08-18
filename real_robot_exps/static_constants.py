"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.6904677218865606, 0.0152214400453732, 0.7232030370481919, -0.2997286307747337],
    [-0.7232242273677284, 0.0341234963731554, 0.6897697470486972, 0.5161159476959229],
    [-0.0141789273619551, -0.9993017035701564, 0.0345696870198161, 0.4589750190052962],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)
# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
