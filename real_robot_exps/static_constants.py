"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.6052634416579963, 0.0182030331153183, 0.7958170743190423, -0.3922033954925920],
    [-0.7960015596976864, 0.0215500179849154, 0.6049108311839857, 0.4908737860105326],
    [-0.0061386603724213, -0.9996020439706257, 0.0275330808043636, 0.5054322252680568],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
