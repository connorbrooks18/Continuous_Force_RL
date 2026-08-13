"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.4694122557781100, 0.0240590765388298, 0.8826512872938017, -0.3290018956623336],
    [-0.8828766618109125, 0.0280170806544251, 0.4687684324069012, 0.5381834901283391],
    [-0.0134511767115778, -0.9993178693627482, 0.0343927582111661, 0.4669521387168577],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
