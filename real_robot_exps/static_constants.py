"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.5993415257205795, 0.0293530267347553, 0.7999550833443317, -0.4303178070658898],
    [-0.8004722515974436, 0.0292461272710294, 0.5986558597910537, 0.5797359188329605],
    [-0.0058232267212306, -0.9991411631301933, 0.0410247019416065, 0.4975523011863225],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
