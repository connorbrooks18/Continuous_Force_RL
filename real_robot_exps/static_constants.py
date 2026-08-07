"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.5733989203973684, 0.0323475845966515, 0.8186374727911585, -0.3606743031883851],
    [-0.8192489245341141, 0.0308092772469492, 0.5726098043914529, 0.4996353981949647],
    [-0.0066990847755400, -0.9990017128144892, 0.0441667302037069, 0.4959166135815946],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
