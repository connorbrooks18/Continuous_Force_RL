"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.6087499342056902, 0.0118689840957751, 0.7932733733216450, -0.3392617417927200],
    [-0.7928605604585159, 0.0446514541249340, 0.6077650691788208, 0.5036937756207193],
    [-0.0282072556973520, -0.9989321172437411, 0.0365920191429914, 0.5097597883785713],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
