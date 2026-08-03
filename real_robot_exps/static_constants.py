"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.5927967232320437, 0.0259641927149794, 0.8049334790043283, -0.3898980160317737],
    [-0.8052868445333264, 0.0318355575974711, 0.5920300628295970, 0.4875081392824459],
    [-0.0102539234886063, -0.9991558226668741, 0.0397806370483468, 0.4997449698412138],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
