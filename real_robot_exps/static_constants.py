"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.5835590699358751, 0.0205657512121615, 0.8118102375387095, -0.4284994552031315],
    [-0.8120345348099161, 0.0242117328743360, 0.5831069423933016, 0.4574503315227733],
    [-0.0076633003085937, -0.9994952935698603, 0.0308290765362673, 0.4969614604811930],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
