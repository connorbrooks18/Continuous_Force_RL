"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.6079144035439237, 0.0199488717939320, 0.7937519262830075, -0.3912264416163149],
    [-0.7939960293845110, 0.0193304161044570, 0.6076155366141158, 0.4914282674070335],
    [-0.0032223105788488, -0.9996141143098063, 0.0275905633714794, 0.5061859845584663],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
