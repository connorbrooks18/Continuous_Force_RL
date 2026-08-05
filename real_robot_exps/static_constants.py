"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.6688060040433161, 0.0158798324797609, 0.7432673542380470, -0.4307203044495851],
    [-0.7432065757054482, 0.0391696534004965, 0.6679144586551984, 0.4750678932311900],
    [-0.0185071549351535, -0.9991063853128452, 0.0379988952906071, 0.5042108794794403],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
