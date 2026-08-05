"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.6682000729013156, 0.0136569916037187, 0.7438562691508438, -0.4279607209921443],
    [-0.7438128590681148, 0.0335583356499938, 0.6675449563837026, 0.4798794815820612],
    [-0.0158459224910792, -0.9993434468133258, 0.0325819282988039, 0.5049356381732873],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
