"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.6554711027024384, 0.0210196288528446, 0.7549276844340379, -0.3018227519405943],
    [-0.7551252511430339, 0.0340960911619288, 0.6546932958673456, 0.4790410056146266],
    [-0.0119786730575497, -0.9991974938771445, 0.0382214549881899, 0.4544882201553970],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
