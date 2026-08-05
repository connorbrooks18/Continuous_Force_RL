"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.5709738072668145, 0.0248060318732209, 0.8205934268551895, -0.3841210914203919],
    [-0.8209328322985359, 0.0265395732445417, 0.5704076927131169, 0.4949963081646305],
    [-0.0076286479498409, -0.9993399380764797, 0.0355174871824435, 0.5023164313633123],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
