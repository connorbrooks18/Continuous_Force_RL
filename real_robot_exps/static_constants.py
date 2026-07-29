"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.5814971599449898, 0.0235914229275106, 0.8132063069972874, -0.4132378742637555],
    [-0.8133742172992852, 0.0375440140263544, 0.5805280610302697, 0.5745661350561198],
    [-0.0168355459871734, -0.9990164622142343, 0.0410203926879003, 0.5009706788021850],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
