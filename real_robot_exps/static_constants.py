"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.6789635757059694, 0.0194412632928537, 0.7339145046571449, -0.2978334590028123],
    [-0.7341556166936165, 0.0246484462740928, 0.6785337018700609, 0.5057429919963149],
    [-0.0048982998866885, -0.9995071242256611, 0.0310083098599224, 0.4591895670535989],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
