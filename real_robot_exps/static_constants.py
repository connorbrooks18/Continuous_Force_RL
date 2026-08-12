"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.5146397449785085, 0.0210186665155144, 0.8571488485358685, -0.3556988522232474],
    [-0.8572568268445732, 0.0312933571267260, 0.5139372127294467, 0.5205003052680471],
    [-0.0160207901437216, -0.9992892181233862, 0.0341232006928138, 0.4694620128261053],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
