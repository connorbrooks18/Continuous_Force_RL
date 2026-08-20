"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.6478879987071975, 0.0203672320980010, 0.7614632735646866, -0.3015875270561179],
    [-0.7617026641121938, 0.0266198087258591, 0.6473796701069515, 0.4728317583124961],
    [-0.0070846746974630, -0.9994381229671326, 0.0327604295510039, 0.4566925005507069],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
