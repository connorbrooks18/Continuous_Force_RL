"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.5429887281948167, 0.0279589865765918, 0.8392744105017057, -0.4055594849663202],
    [-0.8397108559683919, 0.0264025206193572, 0.5423915424061988, 0.5067067528162376],
    [-0.0069942420751785, -0.9992603274295216, 0.0378137356424808, 0.4961034538432267],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
