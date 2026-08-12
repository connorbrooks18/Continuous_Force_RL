"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.4659614764131069, 0.0219051842390659, 0.8845338124698063, -0.3213764174325678],
    [-0.8847270250856125, 0.0248072045551737, 0.4654489163005111, 0.5297761614109296],
    [-0.0117470669564700, -0.9994522327283135, 0.0309393101459916, 0.4708262334420840],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
