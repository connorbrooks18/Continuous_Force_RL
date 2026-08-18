"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.6517123983217745, 0.0106808343663528, 0.7583909741359789, -0.3559402797903415],
    [-0.7582879051935909, 0.0308537048683718, 0.6511892979257344, 0.4913977875461222],
    [-0.0164439262585421, -0.9994668422079512, 0.0282068895142602, 0.4579462365982893],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)
# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
