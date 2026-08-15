"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.5492430284391260, 0.0195895436407416, 0.8354330287287934, -0.3173358143348692],
    [-0.8356123872750627, 0.0238412669576100, 0.5488019061759232, 0.5176653352572655],
    [-0.0091670029719725, -0.9995238085057322, 0.0294639149928925, 0.4592512074712277],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
