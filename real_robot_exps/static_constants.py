"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the tag-to-base calibration changes.
# Current convention:
#   base x = tag x
#   base y = tag z
#   base z = -tag y
#
# The translation below is the reference tag origin in the Franka base O frame.
# It is hardcoded here so the unified pipeline stays explicit and easy to audit.
# REFERENCE_TAG_TO_BASE_4X4_DEFAULT = np.array([
#     [1.0, 0.0, 0.0, 0],
#     # Base-frame Y is the approach axis we care about here, so the 0.5 cm
#     # compensation for the reduced modeled stand-off is applied in this term.
#     [0.0, 0.0, 1.0, 1.00],
#     [0.0, -1.0, 0.0, 0.71],
#     [0.0, 0.0, 0.0, 1.0],
# ], dtype=np.float64)

REFERENCE_TAG_TO_BASE_4X4_DEFAULT = np.array([
    [0.9981377484729066, -0.0125972714618971, -0.0596853736283184, 0.0544669641950541],
    [0.0602877822434876, 0.0546093014329102, 0.9966861128304998, 0.9368944520478335],
    [-0.0092961489660242, -0.9984283314032737, 0.0552670667366803, 0.7125016563556590],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)
