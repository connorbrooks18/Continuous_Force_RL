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
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = np.array([
    [1.0, 0.0, 0.0, 0.045],
    # Base-frame Y is the approach axis we care about here, so the 0.5 cm
    # compensation for the reduced modeled stand-off is applied in this term.
    [0.0, 0.0, 1.0, 0.975],
    [0.0, -1.0, 0.0, 0.72],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)
