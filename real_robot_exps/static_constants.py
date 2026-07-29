"""Shared constants for static apple-pull collection and compilation."""

from __future__ import annotations

import numpy as np


# Edit this block when the eye-on-base calibration changes.
# This matrix maps camera-frame positions into the Franka base frame.
CAMERA_TO_BASE_4X4_DEFAULT = np.array([
    [0.5778485494008985, 0.0333384863628897, 0.8154628129364998, -0.4246976658694703],
    [-0.8161314064525409, 0.0180500213105428, 0.5775843870227851, 0.5604167166919226],
    [0.0045366680587181, -0.9992811126293343, 0.0376387643952965, 0.4985162200313040],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)

# Backward-compatible alias for older code paths and metadata fields.
REFERENCE_TAG_TO_BASE_4X4_DEFAULT = CAMERA_TO_BASE_4X4_DEFAULT
