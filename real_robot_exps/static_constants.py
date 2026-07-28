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
    [0.9956693734226544, -0.0347151454326286, -0.0862401154088338, -0.0107583522259106],
    [0.0864749334619379, 0.0053176936140161, 0.9962398345867229, 0.9695125872495270],
    [-0.0341260122324746, -0.9993831001233805, 0.0082966545601291, 0.7401963300677694],
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000],
], dtype=np.float64)