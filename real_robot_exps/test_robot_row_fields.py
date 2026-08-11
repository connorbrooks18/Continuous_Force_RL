import unittest
from types import SimpleNamespace

import numpy as np
import torch

from real_robot_exps.apple_pullto_static import _append_robot_sample


class RobotRowFieldTest(unittest.TestCase):
    def test_sample_row_includes_replay_fields(self):
        snap = SimpleNamespace(
            force_torque=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32),
            tau_J_d=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32),
            joint_pos=torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32),
            ee_linvel=torch.tensor([0.01, 0.02, 0.03], dtype=torch.float32),
            ee_angvel=torch.tensor([0.04, 0.05, 0.06], dtype=torch.float32),
            ee_pos=torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32),
            ee_quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        )
        rows = []

        _append_robot_sample(
            rows,
            timestamp=1.23,
            hold_step_idx=4,
            hold_index=2,
            phase=0,
            phase_name="pull",
            sample_label="pull",
            amplitude_m=0.05,
            target_pose_4x4=np.eye(4, dtype=np.float64),
            task_prop_gains=np.array([100, 100, 100, 30, 30, 30], dtype=np.float64),
            task_deriv_gains=np.array([10, 10, 10, 5, 5, 5], dtype=np.float64),
            hold_one_hot=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            direction_one_hot=np.array([1.0], dtype=np.float32),
            excitation_direction=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            snap=snap,
            action=np.zeros(6, dtype=np.float32),
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        np.testing.assert_allclose(
            row["task_prop_gains"],
            np.array([100.0, 100.0, 100.0, 30.0, 30.0, 30.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            row["task_deriv_gains"],
            np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
