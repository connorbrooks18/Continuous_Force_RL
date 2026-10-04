"""EE-frame (tool-frame) pose gains: K_base = R diag(k) R^T, matching the sim's
compute_vic_spatial_wrench_aniso_tool (apple_pick_sim/coupled_fruiting/vic_wrench.py)."""

import math
import unittest

import torch

from real_robot_exps.hybrid_controller import (
    ControlTargets,
    compute_pose_task_wrench,
    pack_control_targets,
    quat_from_angle_axis,
    unpack_control_targets,
)


def _rotmat_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.tolist()
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class EeFrameGainsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.kp = torch.tensor([60.0, 80.0, 500.0, 5.0, 6.0, 30.0])
        self.kd = 2.0 * torch.sqrt(self.kp)

    def _wrench(self, ee_quat, ee_frame, pos_err, linvel, angvel):
        return compute_pose_task_wrench(
            torch.zeros(3), ee_quat, linvel, angvel,
            pos_err, ee_quat,
            self.kp, self.kd, gains_in_ee_frame=ee_frame,
        )

    def test_identity_rotation_matches_base_frame(self):
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        e, v, w = torch.randn(3), torch.randn(3), torch.randn(3)
        torch.testing.assert_close(self._wrench(q, True, e, v, w), self._wrench(q, False, e, v, w))

    def test_ee_gains_quarter_turn(self):
        # TCP rotated +90 deg about base z: tool x = base y, tool y = -base x, tool z = base z.
        q = quat_from_angle_axis(torch.tensor(math.pi / 2), torch.tensor([0.0, 0.0, 1.0]))
        e = torch.tensor([0.01, 0.0, 0.0])  # base-x error lies along tool -y -> k_y = 80
        out = self._wrench(q, True, e, torch.zeros(3), torch.zeros(3))
        torch.testing.assert_close(out[:3], torch.tensor([0.8, 0.0, 0.0]), atol=1e-6, rtol=0)

    def test_random_rotation_equals_R_diag_RT(self):
        for _ in range(20):
            q = torch.randn(4)
            q = q / q.norm()
            R = _rotmat_wxyz(q)
            e, v, w = torch.randn(3) * 0.02, torch.randn(3) * 0.1, torch.randn(3) * 0.5
            out = self._wrench(q, True, e, v, w)
            k_lin = R @ torch.diag(self.kp[:3]) @ R.T
            d_lin = R @ torch.diag(self.kd[:3]) @ R.T
            d_ang = R @ torch.diag(self.kd[3:]) @ R.T
            torch.testing.assert_close(out[:3], k_lin @ e - d_lin @ v, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(out[3:], -(d_ang @ w), atol=1e-5, rtol=1e-5)  # target_quat == ee_quat

    def test_pack_roundtrip_keeps_gain_frame(self):
        t = _targets(gain_frame="ee")
        self.assertEqual(unpack_control_targets(pack_control_targets(t)).gain_frame, "ee")

    def test_unpack_without_gain_frame_defaults_to_base(self):
        d = pack_control_targets(_targets(gain_frame="ee"))
        del d["gain_frame"]
        self.assertEqual(unpack_control_targets(d).gain_frame, "base")

    def test_existing_call_sites_default_to_base(self):
        self.assertEqual(_targets().gain_frame, "base")


def _targets(**extra) -> ControlTargets:
    z6, z3 = torch.zeros(6), torch.zeros(3)
    return ControlTargets(
        target_pos=z3, target_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]), target_force=z6, sel_matrix=z6,
        task_prop_gains=z6, task_deriv_gains=z6, force_kp=z6, force_di_wrench=z6, pose_ki=z6,
        pose_integral_clamp=0.0, pose_integral_reset_on_target=True, default_dof_pos=torch.zeros(7),
        kp_null=0.0, kd_null=0.0, pos_bounds=z3, goal_position=z3, ctrl_mode="force_only",
        singularity_damping=0.0, partial_inertia_decoupling=False, sep_ori=False, **extra,
    )


if __name__ == "__main__":
    unittest.main()
