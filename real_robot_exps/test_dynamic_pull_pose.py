import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import math
import torch

from real_robot_exps.apple_pullto_static import (
    _load_dynamic_pull_start_pose,
    _pose_4x4_translated_along_direction,
    _reset_to_pose_if_needed,
    pull_test,
)


class DynamicPullPoseTest(unittest.TestCase):
    def test_uses_settled_snapshot_and_apple_radius_from_structure_metadata(self):
        fallback = np.eye(4, dtype=np.float64)
        fallback[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        fallback[:3, :3] = np.array([
            [-0.994, -0.110, 0.0],
            [0.0, 0.0, 1.0],
            [-0.110, 0.991, 0.0],
        ], dtype=np.float64)
        run_metadata = {
            "pre_grasp_geometry": {
                "parts": {
                    "apple": {
                        "radius_m": 0.035,
                    }
                },
                "settled_snapshot": {
                    "apple_pos": [0.4, 0.5, 0.6],
                },
            }
        }

        pose, name, radius_m, surface_pose = _load_dynamic_pull_start_pose(
            run_metadata,
            fallback,
            theta=math.pi / 2.0,
            phi=math.pi / 2.0,
        )
        pose_alt, _, _, surface_pose_alt = _load_dynamic_pull_start_pose(
            run_metadata,
            fallback,
            theta=0.25,
            phi=2.75,
        )

        np.testing.assert_allclose(pose[:3, 3], np.array([0.4, 0.445, 0.6], dtype=np.float64))
        np.testing.assert_allclose(surface_pose[:3, 3], np.array([0.4, 0.465, 0.6], dtype=np.float64))
        np.testing.assert_allclose(pose_alt[:3, 3], pose[:3, 3])
        np.testing.assert_allclose(surface_pose_alt[:3, 3], surface_pose[:3, 3])
        np.testing.assert_allclose(pose[:3, :3], fallback[:3, :3])
        np.testing.assert_allclose(surface_pose[:3, :3], fallback[:3, :3])
        self.assertEqual(name, "settled_snapshot_apple_surface_plus_2cm_pull_direction_offset")
        self.assertEqual(radius_m, 0.035)

    def test_legacy_lengthened_snapshot_is_still_accepted(self):
        fallback = np.eye(4, dtype=np.float64)
        fallback[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        run_metadata = {
            "pre_grasp_geometry": {
                "parts": {
                    "apple": {
                        "radius_m": 0.035,
                    }
                },
                "lengthened_snapshot": {
                    "apple_pos": [0.4, 0.5, 0.6],
                },
            }
        }

        pose, name, radius_m, surface_pose = _load_dynamic_pull_start_pose(
            run_metadata,
            fallback,
            theta=math.pi / 2.0,
            phi=math.pi / 2.0,
        )
        pose_alt, _, _, surface_pose_alt = _load_dynamic_pull_start_pose(
            run_metadata,
            fallback,
            theta=1.2,
            phi=0.3,
        )

        np.testing.assert_allclose(pose[:3, 3], np.array([0.4, 0.445, 0.6], dtype=np.float64))
        np.testing.assert_allclose(surface_pose[:3, 3], np.array([0.4, 0.465, 0.6], dtype=np.float64))
        np.testing.assert_allclose(pose_alt[:3, 3], pose[:3, 3])
        np.testing.assert_allclose(surface_pose_alt[:3, 3], surface_pose[:3, 3])
        np.testing.assert_allclose(pose[:3, :3], fallback[:3, :3])
        np.testing.assert_allclose(surface_pose[:3, :3], fallback[:3, :3])
        self.assertEqual(name, "lengthened_snapshot_apple_surface_plus_2cm_pull_direction_offset")
        self.assertEqual(radius_m, 0.035)

    def test_falls_back_when_snapshot_or_radius_missing(self):
        fallback = np.eye(4, dtype=np.float64)
        fallback[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        pose, name, radius_m, surface_pose = _load_dynamic_pull_start_pose({}, fallback)

        np.testing.assert_allclose(pose, fallback)
        np.testing.assert_allclose(surface_pose, fallback)
        self.assertEqual(name, "apple_pose_4x4")
        self.assertIsNone(radius_m)

    def test_manual_setup_does_not_command_a_reset(self):
        class FakeRobot:
            def __init__(self):
                self.reset_calls = []
                self.started_torque = False
                self.ended_control = False
                self._control_rate_hz = 15.0
                self.snapshot = SimpleNamespace(
                    ee_pos=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
                    ee_quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                    ee_linvel=torch.zeros(3, dtype=torch.float32),
                    ee_angvel=torch.zeros(3, dtype=torch.float32),
                    force_torque=torch.zeros(6, dtype=torch.float32),
                    tau_J_d=torch.zeros(7, dtype=torch.float32),
                    joint_pos=torch.zeros(7, dtype=torch.float32),
                )

            def refresh_state_snapshot(self):
                return None

            def get_state_snapshot(self):
                return self.snapshot

            def reset_to_start_pose(self, pose):
                self.reset_calls.append(np.asarray(pose, dtype=np.float64).copy())

            def start_torque_mode(self):
                self.started_torque = True

            def end_control(self):
                self.ended_control = True

            def wait_for_policy_step(self):
                return None

            def check_safety(self, snap):
                return None

            def set_control_targets(self, targets):
                return None

        fake_robot = FakeRobot()
        fake_gc = SimpleNamespace(
            send_request=lambda *args, **kwargs: None,
        )
        gains = {
            "task_prop_gains": torch.ones(6, dtype=torch.float32),
            "task_deriv_gains": torch.ones(6, dtype=torch.float32),
            "kp_null": 0.0,
            "kd_null": 0.0,
            "singularity_damping": 0.0,
            "partial_inertia_decoupling": False,
            "sep_ori": False,
            "pose_ki": torch.zeros(6, dtype=torch.float32),
            "pose_integral_clamp": 50.0,
            "pose_integral_reset_on_target": True,
        }
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        manual_args = {
            "manual_setup": True,
            "only_metadata": True,
            "direction_index": 0,
            "num_directions": 1,
            "skip_enter": True,
            "post_grasp_camera_request": None,
            "post_grasp_camera_output": None,
            "config_path": "real_robot_exps/config.yaml",
        }
        pre_grasp_geometry = {
            "snapshot": {},
            "lengthened_snapshot": {},
            "settled_snapshot": {},
            "parts": {"apple": {"radius_m": 0.035}},
        }

        with patch("real_robot_exps.apple_pullto_static.time.sleep", return_value=None), \
             patch("real_robot_exps.apple_pullto_static._capture_camera_snapshot", return_value={"camera": True}), \
             patch("real_robot_exps.apple_pullto_static.save_robot_hold_parquet", return_value=Path("/tmp/manual_setup.parquet")):
            pull_test(
                math.pi / 2.0,
                math.pi / 2.0,
                fake_robot,
                pose,
                pose,
                torch.zeros(7, dtype=torch.float32),
                gains,
                pose,
                fake_gc,
                args=manual_args,
                pre_grasp_geometry=pre_grasp_geometry,
            )

        self.assertEqual(fake_robot.reset_calls, [])
        self.assertFalse(fake_robot.started_torque)
        self.assertTrue(fake_robot.ended_control)

    def test_initial_alignment_skip_helper_does_not_reset_when_already_close(self):
        class FakeRobot:
            def __init__(self):
                self.reset_calls = []
                self.snapshot = SimpleNamespace(
                    ee_pos=torch.tensor([0.101, 0.201, 0.299], dtype=torch.float32),
                )

            def get_state_snapshot(self):
                return self.snapshot

            def reset_to_start_pose(self, pose):
                self.reset_calls.append(np.asarray(pose, dtype=np.float64).copy())

        fake_robot = FakeRobot()
        target = np.eye(4, dtype=np.float64)
        target[:3, 3] = np.array([0.10, 0.20, 0.30], dtype=np.float64)

        did_reset = _reset_to_pose_if_needed(fake_robot, target, label="Pull-start reset")

        self.assertFalse(did_reset)
        self.assertEqual(fake_robot.reset_calls, [])

    def test_pull_translation_is_applied_from_the_post_grasp_origin(self):
        origin = np.eye(4, dtype=np.float64)
        origin[:3, 3] = np.array([0.4, 0.465, 0.6], dtype=np.float64)
        direction = np.array([0.0, -1.0, 0.0], dtype=np.float64)

        first_step = _pose_4x4_translated_along_direction(origin, direction, 0.01)
        second_step = _pose_4x4_translated_along_direction(origin, direction, 0.03)

        np.testing.assert_allclose(first_step[:3, 3], np.array([0.4, 0.455, 0.6], dtype=np.float64))
        np.testing.assert_allclose(second_step[:3, 3], np.array([0.4, 0.435, 0.6], dtype=np.float64))
        np.testing.assert_allclose(first_step[:3, :3], origin[:3, :3])
        np.testing.assert_allclose(second_step[:3, :3], origin[:3, :3])

if __name__ == "__main__":
    unittest.main()
