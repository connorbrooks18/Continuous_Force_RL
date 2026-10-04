"""rl_policy_env vs the training code (fixture from tools/make_vic_harvest_parity_fixture.py)."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from real_robot_exps import rl_policy_env as rpe
from real_robot_exps.pro_robot_interface import StateSnapshot

FIXTURE = json.loads((Path(__file__).parent / "testdata" / "vic_harvest_tool_parity.json").read_text())
BOUNDS = rpe.HarvestActionBounds(**FIXTURE["bounds"])


def T(x):
    return torch.as_tensor(x, dtype=torch.float32)


def snap(pos, quat, linvel=(0, 0, 0), angvel=(0, 0, 0), ft=(0,) * 6):
    z = torch.zeros
    return StateSnapshot(
        T(pos), T(quat), T(linvel), T(angvel), T(ft), z(7), z(7), z(7), z(7), z(7), z(7), z(6, 7), z(7, 7)
    )


class ToolObsParityTest(unittest.TestCase):
    def test_obs_matches_training(self):
        for c in FIXTURE["obs_cases"]:
            s = snap(c["tcp_pos"], c["tcp_quat_wxyz"], c["linvel"], c["angvel"], c["ft_tool"])
            obs = rpe.build_tool_actor_obs(
                s, T(c["start_pose_wxyz"]), T(c["target_pose_wxyz"]), T(c["last_action_env"])
            )
            torch.testing.assert_close(obs, T(c["expected_obs"]), atol=1e-5, rtol=1e-5)

    def test_obs_sign_invariant(self):
        c = FIXTURE["obs_cases"][0]

        def flip(p):
            return torch.cat([T(p)[:3], -T(p)[3:]])

        a = rpe.build_tool_actor_obs(
            snap(c["tcp_pos"], c["tcp_quat_wxyz"]), T(c["start_pose_wxyz"]), T(c["target_pose_wxyz"]), torch.zeros(13)
        )
        b = rpe.build_tool_actor_obs(
            snap(c["tcp_pos"], (-T(c["tcp_quat_wxyz"])).tolist()),
            flip(c["start_pose_wxyz"]),
            flip(c["target_pose_wxyz"]),
            torch.zeros(13),
        )
        torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


class ToolActionParityTest(unittest.TestCase):
    def test_action_chain_matches_training(self):
        scaler = rpe.HarvestActionScaler(BOUNDS)
        for c in FIXTURE["action_cases"]:
            env_action = scaler.to_env(T(c["u"]))
            torch.testing.assert_close(env_action, T(c["expected_env_action"]), atol=1e-5, rtol=1e-5)
            split = rpe.split_harvest_action(env_action, BOUNDS)
            tcp = T(c["tcp_pose_wxyz"])
            target = rpe.integrate_delta_pose(T(c["target_pose_wxyz"]), rpe.tool_delta_to_world(split.delta, tcp[3:7]))
            target = rpe.leash_target_pose(target, tcp, BOUNDS)
            # q and -q are the same pose; compare position + |dot|
            exp = T(c["expected_target_pose_wxyz"])
            torch.testing.assert_close(target[:3], exp[:3], atol=1e-5, rtol=1e-5)
            self.assertAlmostEqual(abs(float(torch.dot(target[3:], exp[3:]))), 1.0, places=5)
            vic = rpe.pack_vic_pose_action(target, split.linear_k, split.angular_k, split.zeta)
            torch.testing.assert_close(vic[7:], T(c["expected_vic_action"])[7:], atol=1e-4, rtol=1e-5)


class EnvStepContractTest(unittest.TestCase):
    """FrankaVicHarvestEnv against a fake robot: what it sends and what it feeds back."""

    def _env(self):
        gains = {k: 0.0 for k in (
            "pose_integral_clamp", "pose_integral_reset_on_target", "kp_null", "kd_null",
            "singularity_damping", "partial_inertia_decoupling", "sep_ori")}
        with patch.object(rpe, "load_gains_from_config", return_value=gains):
            env = rpe.FrankaVicHarvestEnv({"robot": {}}, BOUNDS, 250)
        env.robot = MagicMock()
        q90z = [0.7071068, 0.0, 0.0, 0.7071068]  # TCP rotated +90 deg about base z
        env._snap = snap([0.4, 0.0, 0.3], q90z)
        env._target_pose = torch.cat([env._snap.ee_pos, env._snap.ee_quat])
        env._start_pose = env._target_pose.clone()
        env._cage_origin = env._target_pose.clone()
        env._default_dof_pos = torch.zeros(7)
        env.robot.get_state_snapshot.return_value = snap([0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0])  # post-action pose
        return env

    def test_step_uses_pre_action_tcp(self):
        env = self._env()
        u = torch.zeros(13)
        u[0] = 1.0  # full-speed +x_tool
        env.step(u.numpy())
        sent = env.robot.set_control_targets.call_args[0][0]
        # +x_tool under the pre-action +90 deg z TCP is +y_base, 2 mm
        torch.testing.assert_close(sent.target_pos, T([0.4, 0.002, 0.3]), atol=1e-6, rtol=0)
        self.assertEqual(sent.gain_frame, "ee")

    def test_last_action_is_env_units(self):
        env = self._env()
        u = torch.zeros(13)
        obs, *_ = env.step(u.numpy())
        expected = rpe.HarvestActionScaler(BOUNDS).to_env(u)
        torch.testing.assert_close(T(obs[27:40]), expected, atol=1e-5, rtol=1e-5)
        self.assertGreater(float(obs[33]), 50.0)  # u=0 -> geometric-mean K, ~100 N/m, not 0


REPO = Path(__file__).resolve().parents[1]
V2B = REPO / "checkpoint_cache" / "vic_harvest" / "v2b_s0" / "ckpt_000016000"
D8B = REPO / "checkpoint_cache" / "vic_harvest" / "d8b_tanh15_s1" / "ckpt_000030400"


@unittest.skipUnless(V2B.exists(), "v2b checkpoint not cached (copy runs/vic_harvest/v2b_s0/checkpoints/ckpt_000016000)")
class CheckpointParityTest(unittest.TestCase):
    def test_policy_matches_training_actor(self):
        roll = FIXTURE["policy_rollout"]
        policy = rpe.HarvestPolicy(V2B)
        for raw, exp in zip(roll["raw_obs"], roll["expected_action"]):
            torch.testing.assert_close(policy.act(T(raw)), T(exp), atol=1e-5, rtol=1e-5)

    def test_bounds_come_from_meta(self):
        policy = rpe.HarvestPolicy(V2B)
        self.assertEqual(policy.action_bounds.linear_delta_m, 0.002)
        self.assertEqual(policy.action_bounds.k_lin_max, 500.0)
        self.assertEqual(policy.max_episode_steps, 250)


@unittest.skipUnless(D8B.exists(), "d8b checkpoint not cached")
class WrongContractTest(unittest.TestCase):
    def test_refuses_world_layout_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "actor_layout"):
            rpe.HarvestPolicy(D8B)

if __name__ == "__main__":
    unittest.main()
