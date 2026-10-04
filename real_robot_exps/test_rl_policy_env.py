"""rl_policy_env vs the training code (fixture from tools/make_vic_harvest_parity_fixture.py)."""

import dataclasses
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


class NoAdvanceWallParityTest(unittest.TestCase):
    """[D17m] the wall vs harvest_action.clamp_target_advance / grip_axis (fixture wall_cases)."""

    def test_wall_chain_matches_training(self):
        scaler = rpe.HarvestActionScaler(BOUNDS)
        for m, cases in FIXTURE["wall_cases"].items():
            b = dataclasses.replace(BOUNDS, max_advance_m=float(m))
            if float(m) == FIXTURE["wall_bounds"]["max_advance_m"]:
                self.assertEqual(b, rpe.HarvestActionBounds(**FIXTURE["wall_bounds"]))  # = the v2c bounds
            active = 0
            for c in cases:
                start = T(c["start_pose_wxyz"])
                torch.testing.assert_close(rpe.grip_axis(start[3:7]), T(c["grip_axis"]), atol=1e-6, rtol=0)
                split = rpe.split_harvest_action(scaler.to_env(T(c["u"])), b)
                tcp = T(c["tcp_pose_wxyz"])
                target = rpe.integrate_delta_pose(T(c["target_pose_wxyz"]), rpe.tool_delta_to_world(split.delta, tcp[3:7]))
                target = rpe.leash_target_pose(target, tcp, b)
                walled = rpe.clamp_target_advance(target, start[0:3], rpe.grip_axis(start[3:7]), b.max_advance_m)
                exp = T(c["expected_target_pose_wxyz"])
                torch.testing.assert_close(walled[:3], exp[:3], atol=1e-5, rtol=1e-5)
                self.assertAlmostEqual(abs(float(torch.dot(walled[3:], exp[3:]))), 1.0, places=5)
                active += bool((T(c["leashed_target_pose_wxyz"])[:3] - exp[:3]).abs().max() > 1e-6)
            self.assertGreater(active, 10, f"max_advance_m={m}: too few cases exercise the wall")

    def test_bounds_take_max_advance_from_meta(self):
        meta = {"action_bounds": dict(FIXTURE["bounds"], max_advance_m=0.0)}
        self.assertEqual(rpe.HarvestActionBounds.from_meta(meta).max_advance_m, 0.0)
        self.assertIsNone(rpe.HarvestActionBounds.from_meta({"action_bounds": FIXTURE["bounds"]}).max_advance_m)
        with self.assertRaises(ValueError):
            dataclasses.replace(BOUNDS, max_advance_m=-0.01)


class EnvStepContractTest(unittest.TestCase):
    """FrankaVicHarvestEnv against a fake robot: what it sends and what it feeds back."""

    def _env(self, bounds=BOUNDS, start_quat=(0.7071068, 0.0, 0.0, 0.7071068)):
        gains = {k: 0.0 for k in (
            "pose_integral_clamp", "pose_integral_reset_on_target", "kp_null", "kd_null",
            "singularity_damping", "partial_inertia_decoupling", "sep_ori")}
        with patch.object(rpe, "load_gains_from_config", return_value=gains):
            env = rpe.FrankaVicHarvestEnv({"robot": {}}, bounds, 250)
        env.robot = MagicMock()
        env._snap = snap([0.4, 0.0, 0.3], list(start_quat))  # default: TCP rotated +90 deg about base z
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


    def _push(self, env, u, steps):
        """Run ``steps`` steps of ``u`` with the arm not moving (the TCP stays at the start); return the
        target sent last."""
        env.robot.get_state_snapshot.return_value = env._snap
        for _ in range(steps):
            env.step(u.numpy())
        return env.robot.set_control_targets.call_args[0][0]

    def test_wall_stops_the_target_at_the_grasp_plane(self):
        """[D17m] +z_tool at full speed: with the wall the target never passes the start plane;
        sideways motion still goes through. Without it (v2b) the target goes in."""
        u = torch.zeros(13)
        u[0], u[2] = 1.0, 1.0  # +x_tool and +z_tool (into the apple)
        on = self._push(self._env(dataclasses.replace(BOUNDS, max_advance_m=0.0)), u, 10)
        off = self._push(self._env(BOUNDS), u, 10)
        # +90 deg about base z: z_tool = +z_base, x_tool = +y_base; the delta norm-clamps to 2 mm/step
        step = 0.002 / 2 ** 0.5
        torch.testing.assert_close(on.target_pos, T([0.4, 10 * step, 0.3]), atol=1e-5, rtol=0)
        torch.testing.assert_close(off.target_pos, T([0.4, 10 * step, 0.3 + 10 * step]), atol=1e-5, rtol=0)

    def test_wall_lets_the_target_pull_back(self):
        u = torch.zeros(13)
        u[2] = -1.0  # -z_tool: out of the tree
        sent = self._push(self._env(dataclasses.replace(BOUNDS, max_advance_m=0.0)), u, 5)
        torch.testing.assert_close(sent.target_pos, T([0.4, 0.0, 0.3 - 0.01]), atol=1e-5, rtol=0)

    def test_wall_follows_a_tilted_grip_axis(self):
        """TCP rotated +90 deg about base x: the grip axis z_tool is -y_base, and the wall is on y."""
        qx = (0.7071068, 0.7071068, 0.0, 0.0)
        env = self._env(dataclasses.replace(BOUNDS, max_advance_m=0.005), start_quat=qx)
        u = torch.zeros(13)
        u[2] = 1.0
        sent = self._push(env, u, 10)
        torch.testing.assert_close(sent.target_pos, T([0.4, -0.005, 0.3]), atol=1e-5, rtol=0)

    def test_info_reports_tcp_advance(self):
        env = self._env(dataclasses.replace(BOUNDS, max_advance_m=0.0))
        env.robot.get_state_snapshot.return_value = snap([0.4, 0.01, 0.303], [0.7071068, 0.0, 0.0, 0.7071068])
        *_, info = env.step(torch.zeros(13).numpy())
        self.assertAlmostEqual(info["tcp_advance_m"], 0.003, places=6)  # along the reset grip axis (+z_base)


class RunRolloutWallTest(unittest.TestCase):
    """run_rollout tells the operator whether the checkpoint's wall is on and logs the TCP advance."""

    def _run(self, max_advance_m):
        env, policy = MagicMock(), MagicMock()
        env.action_bounds = dataclasses.replace(BOUNDS, max_advance_m=max_advance_m)
        env.step.return_value = (torch.zeros(40).numpy(), 0.0, False, False,
                                 {"vic_action": torch.zeros(19).numpy(), "env_action": torch.zeros(13).numpy(),
                                  "tcp_advance_m": 0.0004})
        policy.out_of_distribution.return_value = []
        policy.act.return_value = torch.zeros(13)
        said, log = [], rpe.new_rollout_log()
        self.assertTrue(rpe.run_rollout(policy, env, torch.zeros(40).numpy(), 2, log, say=said.append))
        return said, log

    def test_wall_on(self):
        said, log = self._run(0.0)
        self.assertIn("No-advance wall ON: target held within 0.0 mm of the grasp plane along TCP +z.", said)
        self.assertEqual(log["tcp_advance_m"], [0.0004, 0.0004])
        self.assertIn("adv=+0.4mm", said[-1])

    def test_wall_off(self):
        said, _ = self._run(None)
        self.assertIn("No-advance wall OFF (checkpoint trained without one).", said)


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


WALL_ROLL = FIXTURE["policy_rollout_wall"]
V2C = REPO / "checkpoint_cache" / "vic_harvest" / WALL_ROLL["checkpoint"]


@unittest.skipUnless(V2C.exists(), f"v2c checkpoint not cached (copy runs/vic_harvest/{WALL_ROLL['checkpoint']})")
class WallCheckpointParityTest(unittest.TestCase):
    """[D17m] the v2c checkpoint (trained with the no-advance wall) against the training actor."""

    def test_policy_matches_training_actor(self):
        policy = rpe.HarvestPolicy(V2C)
        for raw, exp in zip(WALL_ROLL["raw_obs"], WALL_ROLL["expected_action"]):
            torch.testing.assert_close(policy.act(T(raw)), T(exp), atol=1e-5, rtol=1e-5)

    def test_brings_its_wall(self):
        policy = rpe.HarvestPolicy(V2C)
        self.assertEqual(policy.action_bounds, rpe.HarvestActionBounds(**FIXTURE["wall_bounds"]))
        self.assertEqual(policy.action_bounds.max_advance_m, 0.0)
        # the wall is the only change from v2b
        self.assertEqual(dataclasses.replace(policy.action_bounds, max_advance_m=None), BOUNDS)

    def test_env_from_checkpoint_holds_the_wall(self):
        """Closed loop, as main() runs it, on a fake arm that does not move: v2c keeps asking for
        +z_tool (into the tree) and the target never passes the grasp plane."""
        policy = rpe.HarvestPolicy(V2C)
        # gripper pointing down (mock_pylibfranka's home pose): v2c pushes in from here
        env = EnvStepContractTest()._env(policy.action_bounds, start_quat=(-0.0145, 0.9996, -0.0018, -0.0254))
        env.robot.get_state_snapshot.return_value = env._snap
        axis = rpe.grip_axis(env._start_pose[3:7])
        obs, pushes = env._obs(env._snap), 0
        for _ in range(40):
            obs, _r, _te, _tr, info = env.step(policy.act(T(obs)).numpy())
            pushes += info["env_action"][2] > 0
            sent = env.robot.set_control_targets.call_args[0][0]
            self.assertLessEqual(float(torch.dot(sent.target_pos - env._start_pose[0:3], axis)), 1e-6)
        self.assertGreater(pushes, 20, "the policy never pushed in, so this did not test the wall")

    def test_v2b_has_no_wall(self):
        if not V2B.exists():
            self.skipTest("v2b checkpoint not cached")
        self.assertIsNone(rpe.HarvestPolicy(V2B).action_bounds.max_advance_m)


@unittest.skipUnless(D8B.exists(), "d8b checkpoint not cached")
class WrongContractTest(unittest.TestCase):
    def test_refuses_world_layout_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "actor_layout"):
            rpe.HarvestPolicy(D8B)

if __name__ == "__main__":
    unittest.main()
