"""Gym-like real-robot controller for the skrl VIC-harvest policy.

Runs a checkpoint trained in ``apple_pick_sim`` (branch ``feature/rl-skrl-ppo``,
``apple_pick_gym.rl.models.LstmGaussianActor``) on the real Franka FR3 arm through
this repo's existing ``hybrid_controller``/``pro_robot_interface`` stack.

Deliberately does **not** import ``apple_pick_gym``/``apple_pick_sim``/``skrl``: the
sim repo's action/obs helper modules transitively import ``warp``/``newton`` (via
``apple_pick_sim.robot.fr3_robot.controllers.batched_action_twists``) even though the
handful of functions we need are pure-tensor math with no simulator dependency, and
this repo's Python environment has neither those GPU packages nor skrl installed.
Everything below is a from-scratch, pure-torch re-implementation of that math, each
piece commented with its exact training-side source so a future change there is easy
to notice and resync. Checkpoint loading also needs no skrl import: skrl's
``Agent.save()`` writes a plain ``{module_name: state_dict()}`` dict of tensors (see
``skrl/agents/torch/base.py::save`` / ``_get_internal_value``), so ``torch.load`` alone
is sufficient.

Usage (smoke test, no field_session needed -- use ``--mock`` before ever touching a
real arm)::

    python -m real_robot_exps.rl_policy_env --checkpoint <ckpt_dir> --steps 200 --mock
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import yaml

from real_robot_exps.apple_pullto_static import build_position_targets, load_gains_from_config
from real_robot_exps.hybrid_controller import (
    ControlTargets,
    axis_angle_from_quat,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
)
from real_robot_exps.pro_robot_interface import FrankaInterface, SafetyViolation, StateSnapshot

_ACTION_DIM = 13
_VIC_POSE_ACTION_DIM = 19
_OBS_DIM = 40
# FR3 root position in the sim's world frame, identical in every training world (identity
# rotation) -- read from harvest_worlds_v2_all2000/shard_00_snapshot.npz arr__robot_body_q[:, 0].
# The sim observes tcp_pos in world frame, so real base-frame positions need this added.
_SIM_ROBOT_BASE_POS = (0.0, 0.2, 0.0)
_N_PROPRIO = 26  # obs dims before last_action/step_frac: the ones a start pose can put out of distribution
_PROPRIO_NAMES = (
    ["tcp_pos.x", "tcp_pos.y", "tcp_pos.z", "tcp_quat.x", "tcp_quat.y", "tcp_quat.z", "tcp_quat.w"]
    + ["tcp_vel.vx", "tcp_vel.vy", "tcp_vel.vz", "tcp_vel.wx", "tcp_vel.wy", "tcp_vel.wz"]
    + ["ft.Fx", "ft.Fy", "ft.Fz", "ft.Tx", "ft.Ty", "ft.Tz"]
    + [f"joint_q.{i}" for i in range(7)]
)
# [name, start, width] rows exactly as rl/checkpoint.py records them in meta.json.
_ACTOR_LAYOUT = [
    ["tcp_pos", 0, 3],
    ["tcp_quat", 3, 4],
    ["tcp_velocity", 7, 6],
    ["ft_wrist", 13, 6],
    ["robot_joint_q", 19, 7],
    ["last_action", 26, 13],
    ["step_frac", 39, 1],
]


# ============================================================================
# Action bounds + post-processing.
# Ported from apple_pick_sim/.claude/worktrees/rl-skrl-ppo/apple_pick_gym/
#   batched_envs/harvest_action.py and rl/action_scaling.py.
# Unbatched (N=1) versions of the sim's (N, D) tensor math.
# ============================================================================


@dataclasses.dataclass(frozen=True)
class HarvestActionBounds:
    """Mirrors ``apple_pick_gym.batched_envs.harvest_action.HarvestActionBounds``.

    Always build via :meth:`from_meta` so values match whatever a given checkpoint
    was actually trained with -- do not hardcode these.
    """

    linear_delta_m: float
    angular_delta_rad: float
    k_lin_min: float
    k_lin_max: float
    k_ang_min: float
    k_ang_max: float
    zeta_min: float
    zeta_max: float
    max_target_pos_offset_m: float | None = None
    max_target_rot_offset_rad: float | None = None

    @classmethod
    def from_meta(cls, meta: dict) -> "HarvestActionBounds":
        return cls(**meta["action_bounds"])


def derive_critical_damping(stiffness: torch.Tensor, zeta: torch.Tensor | float) -> torch.Tensor:
    """``D = 2*zeta*sqrt(K)``. See ``harvest_action.derive_critical_damping``."""
    if isinstance(zeta, torch.Tensor):
        zeta_b = zeta.reshape(*([1] * stiffness.dim()))
    else:
        zeta_b = float(zeta)
    return 2.0 * zeta_b * torch.sqrt(torch.clamp(stiffness, min=0.0))


def _clip_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Norm-clamp a ``[D]`` vector to ``max_norm``.

    See ``batched_sim.../batched_action_twists.py::clip_action_tensor`` -- re-implemented
    here to avoid that module's unconditional ``import warp``.
    """
    norm = torch.linalg.norm(vec)
    if norm > max_norm and norm > 0:
        return vec * (max_norm / norm)
    return vec


@dataclasses.dataclass
class SplitHarvestAction:
    delta: torch.Tensor  # [6]: dp(3), drot(3)
    linear_k: torch.Tensor  # [3]
    angular_k: torch.Tensor  # [3]
    zeta: torch.Tensor  # scalar tensor


def split_harvest_action(action: torch.Tensor, bounds: HarvestActionBounds) -> SplitHarvestAction:
    """See ``harvest_action.split_harvest_action``. ``action`` is ``[13]``, already in
    env units (post :meth:`HarvestActionScaler.to_env`)."""
    if action.shape != (_ACTION_DIM,):
        raise ValueError(f"expected action shape ({_ACTION_DIM},), got {tuple(action.shape)}")
    dp = _clip_norm(action[0:3], bounds.linear_delta_m)
    drot = _clip_norm(action[3:6], bounds.angular_delta_rad)
    delta = torch.cat([dp, drot])
    linear_k = torch.clamp(action[6:9], min=bounds.k_lin_min, max=bounds.k_lin_max)
    angular_k = torch.clamp(action[9:12], min=bounds.k_ang_min, max=bounds.k_ang_max)
    zeta = torch.clamp(action[12], min=bounds.zeta_min, max=bounds.zeta_max)
    return SplitHarvestAction(delta=delta, linear_k=linear_k, angular_k=angular_k, zeta=zeta)


class HarvestActionScaler:
    """Maps the policy's ``[-1, 1]^13`` box to env units. See ``rl/action_scaling.py``."""

    def __init__(self, bounds: HarvestActionBounds) -> None:
        b = bounds
        self._delta_scale = torch.tensor([b.linear_delta_m] * 3 + [b.angular_delta_rad] * 3)
        self._log_k_lo = torch.tensor([math.log(b.k_lin_min)] * 3 + [math.log(b.k_ang_min)] * 3)
        self._log_k_hi = torch.tensor([math.log(b.k_lin_max)] * 3 + [math.log(b.k_ang_max)] * 3)
        self._zeta_lo = float(b.zeta_min)
        self._zeta_hi = float(b.zeta_max)

    def to_env(self, u: torch.Tensor) -> torch.Tensor:
        if u.shape != (_ACTION_DIM,):
            raise ValueError(f"expected action shape ({_ACTION_DIM},), got {tuple(u.shape)}")
        u = torch.clamp(u, -1.0, 1.0)
        delta = u[0:6] * self._delta_scale.to(u.dtype)
        lo, hi = self._log_k_lo.to(u.dtype), self._log_k_hi.to(u.dtype)
        k = torch.exp(lo + 0.5 * (u[6:12] + 1.0) * (hi - lo))
        zeta = self._zeta_lo + 0.5 * (u[12] + 1.0) * (self._zeta_hi - self._zeta_lo)
        return torch.cat([delta, k, zeta.reshape(1)])


def integrate_delta_pose(target: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """World-frame incremental rotation onto a ``[7]`` ``(pos, quat_wxyz)`` target.

    See ``harvest_action.integrate_delta_pose``. Built on this repo's existing
    ``hybrid_controller.quat_mul``/``quat_from_angle_axis`` (already wxyz, already
    field-tested) instead of re-deriving quaternion math from scratch.
    """
    pos, quat = target[0:3], target[3:7]
    dp, drot = delta[0:3], delta[3:6]
    pos_new = pos + dp
    angle = torch.linalg.norm(drot)
    if angle > 1e-12:
        delta_q = quat_from_angle_axis(angle, drot / angle)
    else:
        delta_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=target.dtype, device=target.device)
    quat_new = quat_mul(delta_q, quat)
    quat_new = quat_new / torch.linalg.norm(quat_new).clamp_min(1e-9)
    return torch.cat([pos_new, quat_new])


def _leash_or_cage(
    target: torch.Tensor,
    origin: torch.Tensor,
    *,
    max_pos_offset_m: float | None,
    max_rot_offset_rad: float | None,
) -> torch.Tensor:
    """Shared math for :func:`leash_target_pose` (checkpoint bounds) and
    :func:`cage_target_pose` (real-world backstop). See
    ``harvest_action.leash_target_pose``."""
    out = target.clone()
    if max_pos_offset_m is not None:
        offset = target[0:3] - origin[0:3]
        dist = torch.linalg.norm(offset)
        scale = min(float(max_pos_offset_m) / max(float(dist), 1e-12), 1.0)
        out[0:3] = origin[0:3] + offset * scale
    if max_rot_offset_rad is not None:
        q_origin = origin[3:7]
        rel = quat_mul(target[3:7], quat_conjugate(q_origin))
        aa = axis_angle_from_quat(rel)  # shortest-arc handled internally
        angle = torch.linalg.norm(aa)
        if angle > float(max_rot_offset_rad):
            axis = aa / angle.clamp_min(1e-12)
            angle_t = torch.as_tensor(float(max_rot_offset_rad), dtype=target.dtype, device=target.device)
            rel_new = quat_from_angle_axis(angle_t, axis)
            q_new = quat_mul(rel_new, q_origin)
            out[3:7] = q_new / torch.linalg.norm(q_new).clamp_min(1e-9)
    return out


def leash_target_pose(target: torch.Tensor, tcp: torch.Tensor, bounds: HarvestActionBounds) -> torch.Tensor:
    """Checkpoint-recorded leash (may be unbounded -- some checkpoints train with
    ``max_target_pos_offset_m=None``). See :func:`cage_target_pose` for the
    independent, always-on real-world backstop."""
    return _leash_or_cage(
        target, tcp,
        max_pos_offset_m=bounds.max_target_pos_offset_m,
        max_rot_offset_rad=bounds.max_target_rot_offset_rad,
    )


def cage_target_pose(
    target: torch.Tensor, origin: torch.Tensor, *, max_pos_offset_m: float, max_rot_offset_rad: float
) -> torch.Tensor:
    """Real-world-only hard backstop (no sim equivalent): re-leashes the target to a
    radius around the pose captured at ``reset()``, independent of whatever the
    checkpoint's own leash allows."""
    return _leash_or_cage(target, origin, max_pos_offset_m=max_pos_offset_m, max_rot_offset_rad=max_rot_offset_rad)


def pack_vic_pose_action(
    target: torch.Tensor, linear_k: torch.Tensor, angular_k: torch.Tensor, zeta: torch.Tensor
) -> torch.Tensor:
    """``[target(7), Kp(6), Kd(6)]``. See ``harvest_action.pack_vic_pose_action``."""
    kp = torch.cat([linear_k, angular_k])
    kd = derive_critical_damping(kp, zeta)
    out = torch.cat([target, kp, kd])
    assert out.shape == (_VIC_POSE_ACTION_DIM,)
    return out


# ============================================================================
# Actor network.
# Mirrors apple_pick_gym.rl.models.LstmGaussianActor / _RecurrentTower.
# Deterministic forward only (no sampling / log_std): eval always uses the clamped
# mean -- see rl/eval_vic_harvest.py::RecurrentPolicyRunner.act.
# ============================================================================

_LSTM_HIDDEN = 256
_PRE_MLP: tuple[int, ...] = (256,)
_POST_MLP: tuple[int, ...] = (256, 128)
_MEAN_BOUND = 1.5


def _mlp(in_dim: int, sizes: tuple[int, ...]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    d = in_dim
    for h in sizes:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    return nn.Sequential(*layers), d


class _RecurrentTower(nn.Module):
    """Attribute names (``pre``/``lstm``/``post``) must match the training-side
    ``_RecurrentTower`` exactly so a checkpoint's ``policy`` state_dict loads by name."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.pre, d = _mlp(in_dim, _PRE_MLP)
        self.lstm = nn.LSTM(input_size=d, hidden_size=_LSTM_HIDDEN, num_layers=1, batch_first=True)
        self.post, self.out_dim = _mlp(_LSTM_HIDDEN, _POST_MLP)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.pre(x)
        out, (h, c) = self.lstm(feats.unsqueeze(1), (h, c))
        return self.post(out.squeeze(1)), h, c


class HarvestActorNet(nn.Module):
    """Deterministic-eval mirror of ``LstmGaussianActor``. Attribute names
    (``tower.pre``/``tower.lstm``/``tower.post``/``mean_head``) must match the
    training-side module exactly for :func:`load_actor_checkpoint`'s
    ``load_state_dict`` to line up."""

    def __init__(self, obs_dim: int = _OBS_DIM, action_dim: int = _ACTION_DIM) -> None:
        super().__init__()
        self.tower = _RecurrentTower(obs_dim)
        self.mean_head = nn.Linear(self.tower.out_dim, action_dim)

    def initial_state(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, 1, _LSTM_HIDDEN, device=device)
        c = torch.zeros(1, 1, _LSTM_HIDDEN, device=device)
        return h, c

    def forward(
        self, obs: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``obs``: ``[obs_dim]``, already normalized. Returns the bounded mean action
        ``[action_dim]`` (``1.5*tanh(raw/1.5)``) plus the updated LSTM state."""
        feats, h, c = self.tower(obs.unsqueeze(0), h, c)
        raw = self.mean_head(feats).squeeze(0)
        mean = _MEAN_BOUND * torch.tanh(raw / _MEAN_BOUND)
        return mean, h, c


def load_actor_checkpoint(
    checkpoint_dir: str | Path, device: str = "cpu"
) -> tuple[HarvestActorNet, torch.Tensor, torch.Tensor, HarvestActionBounds, int]:
    """Load a skrl checkpoint's actor weights + observation normalization stats
    without importing skrl/apple_pick_gym.

    See ``rl/checkpoint.py`` (``agent.pt`` = ``{module_name: state_dict()}``) and
    ``skrl/agents/torch/ppo/ppo_rnn.py`` (registers
    ``checkpoint_modules["observation_preprocessor"]``).
    """
    checkpoint_dir = Path(checkpoint_dir)
    meta = json.loads((checkpoint_dir / "meta.json").read_text())
    if meta.get("actor_layout") != _ACTOR_LAYOUT:
        raise RuntimeError(
            f"{checkpoint_dir}: checkpoint actor_layout {meta.get('actor_layout')} does not match the "
            f"observation _build_actor_obs produces {_ACTOR_LAYOUT} -- the training-side "
            "harvest_obs.py layout changed; resync _build_actor_obs before running this policy."
        )
    modules = torch.load(checkpoint_dir / "agent.pt", map_location=device, weights_only=False)

    net = HarvestActorNet().to(device)
    missing, unexpected = net.load_state_dict(modules["policy"], strict=False)
    still_missing = [k for k in missing if k.startswith("tower.") or k.startswith("mean_head.")]
    consumed_any = any(k.startswith("tower.") or k.startswith("mean_head.") for k in modules["policy"])
    if still_missing or not consumed_any:
        raise RuntimeError(
            f"Actor checkpoint at {checkpoint_dir} did not load cleanly: "
            f"missing={still_missing}, unexpected={unexpected}. The training-side "
            "LstmGaussianActor architecture may have changed -- resync HarvestActorNet "
            "with apple_pick_gym/rl/models.py."
        )
    net.eval()

    obs_pp = modules["observation_preprocessor"]
    obs_mean = obs_pp["running_mean"].to(device=device, dtype=torch.float32)
    obs_var = obs_pp["running_variance"].to(device=device, dtype=torch.float32)
    if obs_mean.shape != (_OBS_DIM,):
        raise RuntimeError(f"expected observation_preprocessor size {_OBS_DIM}, got {tuple(obs_mean.shape)}")

    bounds = HarvestActionBounds.from_meta(meta)
    max_episode_steps = int(meta["config"]["env"]["max_episode_steps"])
    return net, obs_mean, obs_var, bounds, max_episode_steps


class HarvestPolicy:
    """Loads a checkpoint once; holds LSTM state across a rollout. Applies the exact
    training-time normalization (skrl ``RunningStandardScaler``) before the network
    forward pass -- see ``skrl/agents/torch/base.py::RunningStandardScaler._compute``
    (defaults ``epsilon=1e-8``, ``clip_threshold=5.0``, unmodified by this project's
    ``build_agent``)."""

    _EPS = 1e-8
    _CLIP = 5.0

    def __init__(self, checkpoint_dir: str | Path, device: str = "cpu") -> None:
        self.device = device
        self.net, self._obs_mean, self._obs_var, self.action_bounds, self.max_episode_steps = (
            load_actor_checkpoint(checkpoint_dir, device=device)
        )
        self._h, self._c = self.net.initial_state(device)

    def reset(self) -> None:
        self._h, self._c = self.net.initial_state(self.device)

    def _zscore(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self._obs_mean) / (torch.sqrt(self._obs_var) + self._EPS)

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self._zscore(obs), min=-self._CLIP, max=self._CLIP)

    @property
    def training_quat_mean_xyzw(self) -> torch.Tensor:
        return self._obs_mean[3:7].clone()

    def out_of_distribution(self, raw_obs, z_max: float = 3.0) -> list[tuple[str, float, float]]:
        """``(field, value, z)`` for every proprioceptive/F-T dim more than ``z_max`` training
        std devs from the training mean. Beyond 5 the normalizer clips, so the policy can't
        even tell how far off the input is."""
        obs = torch.as_tensor(raw_obs, dtype=torch.float32, device=self.device)
        z = self._zscore(obs)[:_N_PROPRIO]
        return [
            (_PROPRIO_NAMES[i], float(obs[i]), float(z[i])) for i in range(_N_PROPRIO) if abs(float(z[i])) > z_max
        ]

    @torch.no_grad()
    def act(self, raw_obs: torch.Tensor) -> torch.Tensor:
        """``raw_obs``: ``[40]`` unnormalized actor observation (see
        :func:`_build_actor_obs`). Returns ``[13]`` raw action in ``[-1, 1]`` (the
        deterministic policy mean, clamped -- matches ``RecurrentPolicyRunner.act``)."""
        norm_obs = self._normalize(raw_obs.to(device=self.device, dtype=torch.float32))
        mean, self._h, self._c = self.net(norm_obs, self._h, self._c)
        return mean.clamp(-1.0, 1.0)


# ============================================================================
# Real-robot gym env.
# ============================================================================


def _quat_wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[1], q[2], q[3], q[0]])


def _rotate(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``[3]`` vector ``v`` by unit quaternion ``q`` (``q v q*``)."""
    v_q = torch.cat([torch.zeros(1, dtype=v.dtype, device=v.device), v])
    return quat_mul(quat_mul(q_wxyz, v_q), quat_conjugate(q_wxyz))[1:]


def _build_actor_obs(
    snap: StateSnapshot,
    tcp_quat_xyzw: torch.Tensor,
    last_action: torch.Tensor,
    step_frac: float,
    sim_base_pos: torch.Tensor,
) -> torch.Tensor:
    """``[40]`` actor observation in the exact field order of
    ``apple_pick_gym.batched_envs.harvest_obs._ACTOR_FIXED_FIELDS`` + ``last_action`` +
    ``step_frac``: ``tcp_pos(3), tcp_quat_xyzw(4), tcp_velocity(6), ft_wrist(6),
    robot_joint_q(7), last_action(13), step_frac(1)``, all in the sim's conventions:

    - ``tcp_pos`` is sim-world: real base-frame position + ``sim_base_pos`` (the sim FR3
      root has identity rotation, so orientations/velocities need no change).
    - ``tcp_quat`` is ``xyzw`` (Warp order; ``StateSnapshot.ee_quat`` is ``wxyz``), with
      its sign chosen by the caller (see ``FrankaVicHarvestEnv._obs_quat_xyzw``).
    - ``ft_wrist`` is the plant's wrench on the TCP in world frame. ``snap.force_torque``
      (``-K_F_ext_hat_K``) is in the EE frame, so both halves are rotated by R(tcp) --
      the same conversion the sim uses on real logs
      (``apple_pick_sim/system_id/real_to_batched_sysid.py::world_wrench_from_ee_logged``).
    """
    q = snap.ee_quat
    ft_world = torch.cat([_rotate(q, snap.force_torque[:3]), _rotate(q, snap.force_torque[3:])])
    step_frac_t = torch.tensor([step_frac], dtype=snap.ee_pos.dtype, device=snap.ee_pos.device)
    return torch.cat(
        [
            snap.ee_pos + sim_base_pos,
            tcp_quat_xyzw,
            snap.ee_linvel,
            snap.ee_angvel,
            ft_world,
            snap.joint_pos,
            last_action,
            step_frac_t,
        ]
    )


class FrankaVicHarvestEnv(gym.Env):
    """Gym-like real-robot controller for the VIC-harvest policy.

    Builds the exact 40-D actor observation each step and turns a policy's raw
    ``[-1, 1]^13`` action into a Cartesian-impedance ``ControlTargets`` command on the
    real Franka arm, through the same post-processing pipeline
    (``HarvestActionScaler`` -> ``split_harvest_action`` -> ``integrate_delta_pose`` ->
    ``leash_target_pose`` -> ``cage_target_pose`` -> ``pack_vic_pose_action``) the sim
    training env uses. No reward and no gripper handling -- those stay session-level
    concerns (mirrors ``field_session``'s separate "grasp" step happening before a
    policy rollout).

    Call :meth:`calibrate_ft_bias` once with the gripper free, then grasp. ``reset()``
    does **not** move the robot: it assumes the arm is already positioned (e.g. by a
    prior grasp step), starts torque mode, and captures the current pose as both the
    episode's integration origin and the real-world safety cage's center.

    On ``terminated=True`` (a ``SafetyViolation`` seen after the action ran), stop calling
    ``step()``. The env does **not** open the gripper or otherwise recover the arm -- that
    is the caller's responsibility, same as the rest of this repo's field code.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict,
        action_bounds: HarvestActionBounds,
        max_episode_steps: int,
        *,
        cage_pos_m: float = 0.12,
        cage_rot_rad: float = 0.5,
        control_rate_hz: float = 60.0,  # matches sim training's RuntimeConfig.control_hz
        sim_base_pos: tuple[float, float, float] = _SIM_ROBOT_BASE_POS,
        quat_hint_xyzw=None,
        device: str = "cpu",
    ) -> None:
        """``quat_hint_xyzw``: the quaternion sign at reset is chosen to agree with this
        (pass ``HarvestPolicy.training_quat_mean_xyzw``); ``None`` picks ``w <= 0``, the sign
        of 85% of the training start poses. After reset the sign is kept continuous, as
        the sim's integration keeps it."""
        super().__init__()
        self.device = device
        self._sim_base_pos = torch.tensor(sim_base_pos, dtype=torch.float32, device=device)
        self._quat_hint = (
            torch.tensor([0.0, 0.0, 0.0, -1.0], device=device)
            if quat_hint_xyzw is None
            else torch.as_tensor(quat_hint_xyzw, dtype=torch.float32, device=device)
        )
        self._prev_obs_quat: torch.Tensor | None = None
        self.action_bounds = action_bounds
        self.max_episode_steps = int(max_episode_steps)
        self.cage_pos_m = float(cage_pos_m)
        self.cage_rot_rad = float(cage_rot_rad)
        self.action_scaler = HarvestActionScaler(action_bounds)

        self._config = dict(config)
        self._config["robot"] = dict(self._config.get("robot", {}))  # don't mutate the caller's dict
        self._config["robot"]["control_rate_hz"] = float(control_rate_hz)
        self.gains = load_gains_from_config(self._config, device)

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32)

        self.robot: FrankaInterface | None = None
        self._target_pose: torch.Tensor | None = None  # [7] pos, quat_wxyz -- integrated per step
        self._cage_origin: torch.Tensor | None = None  # [7] pose captured at reset()
        self._default_dof_pos: torch.Tensor | None = None
        self._last_action = torch.zeros(_ACTION_DIM, device=device)
        self._step_count = 0
        self._snap: StateSnapshot | None = None
        self._ft_calibrated = False

    def _connect(self) -> None:
        if self.robot is None:
            self.robot = FrankaInterface(self._config, device=self.device)

    def _obs_quat_xyzw(self, snap: StateSnapshot) -> torch.Tensor:
        q = _quat_wxyz_to_xyzw(snap.ee_quat)
        ref = self._quat_hint if self._prev_obs_quat is None else self._prev_obs_quat
        if torch.dot(q, ref) < 0:
            q = -q
        self._prev_obs_quat = q
        return q

    def _obs(self, snap: StateSnapshot, step_frac: float) -> np.ndarray:
        obs = _build_actor_obs(snap, self._obs_quat_xyzw(snap), self._last_action, step_frac, self._sim_base_pos)
        return obs.detach().cpu().numpy().astype(np.float32)

    def calibrate_ft_bias(self) -> list:
        """Zero the F/T reading. Call with the gripper free -- **before** grasping: the
        policy was trained seeing the apple/stem load at episode start, so calibrating
        with the apple held would subtract exactly the signal it needs. The bias persists
        in the comm process for every later ``reset()``."""
        self._connect()
        self.robot.end_control()
        bias = self.robot.calibrate_ft_bias()
        self._ft_calibrated = True
        return bias

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if not self._ft_calibrated:
            raise RuntimeError("call calibrate_ft_bias() with the gripper free before grasping, then reset()")
        self.robot.end_control()  # no-op unless a previous episode left torque mode on
        self.robot.start_torque_mode()
        snap = self.robot.get_state_snapshot()
        self._target_pose = torch.cat([snap.ee_pos, snap.ee_quat]).clone()
        self._cage_origin = self._target_pose.clone()
        self._default_dof_pos = snap.joint_pos.clone()
        # Hold in place with the config gains until the first action arrives.
        self.robot.set_control_targets(
            build_position_targets(self.gains, snap.ee_pos, snap.ee_quat, self._default_dof_pos, self.device)
        )
        # The comm loop restarts the F/T EMA at 0 (~20 ms time constant); let it settle.
        time.sleep(0.2)
        snap = self.robot.get_state_snapshot()
        self._snap = snap
        self._last_action = torch.zeros(_ACTION_DIM, device=self.device)
        self._step_count = 0
        self._prev_obs_quat = None
        return self._obs(snap, 0.0), {}

    def step(self, action):
        """Command ``action`` against the latest known TCP, let it run for one control
        period, then observe -- the sim's order, so the returned obs reflects this action."""
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        env_action = self.action_scaler.to_env(action_t)
        split = split_harvest_action(env_action, self.action_bounds)
        target = integrate_delta_pose(self._target_pose, split.delta)
        tcp_pose = torch.cat([self._snap.ee_pos, self._snap.ee_quat])
        target = leash_target_pose(target, tcp_pose, self.action_bounds)
        target = cage_target_pose(
            target, self._cage_origin, max_pos_offset_m=self.cage_pos_m, max_rot_offset_rad=self.cage_rot_rad
        )
        self._target_pose = target
        vic_action = pack_vic_pose_action(target, split.linear_k, split.angular_k, split.zeta)
        self.robot.set_control_targets(self._build_control_targets(vic_action))

        self.robot.wait_for_policy_step()
        snap = self.robot.get_state_snapshot()
        self._snap = snap
        terminated = False
        info: dict[str, Any] = {"vic_action": vic_action.detach().cpu().numpy()}
        try:
            self.robot.check_safety(snap)
        except SafetyViolation as exc:
            terminated = True
            info["safety_violation"] = str(exc)

        self._last_action = action_t
        self._step_count += 1
        obs = self._obs(snap, self._step_count / self.max_episode_steps)
        truncated = self._step_count >= self.max_episode_steps
        return obs, 0.0, terminated, truncated, info

    def _build_control_targets(self, vic_action: torch.Tensor) -> ControlTargets:
        target_pos = vic_action[0:3]
        target_quat = vic_action[3:7]
        kp = vic_action[7:13]
        kd = vic_action[13:19]
        pos_bounds = torch.full((3,), self.cage_pos_m, device=self.device)
        return ControlTargets(
            target_pos=target_pos,
            target_quat=target_quat,
            target_force=torch.zeros(6, device=self.device),
            sel_matrix=torch.zeros(6, device=self.device),
            task_prop_gains=kp,
            task_deriv_gains=kd,
            force_kp=torch.zeros(6, device=self.device),
            force_di_wrench=torch.zeros(6, device=self.device),
            # No integral: the sim VIC law is pure K/D. A nonzero pose_ki makes the compute
            # process add a per-step-reset sum that stiffens the policy's commanded K.
            pose_ki=torch.zeros(6, device=self.device),
            pose_integral_clamp=self.gains["pose_integral_clamp"],
            pose_integral_reset_on_target=self.gains["pose_integral_reset_on_target"],
            default_dof_pos=self._default_dof_pos,
            kp_null=self.gains["kp_null"],
            kd_null=self.gains["kd_null"],
            pos_bounds=pos_bounds,
            goal_position=self._cage_origin[0:3],
            ctrl_mode="force_only",
            singularity_damping=self.gains["singularity_damping"],
            partial_inertia_decoupling=self.gains["partial_inertia_decoupling"],
            sep_ori=self.gains["sep_ori"],
        )

    def close(self) -> None:
        if self.robot is not None:
            try:
                self.robot.end_control()
            finally:
                self.robot.shutdown()
                self.robot = None


# ============================================================================
# Smoke-test CLI.
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="checkpoint dir (ckpt_<timestep>/, agent.pt + meta.json)")
    parser.add_argument("--config", default="real_robot_exps/config.yaml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--mock", action="store_true", help="use mock_pylibfranka instead of the real arm")
    parser.add_argument("--control-rate-hz", type=float, default=60.0)
    parser.add_argument("--cage-pos-m", type=float, default=0.12)
    parser.add_argument("--cage-rot-rad", type=float, default=0.5)
    parser.add_argument(
        "--allow-ood", action="store_true", help="run even if the start pose is outside the training distribution"
    )
    parser.add_argument("--log", help="write per-step obs/action/vic_action/time to this .npz")
    args = parser.parse_args(argv)

    policy = HarvestPolicy(args.checkpoint)
    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.mock:
        config.setdefault("robot", {})["use_mock"] = True

    env = FrankaVicHarvestEnv(
        config,
        policy.action_bounds,
        policy.max_episode_steps,
        cage_pos_m=args.cage_pos_m,
        cage_rot_rad=args.cage_rot_rad,
        control_rate_hz=args.control_rate_hz,
        quat_hint_xyzw=policy.training_quat_mean_xyzw,
    )
    log: dict[str, list] = {"obs": [], "action": [], "vic_action": [], "t": []}
    try:
        if not args.mock:
            input("Gripper free and open, nothing touching it? Enter to calibrate F/T... ")
        print(f"F/T bias: {np.round(env.calibrate_ft_bias(), 3)}")
        if not args.mock:
            input("Now grasp the apple (field_session grasp / gripper_test close). Enter to start the policy... ")
        obs, _ = env.reset()
        ood = policy.out_of_distribution(obs)
        if ood:
            print("Start pose is outside the training distribution (|z| > 3):")
            for name, value, z in ood:
                print(f"  {name:12s} = {value:+.4f}  (z = {z:+.1f})")
            if not args.allow_ood:
                print("Refusing to run; reposition the arm or pass --allow-ood.")
                return 2
        policy.reset()
        t0 = time.monotonic()
        for t in range(args.steps):
            action = policy.act(torch.as_tensor(obs, dtype=torch.float32))
            log["obs"].append(obs)
            log["action"].append(action.numpy())
            obs, _reward, terminated, truncated, info = env.step(action.numpy())
            log["vic_action"].append(info["vic_action"])
            log["t"].append(time.monotonic() - t0)
            print(
                f"step {t:4d} action={np.round(action.numpy(), 3)} "
                f"tcp_pos={np.round(obs[:3], 4)} ft={np.round(obs[13:19], 2)}"
            )
            if terminated or truncated:
                print(f"episode ended: terminated={terminated} truncated={truncated} "
                      f"{info.get('safety_violation', '')}")
                break
    finally:
        env.close()
        if args.log and log["t"]:
            np.savez(args.log, **{k: np.asarray(v) for k, v in log.items()})
            print(f"wrote {len(log['t'])} steps to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
